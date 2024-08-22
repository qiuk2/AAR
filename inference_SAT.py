import os
import sys
import argparse
from glob import glob
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

from ruamel.yaml import YAML
import numpy as np

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

import torch.distributed as dist
from tqdm.auto import tqdm

# from datasets import load_dataset
from datasets import create_dataset, PrefetchLoader_split
import wandb

from model import SAT
from torcheval.metrics import FrechetAudioDistance
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, ScaleInvariantSignalNoiseRatio, ShortTimeObjectiveIntelligibility


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")

    # data
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--train_dir", type=str, default='/voyager/AudioSet/audioset_unbalanced_train_mp3', help="data folder")
    parser.add_argument("--test_dir", type=str, default='')
    parser.add_argument("--train_csv", type=str, default='/voyager/AudioSet/unbalanced_train_segments.csv')
    parser.add_argument("--dataset_name", type=str, default="audioset", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="per gpu batch size")
    parser.add_argument("--tensor_cut", type=int, default=24000)
    parser.add_argument("--fixed_length", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--use_prefetcher", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')


    # training
    parser.add_argument("--train_disc", type=bool, default=False)
    parser.add_argument("--warmup_epoch", type=int, default=0)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--split_run", type=bool, default=False)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help='lr scheduler')
    parser.add_argument("--lr_warmup_steps", type=float, default=0.03, help="warmup steps")
    parser.add_argument("--log_interval", type=int, default=500, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=str, default='epoch', help='save interval')
    parser.add_argument("--mixed_precision", type=str, default='bf16', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--clip", type=float, default=1, help='gradient clip, set to -1 if not used')
    parser.add_argument("--resume", type=str, default=False, help='resume')



    # audio-vqvae
    parser.add_argument('--sample_rate', type=int, default=24000)
    parser.add_argument('--window', type=float, default=1)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--model_norm', type=str, default='weight_norm')
    parser.add_argument('--audio_normalize', type=bool, default=False)
    parser.add_argument('--ratios', nargs='+', type=int, default=[8, 5, 4, 2])
    parser.add_argument('--multi_scale', nargs='+', type=int, default=None)
    parser.add_argument('--phi_kernel', nargs='+', type=int, default=None)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=32)
    # discriminator
    parser.add_argument("--filters", type=int, default=32, help="filter for disc")
    parser.add_argument('--disc_win_lengths', nargs='+', type=int, default=[1024, 2048, 512])
    parser.add_argument('--disc_hop_lengths', nargs='+', type=int, default=[256, 512, 128])
    parser.add_argument('--disc_n_ffts', nargs='+', type=int, default=[1024, 2048, 512])
    parser.add_argument('--clap_process', type=bool, default=False)
    parser.add_argument('--gen_dir', type=str, default='Audio_gen')
    parser.add_argument('--scale', type=bool, default=False)
    # fFirst parse of command-line args to check for config file
    args = parser.parse_args()
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml = YAML(typ='safe')
            with open(args.config, 'r', encoding='utf-8') as file:
                config_args = yaml.load(file)
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()

    return args

def process(args):
    audiovae = SAT(
        args.sample_rate,
        args.channels, 
        causal=False, model_norm=args.model_norm, 
        audio_normalize=args.audio_normalize,
        ratios=args.ratios,
        multi_scale=args.multi_scale,
        phi_kernel=args.phi_kernel,
        dimension=args.dimension,
        latent_dim=args.latent_dim
    ).to(device)
    state_dict = torch.load(args.resume, map_location=torch.device("cpu"))
    if "generator_state_dict" in state_dict.keys():
        generator_state_dict = state_dict['generator_state_dict']
        new_state_dict = {}
        for k, v in generator_state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        audiovae.load_state_dict(new_state_dict, strict=True)

        print(f"resume from checkpoint: {args.resume}")
    
    print("create dataset for inference")
    dataset = create_dataset("audioset", args, split="test")
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dataloader = PrefetchLoader_split(dataloader, device=device, tensor_cut=args.tensor_cut)
    
    audiovae.eval()
    
    for batch_idx, input_wav in enumerate(tqdm(dataloader, desc="Process reconstruction")):
        with torch.no_grad():
            output_wav, _, _ = audiovae(input_wav)

        for idx in range(0, input_wav.shape[0], 240000 // args.tensor_cut):
            output_wavs = torch.zeros((1, 240000))
            for i in range(240000 // args.tensor_cut):
                output_wavs[0, i*args.tensor_cut:(i+1)*args.tensor_cut] = output_wav[idx+i].flatten()
            torchaudio.save(os.path.join(args.gen_dir, f'{batch_idx*args.batch_size+idx//(240000 // args.tensor_cut)}.wav'), output_wavs, sample_rate=args.sample_rate)
                


if __name__ == '__main__':
    args = parse_args()
    process(args)
