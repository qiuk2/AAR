import os, sys
import argparse, glob
import math
import random

import numpy as np
from time import time
from datetime import datetime
from tqdm.auto import tqdm
import wandb

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from transformers import get_scheduler

from datasets import create_dataset, PrefetchLoader
from model import SAT, build_aar
from ruamel.yaml import YAML
from utils import seed_everything

def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")

    # data
    parser.add_argument("--clap_process", type=bool, default=True)
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--test_dir", type=str, default='/voyager/AudioSet/audioset_unbalanced_train_mp3', help="data folder")
    parser.add_argument("--dataset_name", type=str, default="audioset", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=8, help="per gpu batch size")
    parser.add_argument("--tensor_cut", type=int, default=24000)
    parser.add_argument("--fixed_length", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=12, help="batch size")
    parser.add_argument("--use_prefetcher", type=bool, default=False)


    # training
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='lin0', help='lr scheduler')
    parser.add_argument("--lr_warmup_steps", type=float, default=0.03, help="warmup steps")
    parser.add_argument("--log_interval", type=int, default=500, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=str, default='3000', help='save interval')
    parser.add_argument("--mixed_precision", type=str, default='bf16', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')
    parser.add_argument("--clip", type=float, default=2., help='gradient clip, set to -1 if not used')
    parser.add_argument("--wp0", type=float, default=0.005, help='initial lr ratio at the begging of lr warm up')
    parser.add_argument("--wpe", type=float, default=0.01, help='final lr ratio at the end of training')
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--weight_decay_end", type=float, default=0, help='final lr ratio at the end of training')
    parser.add_argument("--resume", type=str, default=False, help='resume')
    parser.add_argument("--ignore_mask", type=bool, default=False, help='ignore_mask')
    parser.add_argument("--val_only", type=bool, default=False, help='validation only')
    parser.add_argument("--cfg", type=float, default=4, help='cfg guidance scale')
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gibbs", type=int, default=0, help='use gibbs sampling during inference')
    parser.add_argument("--save_val", type=bool, default=False, help='save val images')
    
    # audio-vqvae
    parser.add_argument('--vqvae_pretrained_path', type=str, default=None)
    parser.add_argument('--sample_rate', type=int, default=24000)
    parser.add_argument('--window', type=float, default=1)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--model_norm', type=str, default='weight_norm')
    parser.add_argument('--audio_normalize', type=bool, default=False)
    parser.add_argument('--name', type=str, default='audiovae')
    parser.add_argument('--ratios', nargs='+', type=int, default=[8, 5, 4, 2])
    parser.add_argument('--multi_scale', nargs='+', type=int, default=None)
    parser.add_argument('--phi_kernel', nargs='+', type=int, default=None)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    parser.add_argument('--lstm', type=int, default=2)
    # vpq model
    parser.add_argument("--aar_pretrained_path", type=str, default=None)
    parser.add_argument("--depth", type=int, default=16, help="depth of vpq model")
    parser.add_argument("--embed_dim", type=int, default=128, help="embedding dimension of vpq model")
    parser.add_argument("--cos_attn", type=bool, default=False, help="weather to cos attention")
    parser.add_argument("--num_heads", type=int, default=16, help="number of heads of vpq model")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="mlp ratio of vpq model")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="drop rate of vpq model")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="attn drop rate of vpq model")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="drop path rate of vpq model")
    parser.add_argument("--mask_type", type=str, default='interleave_append', help="[interleave_append, replace]")
    parser.add_argument("--uncond", type=bool, default=False, help="uncond gen")
    parser.add_argument("--type_pos", type=bool, default=False, help="use type pos embed")
    parser.add_argument("--interpos", type=bool, default=False, help="interpolate positional encoding")
    parser.add_argument("--mpos", type=bool, default=False, help="minus positional encoding")
    # condition model
    parser.add_argument("--condition_model", type=str, default='clap_embedder', help="condition model")
    parser.add_argument("--cond_drop_rate", type=float, default=0.1, help="drop rate of condition model")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

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


@torch.no_grad()
def inference(var, vqvae, cond_model, conditions, guidance_scale=4.0, top_k=900, top_p=0.95, seed=42):
    if cond_model:
        cond_model.eval()
    # conditions = [474, 474, 474, 474]
    audios = var.autoregressive_infer_cfg(B=len(conditions), label_B=conditions,
                                          cfg=guidance_scale, top_k=top_k, top_p=top_p, g_seed=seed)
    # audios = audios.squeeze(1).float().cpu().numpy()

    return audios
    # if cond_model:
    #     cond_model.train()

def resume(var, optimizer, args):
    state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
    if 'model_state_dict' in state_dict.keys():
        var_state_dict = state_dict['model_state_dict']

        var.load_state_dict(var_state_dict, strict=True)

    if 'optimizer_state_dict' in state_dict.keys():
        opt_state_dict = state_dict['optimizer_state_dict']
        optimizer.load_state_dict(opt_state_dict)

    args.completed_steps = (state_dict['epoch']+1) * args.num_update_steps_per_epoch
    args.starting_epoch = state_dict['epoch']

    if 'latest' not in args.resume:
        args.starting_epoch += 1

    print(f'Resume from step: {args.completed_steps}, epoch: {args.starting_epoch}')

def process(args):
    device = torch.device(f"cuda")
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_dir = './wandb'
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    os.environ["WANDB_CONFIG_DIR"] = './wandb'
    os.environ["WANDB_CACHE_DIR"] = './wandb'
    os.environ["WANDB_DIR"] = './wandb'
    wandb.login()
    wandb.init(project="Evaluation")

    # Setup accelerator:
    if args.run_name is None:
        model_name = f'audio-vae_var_d{args.depth}e{args.embed_dim}h{args.num_heads}_{args.dataset_name}_ep{args.num_epochs}_bs{args.batch_size}_clip{args.clip}'
    else:
        model_name = args.run_name

    args.model_name = model_name
    args.embed_dim = args.depth * 64
    timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')

    if args.save_interval is not None and args.save_interval.isdigit():
        args.save_interval = int(args.save_interval)

    # create dataset
    print(f"Creating dataset {args.dataset_name}")
    dataset = create_dataset('audioset', args, split='test')
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if args.use_prefetcher:
        dataloader = PrefetchLoader(dataloader, device=device)
    # val_sampler = DistributedSampler(val_dataset, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size,
    #                         num_workers=args.num_workers, pin_memory=True, drop_last=False)
    # Calculate total batch size
    total_batch_size = args.batch_size * args.gpus * args.gradient_accumulation_steps
    args.total_batch_size = total_batch_size

    # Create VQVAE Model
    print("Creating VQVAE model")
    vqvae = SAT(
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

    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    if args.vqvae_pretrained_path is not None:
        state_dict = torch.load(args.vqvae_pretrained_path, map_location=torch.device('cpu'))['generator_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        vqvae.load_state_dict(new_state_dict, strict=True)

        print(f'load from ckpt: {args.vqvae_pretrained_path}')
    # Create VPA Model
    print("Creating AAR model")
    aar = build_aar(vae=vqvae, depth=args.depth, patch_nums=args.multi_scale, cos_attn=args.cos_attn)

    aar = aar.to(device)

    if args.aar_pretrained_path is not None:
        state_dict = torch.load(args.aar_pretrained_path, map_location=torch.device('cpu'))['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        aar.load_state_dict(new_state_dict, strict=True)

        print(f'load from ckpt: {args.aar_pretrained_path}')
    
    aar.eval()

    for p in aar.parameters():
        p.requires_grad_(False)

    # Create Condition Model
    print("Creating conditional model")
    if args.condition_model is None:
        cond_model = None
    elif args.condition_model == 'clap_embedder':
        from transformers import ClapModel
        cond_model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
        cond_model.eval()
        for p in cond_model.parameters():
            p.requires_grad_(False)
    else:
        raise NotImplementedError(f"Condition model {args.condition_model} is not implemented")

    print("***** Test arguments *****")
    print(args)
    
    # Only show the progress bar once on each machine.
    os.makedirs(args.output_dir, exist_ok=True)
    for batch_idx, (input_wav, cond) in enumerate(tqdm(dataloader)):
        cond = cond_model.get_audio_features(**cond)
        output_wav = inference(aar, vqvae, cond_model, cond, guidance_scale=args.cfg, top_k=args.top_k, top_p=args.top_p)
        output_wav = output_wav.cpu()
        wandb.log({f"output_audio": wandb.Audio(output_wav[0].squeeze(0).numpy(), sample_rate=24000) })
        for i in range(input_wav.shape[0]):
            torchaudio.save(os.path.join(f'{args.output_dir}', f'{batch_idx*args.batch_size+i}.wav'), output_wav[i], sample_rate=args.sample_rate)

if __name__ == '__main__':
    args = parse_args()
    process(args)