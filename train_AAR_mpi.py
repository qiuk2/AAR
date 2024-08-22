import os
import argparse
import math
import random

import numpy as np
from itertools import chain
from time import time
from datetime import datetime
from tqdm.auto import tqdm
import wandb
from PIL import Image
from collections import OrderedDict

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
from model import SAT, AAR, build_aar
from ruamel.yaml import YAML
from utils import seed_everything, lr_wd_annealing, filter_params

def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")

    # data
    parser.add_argument("--clap_process", type=bool, default=True)
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--train_dir", type=str, default='/voyager/AudioSet/audioset_unbalanced_train_mp3', help="data folder")
    parser.add_argument("--train_csv", type=str, default='/voyager/AudioSet/unbalanced_train_segments.csv')
    parser.add_argument("--dataset_name", type=str, default="audioset", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="per gpu batch size")
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
    parser.add_argument("--gibbs", type=int, default=0, help='use gibbs sampling during inference')
    parser.add_argument("--save_val", type=bool, default=False, help='save val images')
    
    # audio-vqvae
    parser.add_argument('--vqvae_pretrained_path', type=str, default=None)
    parser.add_argument('--aar_pretrained_path', type=str, default=None)
    parser.add_argument('--target_bandwidths', nargs='+', type=float, default=[12])
    parser.add_argument('--sample_rate', type=int, default=24000)
    parser.add_argument('--window', type=float, default=1)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--model_norm', type=str, default='weight_norm')
    parser.add_argument('--audio_normalize', type=bool, default=False)
    parser.add_argument('--name', type=str, default='audiovae')
    parser.add_argument('--ratios', nargs='+', type=int, default=[8, 5, 4, 2])
    parser.add_argument('--multi_scale', nargs='+', type=int, default=[1, 2, 3, 5, 8, 12, 21, 30, 50, 75])
    parser.add_argument('--phi_kernel', nargs='+', type=int, default=[9,9,9,9])
    parser.add_argument('--shared_codebook', type=bool, default=True)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    parser.add_argument('--lstm', type=int, default=2)
    # vpq model
    parser.add_argument("--v_patch_layers", type=int, default=[1, 2, 3, 5, 8, 12, 21, 30, 50, 75], help="index of layers for predicting each scale")
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
    parser.add_argument("--input_dim", type=int, default=512)

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


def train_epoch(aar, vqvae, cond_model, dataloader, optimizer, progress_bar, rank, args):
    device = aar.device
    aar.train()
    if cond_model is not None:
        cond_model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    train_loss = []

    for batch_idx, (input_wav, cond) in enumerate(dataloader):

        if not args.use_prefetcher:
            input_wav = input_wav.unsqueeze(1).to(device)
            cond = cond.to(device)

        input_wav = input_wav.contiguous()

        _ = lr_wd_annealing(args.lr_scheduler, optimizer, args.scaled_lr,
                                                             args.weight_decay, args.weight_decay_end,
                                                             args.completed_steps, args.num_warmup_steps,
                                                             args.max_train_steps, wp0=args.wp0, wpe=args.wpe)

        # forward to get input ids
        with torch.no_grad():
            # labels_list: List[(B, 1), (B, 4), (B, 9)]
            labels_list, _ = vqvae.audio_to_idxBl(input_wav)
            # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
            input_h_list = vqvae.idxBl_to_h(labels_list)

            conditions = cond_model.get_audio_features(**cond)

        x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1).to(device)
        # forwad through model
        if args.mixed_precision == 'bf16':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = aar(conditions, x_BLCv_wo_first_l)  # BLC, C=vocab size
        else:
            logits = aar(conditions, x_BLCv_wo_first_l)  # BLC, C=vocab size
        b, l, v = logits.size()
        logits = logits.view(-1, v)
        labels = torch.cat(labels_list, dim=1)
        labels = labels.view(-1)
        loss = loss_fn(logits, labels).view(b, -1)
        loss = loss.mul(1.0 / l).sum(dim=-1).mean()

        # loss = loss.mean()

        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(aar.parameters(), args.clip)

        optimizer.step()
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
            progress_bar.set_description(f"train/loss: {loss.item()}")
            args.completed_steps += 1
            progress_bar.update(1)

        train_loss.append(loss.item())

        if rank == 0:
            # Log metrics
            if args.completed_steps % args.log_interval == 0 and batch_idx % args.gradient_accumulation_steps == 0:
                train_loss_mean = torch.tensor(sum(train_loss) / len(train_loss))  #.to(device)
                # dist.all_reduce(train_loss_mean, op=dist.ReduceOp.SUM)
                wandb.log(
                    {
                        "train/loss": train_loss_mean.item(),
                        "step": args.completed_steps,
                        "epoch": args.epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "weight_decay": optimizer.param_groups[0]["weight_decay"],
                    },
                    step=args.completed_steps)
                text = ["the sound of a cat", "the sound of a dog"]
                inference(aar, vqvae, cond_model, conditions[:4], rank=rank,
                          guidance_scale=4.0, top_k=900, top_p=0.95, seed=42)



@torch.no_grad()
def inference(aar, vqvae, cond_model, conditions, rank=0, guidance_scale=4.0, top_k=900, top_p=0.95, seed=42):
    aar.eval()
    if cond_model:
        cond_model.eval()
    # conditions = [474, 474, 474, 474]
    audios = aar.module.autoregressive_infer_cfg(B=len(conditions), label_B=conditions,
                                          cfg=guidance_scale, top_k=top_k, top_p=top_p, g_seed=seed)
    audios = audios.squeeze(1).float().cpu().numpy()

    wandb.log({f"output_audio": [wandb.Audio(audios[i], sample_rate=24000) for i in range(audios.shape[0])]})

    aar.train()


def setup(args):
    args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    args.gpus = args.world_size
    args.gpu = int(os.environ['OMPI_COMM_WORLD_RANK'])
    dist.init_process_group(backend='nccl', rank=args.rank, world_size=args.world_size)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, args, save_dir='', latest=False):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epoch,
        'step': args.completed_steps
    }
    step = 'latest' if latest else args.completed_steps
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_step_{step}.pth'))


def extract_step(filename):
    # This assumes the file name format is 'checkpoint_step_{step}.pth'
    # and extracts the numeric step part from the file name.
    base_name = os.path.splitext(filename)[0]  # Removes the extension
    step_part = base_name.split('_')[-1]  # Splits the base_name and takes the last part, which should be the step number
    try:
        return int(step_part)  # Converts the step number to an integer
    except ValueError:
        return -1  # In case of any error (e.g., the name does not end in a number), return -1

def find_latest_checkpoint(args):
    checkpoint_files = [f for f in os.listdir(args.output_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        args.resume = args.aar_pretrained_path
        print(f"find the pth", args.resume)
        return
    latest_file = max(checkpoint_files, key=lambda x: extract_step(x))
    args.resume = os.path.join(args.output_dir, latest_file)
    print(f"find the pth", args.resume)
    return

def resume(aar, optimizer, args):
    state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
    if 'model_state_dict' in state_dict.keys():
        aar_state_dict = state_dict['model_state_dict']

        aar.load_state_dict(aar_state_dict, strict=True)

    if 'optimizer_state_dict' in state_dict.keys():
        opt_state_dict = state_dict['optimizer_state_dict']
        optimizer.load_state_dict(opt_state_dict)

    args.completed_steps = (state_dict['epoch']+1) * args.num_update_steps_per_epoch
    args.starting_epoch = state_dict['epoch']

    if 'latest' not in args.resume:
        args.starting_epoch += 1

    print(f'Resume from step: {args.completed_steps}, epoch: {args.starting_epoch}')


def process(args):
    setup(args)
    print(f"Running DDP on rank {args.rank}.")
    device = torch.device(f"cuda:{os.environ['OMPI_COMM_WORLD_LOCAL_RANK']}")
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.rank == 0:
        wandb_dir = './wandb'
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        os.environ["WANDB_CONFIG_DIR"] = './wandb'
        os.environ["WANDB_CACHE_DIR"] = './wandb'
        os.environ["WANDB_DIR"] = './wandb'
        wandb.login()
        if args.debug:
            wandb.init(project="Debug")
        else:
            wandb.init(project="AAR")

    args.model_name = model_name
    args.embed_dim = args.depth * 64
    timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')

    if args.save_interval is not None and args.save_interval.isdigit():
        args.save_interval = int(args.save_interval)

    # create dataset
    print(f"Creating dataset {args.dataset_name}")
    dataset = create_dataset('audioset', args, split='train')
    # create dataloader
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
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
    aar = build_aar(vae=vqvae, input_dim=args.input_dim, depth=args.depth, patch_nums=args.multi_scale, cos_attn=args.cos_attn)

    aar = DDP(aar.to(device), find_unused_parameters=False)
    aar.train()

    print('Filtering parameters')
    names, paras, para_groups = filter_params(aar, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

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

    # Create Optimizer
    print("Creating optimizer")
    # TODO: support faster optimizer

    args.scaled_lr = args.learning_rate # * total_batch_size / 512
    optimizer = torch.optim.AdamW(para_groups, lr=args.scaled_lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    # Compute max_train_steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # Create Learning Rate Scheduler
    args.num_warmup_steps = int(args.wp0 * args.max_train_steps) if args.lr_warmup_steps < 1.0 else int(args.lr_warmup_steps)
    args.num_update_steps_per_epoch = num_update_steps_per_epoch
    # Start training
    if args.rank == 0:
        print("***** Training arguments *****")
        print(args)
        print("***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {args.num_epochs}")
        print(f"  Instantaneous batch size per device = {args.batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        print(f"  Total optimization steps = {args.max_train_steps}")
        print(f"  Scaled learning rate = {args.scaled_lr}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not args.rank == 0)
    args.completed_steps = 0
    args.starting_epoch = 0

    if args.resume:
        if args.resume == 'latest':
            find_latest_checkpoint(args)
        
        if args.resume and os.path.isfile(args.resume) and args.resume.endswith('.pth'):
            resume(aar, optimizer, args)
            progress_bar.update(args.completed_steps)

    for epoch in range(args.starting_epoch, args.num_epochs):

        args.epoch = epoch
        if args.rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}")
        train_epoch(aar, vqvae, cond_model, dataloader, optimizer, progress_bar, args.rank, args)

        if args.save_interval == 'epoch' and args.rank == 0:
            save_checkpoint(aar, optimizer, args, args.output_dir)

    # end training
    cleanup()


if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    process(args)
