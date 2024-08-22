import os
import argparse
import math
import random
import itertools

from ruamel.yaml import YAML
import numpy as np

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from utils import seed_everything
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from datasets import create_dataset, Pref_wo_cond_Loader as PrefetchLoader
import wandb
from transformers import get_scheduler

from model import SAT, MultiScaleSTFTDiscriminator
from losses import total_loss, disc_loss

def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")

    # data
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument("--train_dir", type=str, default='/ceph/AudioSet/audioset_unbalanced_train_mp3', help="data folder")
    parser.add_argument("--train_csv", type=str, default='/ceph/AudioSet/unbalanced_train_segments.csv')
    parser.add_argument("--dataset_name", type=str, default="audioset", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="per gpu batch size")
    parser.add_argument("--tensor_cut", type=int, default=24000)
    parser.add_argument("--fixed_length", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--use_prefetcher", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')


    # training
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
    parser.add_argument("--lr_scheduler", type=str, default='cos', help='lr scheduler')
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--weight_decay_end", type=float, default=0, help='final lr ratio at the end of training')
    parser.add_argument("--wp0", type=float, default=0.005, help='initial lr ratio at the begging of lr warm up')
    parser.add_argument("--wpe", type=float, default=0.01, help='final lr ratio at the end of training')
    parser.add_argument("--lr_warmup_steps", type=float, default=0.03, help="warmup steps")
    parser.add_argument("--log_interval", type=int, default=500, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=str, default='epoch', help='save interval')
    parser.add_argument("--mixed_precision", type=str, default='bf16', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--clip", type=float, default=1, help='gradient clip, set to -1 if not used')
    parser.add_argument("--resume", type=str, default=False, help='resume')
    parser.add_argument("--clap_process", type=bool, default=False)



    # audio-vqvae
    parser.add_argument('--vae_pretrained_path', type=str, default=None)
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
    parser.add_argument('--lstm', type=int, default=2)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    # discriminator
    parser.add_argument("--filters", type=int, default=32, help="filter for disc")
    parser.add_argument('--disc_win_lengths', nargs='+', type=int, default=[1024, 2048, 512])
    parser.add_argument('--disc_hop_lengths', nargs='+', type=int, default=[256, 512, 128])
    parser.add_argument('--disc_n_ffts', nargs='+', type=int, default=[1024, 2048, 512])
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

def setup(args):
    args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    args.gpus = args.world_size
    args.gpu = int(os.environ['OMPI_COMM_WORLD_RANK'])
    dist.init_process_group(backend='nccl', rank=args.rank, world_size=args.world_size)
    # dist.barrier()

def cleanup():
    dist.destroy_process_group()


def train_epoch(audiovae, disc, dataloader, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, progress_bar, rank, args):
    device = audiovae.device
    audiovae.train()
    disc.train()

    train_generator_loss = []
    train_discriminator_loss = []

    for batch_idx, batch in enumerate(dataloader):

        if not args.use_prefetcher:
            input_wav = batch.to(device)
            input_wav = input_wav.unsqueeze(1)
        else:
            input_wav = batch
        input_wav = input_wav.contiguous()
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer_G.zero_grad()
        if args.mixed_precision == 'bf16':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_wav, commit_loss, _ = audiovae(input_wav)
                logits_real, fmap_real = disc(input_wav)
                logits_fake, fmap_fake = disc(output_wav)
                loss_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=args.sample_rate)
        else:
            output_wav, commit_loss, _ = audiovae(input_wav)
            logits_real, fmap_real = disc(input_wav)
            logits_fake, fmap_fake = disc(output_wav)
            loss_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=args.sample_rate)
        
        generator_loss = 3*loss_g['l_g'] + 3*loss_g['l_feat'] + loss_g['l_t']/10 + loss_g['l_f'] + commit_loss
        generator_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(audiovae.parameters(), args.clip)

        optimizer_G.step()
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer_D.zero_grad()
        update_disc = (batch_idx % 3 != 0)
        if update_disc:
            if args.mixed_precision == 'bf16':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits_real, _ = disc(input_wav)
                    logits_fake, _ = disc(output_wav.detach().contiguous())
                    discriminator_loss = disc_loss(logits_real, logits_fake)
            else:
                logits_real, _ = disc(input_wav)
                logits_fake, _ = disc(output_wav.detach().contiguous())
                discriminator_loss = disc_loss(logits_real, logits_fake)
        
            discriminator_loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(disc.parameters(), args.clip)
            
            optimizer_D.step()

        lr_scheduler_G.step()
        lr_scheduler_D.step()
        progress_bar.set_description(f"train/loss: {generator_loss.item()}")
        args.completed_steps += 1
        progress_bar.update(1)

        if rank==0:
            if args.completed_steps % args.log_interval == 0:
                loss_data = {
                    **{f"train_generator/{key}": value.item() for key, value in loss_g.items()},
                    "train_disc": discriminator_loss.item() if update_disc else np.nan,
                    "commit_loss": commit_loss.item(),
                    "lr": optimizer_G.param_groups[0]['lr']
                }
                wandb.log(
                    loss_data,
                    step = args.completed_steps
                )
                input_audio = input_wav[0].flatten().float().cpu().numpy()
                output_audio = output_wav[0].flatten().float().cpu().detach().numpy()
                wandb.log({"input_audio": wandb.Audio(input_audio, sample_rate=args.sample_rate, caption="Input Audio")})
                wandb.log({"output_audio": wandb.Audio(output_audio, sample_rate=args.sample_rate, caption="Output Audio")})


def save_checkpoint(gen_model, disc_model, optimizer_G, optimizer_D, epoch, step, save_dir):
    checkpoint = {
        'generator_state_dict': gen_model.state_dict(),
        'disc_state_dict': disc_model.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        # 'scheduler_G_state_dict': scheduler_G.state_dict(),
        # 'scheduler_D_state_dict': scheduler_D.state_dict(),
        'epoch': epoch,
        'step': step
    }
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
        args.resume = args.vae_pretrained_path
        print(f"find the pth", args.resume)
        return
    latest_file = max(checkpoint_files, key=lambda x: extract_step(x))
    args.resume = os.path.join(args.output_dir, latest_file)
    print(f"find the pth", args.resume)
    return



def resume(audiovae, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, args):
    state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
    if 'generator_state_dict' in state_dict.keys():
        generator_state_dict = state_dict['generator_state_dict']
        audiovae.load_state_dict(generator_state_dict, strict=True)

    if 'disc_state_dict' in state_dict.keys():
        disc_state_dict = state_dict['disc_state_dict']
        discriminator.load_state_dict(disc_state_dict)
    
    if 'optimizer_G_state_dict' in state_dict.keys():
        optim_G_state_dict = state_dict['optimizer_G_state_dict']
        optimizer_G.load_state_dict(optim_G_state_dict)

    if 'optimizer_D_state_dict' in state_dict.keys():
        optim_D_state_dict = state_dict['optimizer_D_state_dict']
        optimizer_D.load_state_dict(optim_D_state_dict)

    args.completed_steps = (state_dict['epoch']+1) * args.num_update_steps_per_epoch
    args.starting_epoch = state_dict['epoch']
    
    # Set the last_epoch to completed_steps - 1
    scheduler_G.last_epoch = args.completed_steps - 1
    scheduler_D.last_epoch = args.completed_steps - 1

    # Call step once to update the learning rate according to the completed steps
    scheduler_G.step()
    scheduler_D.step()

    if 'latest' not in args.resume:
        args.starting_epoch += 1

    print(f'Resume from step: {args.completed_steps}, epoch: {args.starting_epoch}')


def process(args):
    setup(args)
    print(f"Running DDP on rank {args.rank}.")
    device = torch.device(f"cuda:{os.environ['OMPI_COMM_WORLD_LOCAL_RANK']}")
    # seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_interval is not None and args.save_interval.isdigit():
        args.save_interval = int(args.save_interval)

    # create dataset
    print(f"Creating dataset {args.dataset_name}")
    dataset = create_dataset('audioset', args, split='train')
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    if args.use_prefetcher:
        dataloader = PrefetchLoader(dataloader, device=device)
    
    total_batch_size = args.batch_size * args.gpus * args.gradient_accumulation_steps
    args.total_batch_size = total_batch_size
    print(f"There are {len(dataloader)} data to train the EnCodec ")

    print('Creating VQVAE model')
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
    )

    discriminator = MultiScaleSTFTDiscriminator(filters=args.filters,
                        hop_lengths=args.disc_hop_lengths,
                        win_lengths=args.disc_win_lengths,
                        n_ffts=args.disc_n_ffts)

    
    audiovae = nn.SyncBatchNorm.convert_sync_batchnorm(audiovae)
    discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    audiovae = DDP(audiovae.to(device))
    discriminator = DDP(discriminator.to(device))


    print("Creating optimizer")
    args.scaled_lr = args.learning_rate # * np.sqrt(total_batch_size / 512)
    params = [p for p in audiovae.parameters() if p.requires_grad]
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]

    optimizer_G = torch.optim.Adam([{'params': params, 'lr': args.scaled_lr}], betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam([{'params': disc_params, 'lr': args.scaled_lr}], betas=(0.5, 0.9))

    # Compute max_train_steps
    args.num_update_steps_per_epoch = len(dataloader)
    args.max_train_steps = args.num_epochs * args.num_update_steps_per_epoch

    # Create Learning Rate Scheduler
    scheduler_G = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer_G,
        num_warmup_steps=args.warmup_epoch*len(dataloader),
        num_training_steps=args.max_train_steps
    )
    scheduler_D = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer_D,
        num_warmup_steps=args.warmup_epoch*len(dataloader),
        num_training_steps=args.max_train_steps
    )

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
            wandb.init(project="AudioVAR")


    # Start training
    if args.rank == 0:
        print("***** Training arguments *****")
        print(args)
        print("***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {args.num_epochs}")
        print(f"  Instantaneous batch size per device = {args.batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Total optimization steps per epoch {args.num_update_steps_per_epoch}")
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
            resume(audiovae, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, args)
            progress_bar.update(args.completed_steps)
    
    # dist.barrier()
    
    for epoch in range(args.starting_epoch, args.num_epochs):

        args.epoch = epoch
        if args.rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        train_epoch(audiovae, discriminator, dataloader, optimizer_G, optimizer_D, scheduler_G, 
                    scheduler_D, progress_bar, args.rank, args)

        if args.save_interval == 'epoch' and args.rank == 0:
            save_checkpoint(audiovae, discriminator, optimizer_G, optimizer_D, epoch, args.completed_steps, args.output_dir)
    
    cleanup()



if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    process(args)
