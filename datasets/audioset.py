import glob
import os
import random
import pickle

import librosa
import torchaudio
import numpy as np
import pandas as pd
import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .transform import NormalizeDB, RandomGainDB, NormalizeWave
from transformers import ClapProcessor


class AudioSet(Dataset):
    def __init__(self, config, transform=None,mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        self.mode = mode
        if mode == 'train':
            if os.path.exists(config.train_csv.replace('.csv', '_filter.csv')):
                self.audio_files = pd.read_csv(config.train_csv.replace('.csv', '_filter.csv'), delimiter=', ', skiprows=0, engine='python')
                self.train_dir = config.train_dir
            else:
                self.audio_files = pd.read_csv(config.train_csv, delimiter=', ', skiprows=2, engine='python')
                self.train_dir = config.train_dir
                self.check_files(config.train_csv.replace('.csv', '_filter.csv'))

            print(self.audio_files)
        elif mode == 'test':
            self.audio_files = sorted(glob.glob(os.path.join(config.test_dir,"*.mp3")))
            self.test_dir = config.test_dir
        
        self.transform = transform
        self.fixed_length = config.fixed_length
        self.tensor_cut = config.tensor_cut
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        if config.clap_process:
            self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        else:
            self.processor = None
        
    # check if self.train_csv has all files
    def check_files(self, csv_path):
        original_count = len(self.audio_files)
        # Applying tqdm to the apply function for progress tracking
        tqdm.pandas(desc="Checking file existence")

        def check_existence(ytid):
            file_path = os.path.join(self.train_dir, f"Y{ytid}.mp3")
            # if os.path.exists(file_path):
            #     print(f"File exists: {file_path}")
            return os.path.exists(file_path)

        self.audio_files['file_exists'] = self.audio_files['# YTID'].progress_apply(check_existence)

        # # Optionally filter the DataFrame to keep only those files that exist
        self.audio_files = self.audio_files[self.audio_files['file_exists']]
        with open(csv_path, 'w') as f:
            # Write the header
            f.write(", ".join(self.audio_files.columns) + "\n")
            
            # Write each row
            for index, row in self.audio_files.iterrows():
                f.write(", ".join(f'"{str(x)}"' if ',' in str(x) or '"' in str(x) else str(x) for x in row) + "\n")

        print(f"save filtered csv to {csv_path}")


    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)
    

    def __getitem__(self, idx):
        if self.mode == 'train':
            waveform, _ = librosa.load(os.path.join(self.train_dir, f'Y{self.audio_files.iloc[idx, 0]}.mp3'), 
            sr=self.sample_rate, mono=self.channels)
        else:
            waveform, _ = librosa.load(self.audio_files[idx], sr=self.sample_rate, mono=self.channels)
            self.tensor_cut = 240000

        if self.tensor_cut > 0 and waveform.size > self.tensor_cut:
            start = random.randint(0, waveform.size - self.tensor_cut - 1)
            waveform = waveform[start:start + self.tensor_cut]
        else:
            padding_needed = self.tensor_cut - waveform.size
            waveform = np.pad(waveform, (0, padding_needed), 'constant')

        if self.transform:
            waveform = self.transform(waveform)
        
        if self.processor == None:
            return torch.from_numpy(waveform)
        
        resample_wave, _ = librosa.load(os.path.join(self.train_dir, f'Y{self.audio_files.iloc[idx, 0]}.mp3'),sr=48000, mono=self.channels)
        condition = self.processor(audios=resample_wave, return_tensors="pt", sampling_rate=48000)
        
        return torch.from_numpy(waveform), condition


    # def __getitem__(self, idx):
    #     waveform,sample_rate = librosa.load(os.path.join(self.train_dir, f'Y{self.audio_files.iloc[idx, 0]}.mp3'),sr=self.sample_rate)
    #     waveform = torch.as_tensor(waveform).unsqueeze(0)

    #     if self.tensor_cut > 0:
    #         if waveform.size()[1] > self.tensor_cut:
    #             start = random.randint(0, waveform.size()[1]-self.tensor_cut-1) # random start point
    #             waveform = waveform[:, start:start+self.tensor_cut] # cut tensor
        
    #     if self.transform:
    #         waveform, mean, std = self.transform(waveform)
        
    #     return waveform, mean, std
        

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    # print(batch[0].shape)
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors, means, stds = [], [], []

    for waveform, mean, std in batch:
        tensors += [waveform]
        means += [mean]
        stds += [std]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors, means, stds
