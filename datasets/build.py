import torch 
from .audioset import AudioSet



def create_dataset(dataset_name, args, split='train'):

    dataset = AudioSet(args, transform=None, mode=split)

    return dataset