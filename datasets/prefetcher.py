import logging
import random
from contextlib import suppress
from functools import partial
from itertools import repeat
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.data
import numpy as np

def fast_collate(batch):
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


class PrefetchLoader:

    def __init__(
            self,
            loader,
            device=torch.device('cuda')):
        self.loader = loader
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_input, next_condition in self.loader:

            with stream_context():
                next_input = next_input.to(device=self.device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
                for k, v in next_condition.items():
                    if v.dtype != torch.bool:
                        next_condition[k] = v.to(device=self.device, dtype=torch.float32, non_blocking=True).squeeze(1)
                    else:
                        next_condition[k] = v.to(device=self.device, non_blocking=True)

            if not first:
                yield input, cond
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input, cond = next_input, next_condition

        yield input, cond

            
    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class Pref_wo_cond_Loader:

    def __init__(
            self,
            loader,
            device=torch.device('cuda')):
        self.loader = loader
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_input in self.loader:

            with stream_context():
                next_input = next_input.to(device=self.device, dtype=torch.float32, non_blocking=True).unsqueeze(1)

            if not first:
                yield input
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input = next_input

        yield input

            
    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
