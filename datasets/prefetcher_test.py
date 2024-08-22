import logging
import random
from contextlib import suppress
from functools import partial
from itertools import repeat
from typing import Callable, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.utils.data
import numpy as np

class PrefetchLoader_split:

    def __init__(
            self,
            loader,
            tensor_cut, 
            device=torch.device('cuda')):
        self.loader = loader
        self.device = device
        self.tensor_cut = tensor_cut
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
                next_input = next_input.to(device=self.device, dtype=torch.float32, non_blocking=True)
                next_input = rearrange(next_input, 'b (c n s) -> (b n) c s', s=self.tensor_cut, c=1)

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