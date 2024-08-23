# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn

import quantization as qt
from modules import SEANetEncoder, SEANetDecoder
import random

class SAT(nn.Module):
    def __init__(self,
                 sample_rate: int = 24_000,
                 channels: int = 1,
                 causal: bool = True,
                 model_norm: str = 'weight_norm',
                 audio_normalize: bool = False,
                 ratios=[8, 5, 4, 2],
                 multi_scale=None,
                 phi_kernel=None,
                 dimension=128,
                 latent_dim=32,
                 n_residual_layers=1,
                 lstm=2):
        super().__init__()
        self.encoder = SEANetEncoder(channels=channels, dimension=dimension, norm=model_norm, causal=causal,ratios=ratios, n_residual_layers=n_residual_layers, lstm=lstm)
        self.decoder = SEANetDecoder(channels=channels, dimension=dimension, norm=model_norm, causal=causal,ratios=ratios, n_residual_layers=n_residual_layers, lstm=lstm)
        self.quantizer = qt.ResidualVectorQuantizer(
            dimension=dimension,
            n_q=len(multi_scale),
            bins=1024,
            latent_dim=latent_dim,
            multi_scale=multi_scale,
            phi_kernel=phi_kernel
        )


    def forward(self, x: torch.Tensor):
        e = self.encoder(x)
        quant, code, vq_loss = self.quantizer(e)
        output = self.decoder(quant)
        return output, code, vq_loss
    
    def fhat_to_audio(self, fhat):
        return self.decoder(self.quantizer.post_conv(fhat.permute(0, 2, 1)))
    

    def idxBl_to_h(self, labels_list):
        return self.quantizer.idxBl_to_var_input(labels_list)

    def audio_to_idxBl(self, x):
        length = x.shape[-1] # tensor_cut or original
        emb = self.encoder(x) # [2,1,10000] -> [2,128,32]
        if self.training:
            return emb,scale
        codes = self.quantizer.encode(emb)
        return codes
