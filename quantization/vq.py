# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Residual vector quantizer implementation."""

from dataclasses import dataclass, field
import math
import typing as tp
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .core_vq import ResidualVectorQuantization, Multiscale_ResidualVectorQuantization


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
        if you want to know more information about RVQ, you can read the soundstream paper (https://arxiv.org/abs/2107.03312)
        Residual vector quantizer cascades N_q layers of VQ. 
        the algorithm is described as follows:
        **********************************************************************************
        Input: y = enc(x) the output of the encoder, vector quantizers Q_i for i = 1...N_q
        Output: the quantized y^hat
        
        y^hat <- 0 
        residual <- y
        for i=1 to N_q do
            y^hat += Q_i(residual)
            residual -= Q_i(residual)
        return y^hat

        **********************************************************************************
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        latent_dim: int = 32,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        multi_scale=None,
        phi_kernel=None,
    ):
        super().__init__()
        self.n_q = n_q
        print(f"creating {n_q} quantizer")
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = Multiscale_ResidualVectorQuantization(
                scale=multi_scale,
                phi_kernel=phi_kernel,
                dim=self.dimension,
                latent_dim=latent_dim,
                codebook_size=self.bins,
                num_quantizers=self.n_q,
                decay=self.decay,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                threshold_ema_dead_code=self.threshold_ema_dead_code
            )

    def forward(self, x: torch.Tensor):
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        quantized, codes, commit_loss = self.vq(x)
        return quantized, codes, torch.mean(commit_loss)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        codes = self.vq.encode(x) # vq.encode output -> out_indices
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.vq.decode(codes) # vq.decode output -> quantized_out
        return quantized

    def idxBl_to_var_input(self, label_list):
        return self.vq.idx_to_var_input(label_list)
    
    def post_conv(self, fhat):
        return self.vq.post_conv(fhat)

    def embedding(self, idx_Bl, layer_id):
        return self.vq.embedding(idx_Bl, layer_id)
    
    def get_next_autoregressive_input(self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        return self.vq.get_next_autoregressive_input(si, f_hat, h_BChw)
    
    def decode_each_scale(self, q_indices) -> List[torch.Tensor]:
        return self.vq.decode_each_scale(q_indices)