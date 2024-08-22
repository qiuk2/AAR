# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""
from typing import List, Optional, Sequence, Tuple, Union
import typing as tp
import warnings

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np

# import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    """ema update parameter. moving_avg = moving_avg + (1-decay) * new
    Args:
        moving_avg (_type_): 
        new (_type_): update parameter
        decay (float): update rate
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-3):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        # distrib.broadcast_tensors(self.buffers()) # FIXME: this is not working for some reason

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        # distrib.broadcast_tensors(self.buffers()) # FIXME: this is not working for some reason

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        ) # get the distance between x and embed
        embed_ind = dist.max(dim=-1).indices # get the index of the closest embed
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x) # [2,32,128] -> [64,128]

        self.init_embed_(x) # to better initialize the codebook

        embed_ind = self.quantize(x) # get the index of the closest embed
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training: # update the codebook
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size, the number of vectors in the codebook
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
                            the dimension of each vector in the codebook
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x, scale=None):
        # x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        if scale != None:
            x = F.interpolate(x.permute(0, 2, 1), size=scale, mode='area').permute(0, 2, 1)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind, H=None, conv=None):
        quantize = self._codebook.decode(embed_ind)
        quantize = F.interpolate(quantize.permute(0, 2, 1), size = H, mode='linear') if H != None else quantize.permute(0, 2, 1)
        quantize = conv(quantize).permute(0, 2, 1) if conv != None else quantize.permute(0,2,1)
        quantize = self.project_out(quantize)
        # quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, scale=None, conv=None):
        device = x.device
        H = x.shape[1]
        x = self.project_in(x) # self.project_in(x.permute(0, 2, 1))
        inter_x = F.interpolate(x.permute(0, 2, 1), size=scale, mode='area').permute(0, 2, 1) if scale != None else x
        quantize, embed_ind = self._codebook(inter_x)
        quantize = F.interpolate(quantize.permute(0, 2, 1), size = H, mode='linear').contiguous() if scale != None else quantize.permute(0, 2, 1).contiguous()
        quantize = conv(quantize).permute(0, 2, 1) if conv != None else quantize.permute(0,2,1)

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            warnings.warn('When using RVQ in training model, first check '
                          'https://github.com/facebookresearch/encodec/issues/25 . '
                          'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                # x = F.normalize(x)  
                # quantize = F.normalize(quantize)  
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight
            if conv != None:
                loss = loss + F.mse_loss(quantize, x.detach())
            quantize = x + (quantize - x).detach()

        quantize = self.project_out(quantize)
        # quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class Multiscale_ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, scale, phi_kernel, num_quantizers, latent_dim, **kwargs):
        super().__init__()
        print("try to use multi scale quantization")
        self.scale = scale
        self.dim = latent_dim
        self.project_in = nn.Linear(kwargs['dim'], latent_dim) if kwargs['dim'] != latent_dim else nn.Identity()
        self.project_out = nn.Linear(latent_dim, kwargs['dim']) if kwargs['dim'] != latent_dim else nn.Identity()
        kwargs['dim'] = latent_dim
        print(f"scale: {self.scale}")
        print(f"phi kernel: {phi_kernel}")
        self.quant_resi = PhiPartiallyShared(nn.ModuleList([Phi(latent_dim, 0.5, ks=ks) for ks in phi_kernel]))
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x):

        B, C, H = x.shape
        quantized_out = 0.0
        residual = self.project_in(x.permute(0, 2, 1))

        all_losses = []
        all_indices = []

        n_q = len(self.layers)

        for idx in range(n_q):
            layer = self.layers[idx]
            quantized, indices, loss = layer(residual, self.scale[idx] if self.scale[idx] != H else None, self.quant_resi[idx/(n_q-1)])
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized # y^hat

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses = torch.stack(all_losses)
        output = self.project_out(quantized_out).permute(0, 2, 1)
        return output, all_indices, out_losses

    def encode(self, x: torch.Tensor):
        B, C, H = x.shape
        residual = self.project_in(x.permute(0, 2, 1))
        all_indices = []
        n_q = len(self.scale)
        for idx in range(n_q):
            layer = self.layers[0] if self.shared_codebook else self.layers[idx]
            indices = layer.encode(residual, self.scale[idx] if self.scale[idx] != H else None)
            quantized = layer.decode(indices, H if self.scale[idx] != H else None, self.quant_resi[idx/(n_q-1)])
            residual = residual - quantized
            all_indices.append(indices)

        return all_indices

    def decode_each_scale(self, q_indices) -> List[torch.Tensor]:
        quantized_out = torch.tensor(0.0, device=q_indices[0].device)
        n_q = len(self.scale)
        scale = []
        for i, indices in enumerate(q_indices):
            layer = self.layers[0] if self.shared_codebook else self.layers[i]
            quantized = layer.decode(indices, max(self.scale) if self.scale[i] != 75 else None, self.quant_resi[i/(n_q-1)])
            quantized_out = quantized_out + quantized
            scale.append(self.project_out(quantized_out).permute(0, 2, 1))
        return scale

    def decode(self, q_indices) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices[0].device)
        n_q = len(self.scale)
        for i, indices in enumerate(q_indices):
            layer = self.layers[0] if self.shared_codebook else self.layers[i]

            quantized = layer.decode(indices, self.scale[-1] if i != n_q-1 else None, self.quant_resi[i/(n_q-1)])
            quantized_out = quantized_out + quantized

        return self.project_out(quantized_out).permute(0, 2, 1)
    
    def embedding(self, idx_Bl, layer_id):
        return self.layers[layer_id].decode(idx_Bl).permute(0, 2, 1)
    
    def post_conv(self, fhat):
        return self.project_out(fhat).permute(0, 2, 1)
   
    def get_next_autoregressive_input(self, si: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        H = self.scale[-1]
        SN = len(self.scale)
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=H, mode='linear'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=self.scale[si+1], mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat

    def idx_to_var_input(self, label_list):

        next_scales = []
        B = label_list[0].shape[0]
        C = self.dim
        H = self.scale[-1]
        SN = len(self.scale)

        with torch.autocast(device_type=label_list[0].device.type, enabled=False):
            f_hat = label_list[0].new_zeros(B, C, H, dtype=torch.float32)
            pn_next: int = self.scale[0]
            for si in range(SN-1):
                layer = self.layers[si] if not self.shared_codebook else self.layers[0]
                f_hat.add_(layer.decode(label_list[si], H, self.quant_resi[si/(SN-1)]).permute(0, 2, 1))
                pn_next = self.scale[si+1]
                next_scales.append(F.interpolate(f_hat, size=(pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        
        return next_scales



class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi, ks):
        padding = (ks // 2)  # Adjust padding based on kernel size and dilation
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=padding)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'