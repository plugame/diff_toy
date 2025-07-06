from abc import abstractmethod
import math
from typing import Iterable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,

)

from torch import nn, einsum
from einops import rearrange, repeat

from inspect import isfunction


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=th.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :param dtype: the target data type for the output tensor.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=timesteps.device)
        
        # ここを変更: .float() -> .to(dtype=dtype)
        args = timesteps[:, None].to(dtype=dtype) * freqs[None]
        
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    else:
        # こちらも変更: .to(dtype=dtype) を追加
        embedding = repeat(timesteps, 'b -> b d', d=dim).to(dtype=dtype)
        
    return embedding

def Normalize(in_channels):
    return th.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.heads = heads
        self.dim_head = dim_head  # dim_headをインスタンス変数として保持
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # バッチサイズとヘッド数を取得
        b, n, _ = x.shape
        h = self.heads
        
        # contextがなければ自己注意(self-attention)
        context = x if context is None else context
        n_context = context.shape[1]

        # 1. Query, Key, Valueに行列を適用
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 2. ヘッドごとに分割し、次元を入れ替える
        # (b, n, h*d) -> (b, n, h, d) -> (b, h, n, d)
        q = q.view(b, n, h, self.dim_head).transpose(1, 2)
        k = k.view(b, n_context, h, self.dim_head).transpose(1, 2)
        v = v.view(b, n_context, h, self.dim_head).transpose(1, 2)

        # 3. スケール化ドット積注意を計算
        # QとKの転置を掛けてスコアを算出
        # (b, h, n, d) @ (b, h, d, n_context) -> (b, h, n, n_context)
        scores = th.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = scores.softmax(dim=-1)

        # 4. Attentionを使ってValueの加重和を計算
        # (b, h, n, n_context) @ (b, h, n_context, d) -> (b, h, n, d)
        out = th.matmul(attn, v)

        # 5. ヘッドを結合
        # (b, h, n, d) -> (b, n, h, d) -> (b, n, h*d)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        # 6. 出力層に入力
        return self.to_out(out)
    

class IPAttention(Attention):
    """
    Attentionクラスを継承し、IP-Adapter用の追加のK, Vを生成する機能を追加したクラス。
    """
    def __init__(self, query_dim, context_dim=None, ip_context_dim=None, heads=8, dim_head=64, dropout=0.,ip_scale=1.0):
        # 1. まず親クラス(Attention)の__init__を呼び出し、
        #    to_q, to_k, to_v, to_out などの基本的な層を初期化する
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)

        # 2. IP-Adapter用の新しい層(to_k_ip, to_v_ip)を追加で定義する
        inner_dim = self.to_k.out_features # 親クラスで計算されたinner_dimを取得
        ip_context_dim = ip_context_dim if ip_context_dim is not None else context_dim

        self.to_k_ip = nn.Linear(ip_context_dim, inner_dim, bias=False)
        self.to_v_ip = nn.Linear(ip_context_dim, inner_dim, bias=False)

        self.ip_scale = ip_scale
        print(self.ip_scale)


    def forward(self, x, context=None):
        """
        forwardメソッドをオーバーライド（上書き）して、新しいロジックを組み込む。
        """

        ip_context = context[:,77:,:]
        context = context[:,0:77,:]

        # --- 基本的なQ, K, Vの生成（親クラスと同じ）---
        b, n, _ = x.shape
        h = self.heads
        context = x if context is None else context
        n_context = context.shape[1]

        q = self.to_q(x)
        k = self.to_k(context) # テキストコンテキストから生成
        v = self.to_v(context) # テキストコンテキストから生成

        #----------------
        # 2. ヘッドごとに分割し、次元を入れ替える
        # (b, n, h*d) -> (b, n, h, d) -> (b, h, n, d)
        q = q.view(b, n, h, self.dim_head).transpose(1, 2)
        k = k.view(b, n_context, h, self.dim_head).transpose(1, 2)
        v = v.view(b, n_context, h, self.dim_head).transpose(1, 2)

        scores = th.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = scores.softmax(dim=-1)
        out = th.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        #------------------

        # --- IP-Adapter用のK, Vの追加 ---
        if ip_context.shape[1] > 0:
            n_ip_context = ip_context.shape[1]

            # 画像コンテキストから追加のK, Vを生成
            k_ip = self.to_k_ip(ip_context)
            v_ip = self.to_v_ip(ip_context)

            k_ip = k_ip.view(b, n_ip_context, h,self.dim_head).transpose(1, 2)
            v_ip = v_ip.view(b, n_ip_context, h,self.dim_head).transpose(1, 2)

            ip_scores = th.matmul(q,k_ip.transpose(-1,-2))*self.scale
            ip_attn = ip_scores.softmax(dim=-1)
            ip_out = th.matmul(ip_attn, v_ip)
            ip_out = ip_out.transpose(1, 2).contiguous().view(b, n, -1)

            # 元のK, Vに追加のK_ip, V_ipをシーケンス長(dim=1)の次元で結合する
            #k = th.cat([k, k_ip], dim=1)
            #v = th.cat([v, v_ip], dim=1)
            
            # K, Vのシーケンス長を更新
            #n_context += n_ip_context

        # --- 以降のアテンション計算（親クラスと同じ） ---
        #q = q.view(b, n, h, self.dim_head).transpose(1, 2)
        #k = k.view(b, n_context, h, self.dim_head).transpose(1, 2)
        #v = v.view(b, n_context, h, self.dim_head).transpose(1, 2)

        #scores = th.matmul(q, k.transpose(-1, -2)) * self.scale
        #attn = scores.softmax(dim=-1)

        #out = th.matmul(attn, v)
        #out = out.transpose(1, 2).contiguous().view(b, n, -1)


        out = out + ip_out * self.ip_scale
        
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = Attention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = Attention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  if context_dim is not None else None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        if self.attn2 is not None:
            x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb=None, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                #resnetにはtime_embedsを入れる
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                #transformerにはprompt_embedsを入れる
                x = layer(x, context)
            else:
                #その他はそのまま通す
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

#Downsample(ch, self.conv_resample, dims=self.dims)
class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.op(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = th.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = q.view(b, l, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, l, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, l, self.heads, self.dim_head).transpose(1, 2)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


    
class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(th.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        inner_dim = dim*ff_mult

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        def FeedForward(dim, mult=4):
            inner_dim = int(dim * mult)
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, inner_dim, bias=False),
                nn.GELU(),
                nn.Linear(inner_dim, dim, bias=False),
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim,mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(th.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=th.ones(x.shape[:2], device=x.device, dtype=th.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = th.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)

def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)

class Block(nn.Module):
    def __init__(self,module:nn.Module,input_channel:int,output_channel:int,**kwargs):
        super().__init__()
        self.module = module
        self.inch = input_channel
        self.outch = output_channel
        self.kwargs = kwargs


class UNetModel_SD15_V(nn.Module):
    def __init__(self):
        super().__init__()

        # SDv1.5の固定パラメータをハードコード
        self.model_channels = 320
        self.out_channels = 4
        self.dropout = 0.0
        self.conv_resample = True
        self.transformer_depth = 1
        self.context_dim = 768
        self.num_heads = 8
        self.dtype = th.float16


        input_blocks_list = [
            #in0
            [Block(nn.Conv2d,4,320,kernel_size=3,padding=1)],
            #in1
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #in2
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #3
            [Block(Downsample,320,320)],
            #4
            [Block(ResBlock,320,640), Block(SpatialTransformer,640,640)],
            #5
            [Block(ResBlock,640,640), Block(SpatialTransformer,640,640)],
            #6
            [Block(Downsample,640,640)],
            #7
            [Block(ResBlock,640,1280), Block(SpatialTransformer,1280,1280)],
            #8
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280)],
            #9
            [Block(Downsample,1280,1280)],
            #10
            [Block(ResBlock,1280,1280)],
            #11
            [Block(ResBlock,1280,1280)],
        ]

        output_blocks_list = [
            #in0
            [Block(ResBlock,1280,1280)],
            #in1
            [Block(ResBlock,1280,1280)],
            #in2
            [Block(ResBlock,1280,1280), Block(Upsample,1280,1280)],
            #in3
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280)],
            #in4
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280)],
            #in5
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280), Block(Upsample,1280,1280)],
            #in6
            [Block(ResBlock,1280,640), Block(SpatialTransformer,640,640)],
            #in7
            [Block(ResBlock,640,640), Block(SpatialTransformer,640,640)],
            #in8
            [Block(ResBlock,640,640), Block(SpatialTransformer,640,640), Block(Upsample,640,640)],
            #in9
            [Block(ResBlock,640,320), Block(SpatialTransformer,320,320)],
            #in10
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #in11
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
        ]
        

        # --- 時間埋め込み ---
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        #downblock
        self.input_blocks = nn.ModuleList([])

        # TimestepEmbedSequentialでまとめる
        for _, in_block in enumerate(input_blocks_list):
            layers=[]

            for b in in_block:
                if b.module is nn.Conv2d:
                    layers.append(nn.Conv2d(b.inch,b.outch,**b.kwargs))

                if b.module is ResBlock:
                    layers.append(ResBlock(b.inch, time_embed_dim, self.dropout, out_channels=b.outch))

                elif b.module is SpatialTransformer:
                    dim_head = b.inch // self.num_heads
                    layers.append(SpatialTransformer(b.inch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))

                elif b.module is Downsample:
                    layers.append(Downsample(b.inch, self.conv_resample))

            self.input_blocks.append(TimestepEmbedSequential(*layers))


        #midblock
        ch = input_blocks_list[-1][-1].outch
        dim_head = ch // self.num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
            SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim),
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
        )

        #upblock
        self.output_blocks = nn.ModuleList([])

        for i, out_block in enumerate(output_blocks_list):
            layers = [] 
            skipc = input_blocks_list[-1-i][0].outch #skip connection

            for b in out_block:
                if b.module is ResBlock:
                    layers.append(ResBlock(b.inch+skipc, time_embed_dim, self.dropout, out_channels=b.outch))

                elif b.module is SpatialTransformer:
                    dim_head = b.inch // self.num_heads
                    layers.append(SpatialTransformer(b.inch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))

                elif b.module is Upsample:
                    layers.append(Upsample(b.inch, self.conv_resample))
       
            self.output_blocks.append(TimestepEmbedSequential(*layers))

        
        # --- 出力 ---
        ch = output_blocks_list[-1][-1].outch
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(self.model_channels,self.out_channels,kernel_size=3,padding=1)),
        )

    @classmethod
    def from_pretrained(cls,pretrained_model_state_dict,dtype=None,device=None):
        model = cls()
        model.load_state_dict(pretrained_model_state_dict)

        if dtype is not None:
            model.to(dtype=dtype)
        
        if device is not None:
            model.to(device=device)

        return model


    def forward(
            self, 
            x: th.Tensor,
            timesteps: th.Tensor, 
            encoder_hidden_states: th.Tensor,
            down_block_additional_residuals:list[th.Tensor] = None,
            mid_block_additional_residual:th.Tensor = None
            ) -> th.Tensor:
        h = x.type(self.dtype)

        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        
        # 時間埋め込み
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        emb = self.time_embed(t_emb)

        # Encoder
        skipcs = []
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, encoder_hidden_states)

            if down_block_additional_residuals is not None:
                h = h + down_block_additional_residuals[i]

            skipcs.append(h)
        
        # Bottleneck
        h = self.middle_block(h, emb, encoder_hidden_states)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual

        # Decoder
        for module in self.output_blocks:
            # Skip-connection
            skipc = skipcs.pop()
            h = th.cat([h, skipc], dim=1)
            h = module(h, emb, encoder_hidden_states)
        
        h = h.type(x.dtype)
        return self.out(h)
    

class UNetModel_LTIM(nn.Module):
    def __init__(self):
        super().__init__()

        # SDv1.5の固定パラメータをハードコード
        self.model_channels = 320
        self.out_channels = 4
        self.dropout = 0.0
        self.conv_resample = True
        self.transformer_depth = 1
        self.context_dim = None
        self.num_heads = 8
        self.dtype = th.float16


        input_blocks_list = [
            #in0
            [Block(nn.Conv2d,8,320,kernel_size=3,padding=1)],
            #in1
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #in2
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #3
            [Block(Downsample,320,320)],
            #4
            [Block(ResBlock,320,640), Block(SpatialTransformer,640,640)],
            #5
            [Block(ResBlock,640,640), Block(SpatialTransformer,640,640)],
            #6
            [Block(Downsample,640,640)],
            #7
            [Block(ResBlock,640,1280), Block(SpatialTransformer,1280,1280)],
            #8
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280)],
            #9
            [Block(Downsample,1280,1280)],
            #10
            [Block(ResBlock,1280,1280)],
            #11
            [Block(ResBlock,1280,1280)],
        ]

        output_blocks_list = [
            #in0
            [Block(ResBlock,1280,1280)],
            #in1
            [Block(ResBlock,1280,1280)],
            #in2
            [Block(ResBlock,1280,1280), Block(Upsample,1280,1280)],
            #in3
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280)],
            #in4
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280)],
            #in5
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280), Block(Upsample,1280,1280)],
            #in6
            [Block(ResBlock,1280,640), Block(SpatialTransformer,640,640)],
            #in7
            [Block(ResBlock,640,640), Block(SpatialTransformer,640,640)],
            #in8
            [Block(ResBlock,640,640), Block(SpatialTransformer,640,640), Block(Upsample,640,640)],
            #in9
            [Block(ResBlock,640,320), Block(SpatialTransformer,320,320)],
            #in10
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #in11
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
        ]
        

        # --- 時間埋め込み ---
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        #downblock
        self.input_blocks = nn.ModuleList([])

        # TimestepEmbedSequentialでまとめる
        for _, in_block in enumerate(input_blocks_list):
            layers=[]

            for b in in_block:
                if b.module is nn.Conv2d:
                    layers.append(nn.Conv2d(b.inch,b.outch,**b.kwargs))

                if b.module is ResBlock:
                    layers.append(ResBlock(b.inch, time_embed_dim, self.dropout, out_channels=b.outch))

                elif b.module is SpatialTransformer:
                    dim_head = b.inch // self.num_heads
                    layers.append(SpatialTransformer(b.inch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))

                elif b.module is Downsample:
                    layers.append(Downsample(b.inch, self.conv_resample))

            self.input_blocks.append(TimestepEmbedSequential(*layers))


        #midblock
        ch = input_blocks_list[-1][-1].outch
        dim_head = ch // self.num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
            SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim),
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
        )

        #upblock
        self.output_blocks = nn.ModuleList([])

        for i, out_block in enumerate(output_blocks_list):
            layers = [] 
            skipc = input_blocks_list[-1-i][0].outch #skip connection

            for b in out_block:
                if b.module is ResBlock:
                    layers.append(ResBlock(b.inch+skipc, time_embed_dim, self.dropout, out_channels=b.outch))

                elif b.module is SpatialTransformer:
                    dim_head = b.inch // self.num_heads
                    layers.append(SpatialTransformer(b.inch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))

                elif b.module is Upsample:
                    layers.append(Upsample(b.inch, self.conv_resample))
       
            self.output_blocks.append(TimestepEmbedSequential(*layers))

        
        # --- 出力 ---
        ch = output_blocks_list[-1][-1].outch
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(self.model_channels,self.out_channels,kernel_size=3,padding=1)),
        )

    @classmethod
    def from_pretrained(cls,pretrained_model_state_dict,dtype=None,device=None):
        model = cls()
        model.load_state_dict(pretrained_model_state_dict)

        if dtype is not None:
            model.to(dtype=dtype)
        
        if device is not None:
            model.to(device=device)

        return model


    def forward(
            self, 
            x: th.Tensor,
            timesteps: th.Tensor, 
            encoder_hidden_states: th.Tensor = None,
            down_block_additional_residuals:list[th.Tensor] = None,
            mid_block_additional_residual:th.Tensor = None
            ) -> th.Tensor:
        h = x.type(self.dtype)

        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        
        # 時間埋め込み
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        emb = self.time_embed(t_emb)

        # Encoder
        skipcs = []
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, encoder_hidden_states)

            if down_block_additional_residuals is not None:
                h = h + down_block_additional_residuals[i]
            
            skipcs.append(h)
        
        # Bottleneck
        h = self.middle_block(h, emb, encoder_hidden_states)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual

        # Decoder
        for module in self.output_blocks:
            # Skip-connection
            skipc = skipcs.pop()
            h = th.cat([h, skipc], dim=1)
            h = module(h, emb, encoder_hidden_states)
        
        h = h.type(x.dtype)
        return self.out(h)
    
    




class ControlNetModel_SD15(nn.Module):
    def __init__(self):
        super().__init__()

        # SDv1.5の固定パラメータをハードコード
        self.model_channels = 320
        self.out_channels = 4
        self.dropout = 0.0
        self.conv_resample = True
        self.transformer_depth = 1
        self.context_dim = 768
        self.num_heads = 8
        self.dtype = th.float16

        #(3,512,512) < conditioning_image
        hint_channels = 3

        self.input_blocks = nn.ModuleList([])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(2, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(2, 256, self.model_channels, 3, padding=1))
        )

        input_blocks_list = [
            #in0
            [Block(nn.Conv2d,4,320,kernel_size=3,padding=1)],
            #in1
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #in2
            [Block(ResBlock,320,320), Block(SpatialTransformer,320,320)],
            #3
            [Block(Downsample,320,320)],
            #4
            [Block(ResBlock,320,640), Block(SpatialTransformer,640,640)],
            #5
            [Block(ResBlock,640,640), Block(SpatialTransformer,640,640)],
            #6
            [Block(Downsample,640,640)],
            #7
            [Block(ResBlock,640,1280), Block(SpatialTransformer,1280,1280)],
            #8
            [Block(ResBlock,1280,1280), Block(SpatialTransformer,1280,1280)],
            #9
            [Block(Downsample,1280,1280)],
            #10
            [Block(ResBlock,1280,1280)],
            #11
            [Block(ResBlock,1280,1280)],
        ]


        # --- 時間埋め込み ---
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.zero_convs = nn.ModuleList([])
        
        #downblock
        for _, in_block in enumerate(input_blocks_list):
            layers=[]

            for b in in_block:
                if b.module is nn.Conv2d:
                    layers.append(nn.Conv2d(b.inch,b.outch,**b.kwargs))

                if b.module is ResBlock:
                    layers.append(ResBlock(b.inch, time_embed_dim, self.dropout, out_channels=b.outch))

                elif b.module is SpatialTransformer:
                    dim_head = b.inch // self.num_heads
                    layers.append(SpatialTransformer(b.inch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
    
                elif b.module is Downsample:
                    layers.append(Downsample(b.inch, self.conv_resample))

            self.input_blocks.append(TimestepEmbedSequential(*layers))
            self.zero_convs.append(TimestepEmbedSequential(zero_module(conv_nd(2, b.outch, b.outch, 1, padding=0))))


        #midblock
        ch = input_blocks_list[-1][-1].outch #1280(last output channel)
        dim_head = ch // self.num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
            SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim),
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
        )
        self.middle_block_out = TimestepEmbedSequential(zero_module(conv_nd(2, ch, ch, 1, padding=0)))



    @classmethod
    def from_pretrained(cls,pretrained_model_state_dict,dtype=None,device=None):
        model = cls()
        model.load_state_dict(pretrained_model_state_dict)

        if dtype is not None:
            model.to(dtype=dtype)
        
        if device is not None:
            model.to(device=device)

        return model
        

    def forward(
            self, 
            x: th.Tensor, 
            timesteps: th.Tensor, 
            encoder_hidden_states: th.Tensor,
            controlnet_cond: th.Tensor,
            control_scale: float = 1.0,
            guess_mode: bool =False
            ) -> th.Tensor:
        
        if timesteps.dim() == 0:
            timesteps = timesteps[None]

        # 時間埋め込み
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        emb = self.time_embed(t_emb)

        # controlnet_cond > guided_hint
        guided_hint = self.input_hint_block(controlnet_cond, emb, encoder_hidden_states)

        down_block_res_samples = []

        # input_block
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                #最初だけguid_hintを加算する
                h = module(h, emb, encoder_hidden_states)
                h += guided_hint
                guided_hint=None
            else:
                h = module(h, emb, encoder_hidden_states)
            down_block_res_samples.append(zero_conv(h))
        
        # middle_block
        h = self.middle_block(h,emb,encoder_hidden_states)
        mid_block_res_sample = self.middle_block_out(h)

        if guess_mode:
            scales = th.logspace(-1, 0, len(down_block_res_samples) + 1, device=x.device)  # 0.1 to 1.0
            scales = scales * control_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * control_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * control_scale

        return down_block_res_samples, mid_block_res_sample




class UNetModel_SDXL(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_channels = 320  # これは同じ
        self.out_channels = 4      # これも同じ
        self.dropout = 0.0
        self.conv_resample = True
        # ### 変更点 1 ###
        self.context_dim = 2048    # 768 -> 2048 (2つのテキストエンコーダーの次元数を反映)
        self.num_head_channels = 64 # ヘッドの次元数を64に固定
        # self.num_headsは各Transformerで動的に計算するため、固定値は削除
        self.dtype = th.float16


        input_blocks_list = [
            #in0
            [Block(nn.Conv2d,4,320,kernel_size=3,padding=1)],

            #in1
            [Block(ResBlock,320,320)],

            #in2
            [Block(ResBlock,320,320)],

            #3
            [Block(nn.Conv2d,320,320,kernel_size=3,stride=2,padding=1)],

            #4
            [Block(ResBlock,320,640), 
             Block(SpatialTransformer,640,640,depth=2)],

            #5
            [Block(ResBlock,640,640), 
             Block(SpatialTransformer,640,640,depth=2)],

            #6
            [Block(nn.Conv2d,640,640,kernel_size=3,stride=2,padding=1)],

            #7
            [Block(ResBlock,640,1280), 
             Block(SpatialTransformer,1280,1280,depth=10)],

            #8
            [Block(ResBlock,1280,1280), 
             Block(SpatialTransformer,1280,1280,depth=10)]
        ]

        output_blocks_list = [
            #in0
            [Block(ResBlock,1280,1280), 
             Block(SpatialTransformer,1280,1280,depth=10)],

            #in1
            [Block(ResBlock,1280,1280), 
             Block(SpatialTransformer,1280,1280,depth=10)],

            #in2
            [Block(ResBlock,1280,1280), 
             Block(SpatialTransformer,1280,1280,depth=10),
             Block(Upsample,1280,1280)],

            #in3
            [Block(ResBlock,1280,640), 
             Block(SpatialTransformer,640,640,depth=10)],

            #in4
            [Block(ResBlock,640,640), 
             Block(SpatialTransformer,640,640,depth=2)],

            #in5
            [Block(ResBlock,640,640), 
             Block(SpatialTransformer,640,640,depth=2), 
             Block(Upsample,640,640)],

            #in6
            [Block(ResBlock,640,320)],

            #in7
            [Block(ResBlock,320,320)],

            #in8
            [Block(ResBlock,320,320)]
        ]
        

        # --- 時間埋め込み ---
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.add_embedder = nn.Sequential(
            nn.Linear(2816, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        #downblock
        self.input_blocks = nn.ModuleList([])

        # TimestepEmbedSequentialでまとめる
        for _, in_block in enumerate(input_blocks_list):
            layers=[]

            for b in in_block:
                if b.module is nn.Conv2d:
                    layers.append(nn.Conv2d(b.inch,b.outch,**b.kwargs))

                if b.module is ResBlock:
                    layers.append(ResBlock(b.inch, time_embed_dim, self.dropout, out_channels=b.outch))

                elif b.module is SpatialTransformer:
                    dim_head = b.inch // self.num_heads
                    layers.append(SpatialTransformer(b.inch, self.num_heads, dim_head, context_dim=self.context_dim, **b.kwargs))

                elif b.module is Downsample:
                    layers.append(Downsample(b.inch, self.conv_resample))

            self.input_blocks.append(TimestepEmbedSequential(*layers))


        #midblock
        ch = input_blocks_list[-1][-1].outch
        dim_head = ch // self.num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
            SpatialTransformer(ch, self.num_heads, dim_head, depth=10, context_dim=self.context_dim),
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
        )

        #upblock
        self.output_blocks = nn.ModuleList([])

        for i, out_block in enumerate(output_blocks_list):
            layers = [] 
            skipc = input_blocks_list[-1-i][0].outch #skip connection

            for b in out_block:
                if b.module is ResBlock:
                    layers.append(ResBlock(b.inch+skipc, time_embed_dim, self.dropout, out_channels=b.outch))

                elif b.module is SpatialTransformer:
                    dim_head = b.inch // self.num_heads
                    layers.append(SpatialTransformer(b.inch, self.num_heads, dim_head, context_dim=self.context_dim, **b.kwargs))

                elif b.module is Upsample:
                    layers.append(Upsample(b.inch, self.conv_resample))
       
            self.output_blocks.append(TimestepEmbedSequential(*layers))

        
        # --- 出力 ---
        ch = output_blocks_list[-1][-1].outch
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(self.model_channels,self.out_channels,kernel_size=3,padding=1)),
        )

    @classmethod
    def from_pretrained(cls,pretrained_model_state_dict,dtype=None,device=None):
        model = cls()
        model.load_state_dict(pretrained_model_state_dict)

        if dtype is not None:
            model.to(dtype=dtype)
        
        if device is not None:
            model.to(device=device)

        return model


    def forward(
            self, 
            x: th.Tensor,
            timesteps: th.Tensor, 
            encoder_hidden_states: th.Tensor,
            down_block_additional_residuals:list[th.Tensor] = None,
            mid_block_additional_residual:th.Tensor = None,
            **kwargs
            ) -> th.Tensor:
        h = x.type(self.dtype)

        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        
        # 時間埋め込み
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        t_emb = self.time_embed(t_emb)

        add_embeds = th.cat(
                [kwargs["text_embeds"], kwargs["time_ids"]], dim=-1
            )
        add_embeds = add_embeds.to(self.dtype)
        aug_emb = self.add_embedder(add_embeds)
        emb = t_emb + aug_emb # 時間埋め込みと追加条件を足し合わせる

        # Encoder
        skipcs = []
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, encoder_hidden_states)

            if down_block_additional_residuals is not None:
                h = h + down_block_additional_residuals[i]

            skipcs.append(h)
        
        # Bottleneck
        h = self.middle_block(h, emb, encoder_hidden_states)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual

        # Decoder
        for module in self.output_blocks:
            # Skip-connection
            skipc = skipcs.pop()
            h = th.cat([h, skipc], dim=1)
            h = module(h, emb, encoder_hidden_states)
        
        h = h.type(x.dtype)
        return self.out(h)




"""


class UNetModel_SD15_U(nn.Module):
    def __init__(self):
        super().__init__()

        # SDv1.5の固定パラメータをハードコード
        self.model_channels = 320
        self.out_channels = 4
        self.dropout = 0.0
        self.conv_resample = True
        self.transformer_depth = 1
        self.context_dim = 768
        self.num_heads = 8
        self.dtype = th.float16

        input_blocks_list = [
            #in0
            [("ConvIn",(4,320))],
            #in1
            [("ResBlock", (320, 320)),("Transformer", (320, 320))],
            #in2
            [("ResBlock", (320, 320)),("Transformer", (320, 320))],
            #3
            [("Downsample", (320, 320))],
            #4
            [("ResBlock", (320, 640)),("Transformer", (640, 640))],
            #5
            [("ResBlock", (640, 640)),("Transformer", (640, 640))],
            #6
            [("Downsample", (640, 640))],
            #7
            [("ResBlock", (640, 1280)),("Transformer", (1280, 1280))],
            #8
            [("ResBlock", (1280, 1280)),("Transformer", (1280, 1280))],
            #9
            [("Downsample", (1280, 1280))],
            #10
            [("ResBlock", (1280, 1280))],
            #11
            [("ResBlock", (1280, 1280))],
        ]

        output_blocks_list = [
            #in0
            [("ResBlock", (1280, 1280))],
            #in1
            [("ResBlock", (1280, 1280))],
            #in2
            [("ResBlock", (1280, 1280)),("Upsample", (1280, 1280))],
            #in3
            [("ResBlock", (1280, 1280)),("Transformer", (1280, 1280))],
            #in4
            [("ResBlock", (1280, 1280)),("Transformer", (1280, 1280))],
            #in5
            [("ResBlock", (1280, 1280)),("Transformer", (1280, 1280)),("Upsample", (1280, 1280))],
            #in6
            [("ResBlock", (1280, 640)),("Transformer", (640, 640))],
            #in7
            [("ResBlock", (640, 640)),("Transformer", (640, 640))],
            #in8
            [("ResBlock", (640, 640)),("Transformer", (640, 640)),("Upsample", (640, 640))],
            #in9
            [("ResBlock", (640, 320)),("Transformer", (320, 320))],
            #in10
            [("ResBlock", (320, 320)),("Transformer", (320, 320))],
            #in11
            [("ResBlock", (320, 320)),("Transformer", (320, 320))],
        ]
        

        # --- 時間埋め込み ---
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        #downblock
        self.input_blocks = nn.ModuleList([])
        for _,block in enumerate(input_blocks_list):
            layers=[]
            for module, (inch,outch) in block:
                if module == "ConvIn":
                    layers.append(nn.Conv2d(inch,outch,kernel_size=3,padding=1))
                if module == "ResBlock":
                    layers.append(ResBlock(inch, time_embed_dim, self.dropout, out_channels=outch))
                elif module == "Transformer":
                    dim_head = inch // self.num_heads
                    layers.append(SpatialTransformer(inch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
                elif module == "Downsample":
                    layers.append(Downsample(inch, self.conv_resample))

            self.input_blocks.append(TimestepEmbedSequential(*layers))

        #midblock
        ch = input_blocks_list[-1][-1][-1][-1] #1280
        dim_head = ch // self.num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
            SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim),
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
        )

        #upblock
        self.output_blocks = nn.ModuleList([])
        layers = []
        for i,block in enumerate(output_blocks_list):
            layers = [] 
            skipc = input_blocks_list[len(input_blocks_list)-1-i][0][1][1] #skip connection
            for module, (inch, outch) in block:
                if module == "ResBlock":
                    layers.append(ResBlock(inch+skipc, time_embed_dim, self.dropout, out_channels=outch))
                elif module == "Transformer":
                    dim_head = inch // self.num_heads
                    layers.append(SpatialTransformer(inch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
                elif module == "Upsample":
                    layers.append(Upsample(inch, self.conv_resample))
       
            self.output_blocks.append(TimestepEmbedSequential(*layers))
 
        
        # --- 出力 ---
        ch = 320
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(self.model_channels,self.out_channels,kernel_size=3,padding=1)),
        )


    def forward(self, x: th.Tensor, timesteps: th.Tensor, encoder_hidden_states: th.Tensor) -> th.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        hs = []
        
        # 時間埋め込み
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        emb = self.time_embed(t_emb)

        # Encoder
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, encoder_hidden_states)
            hs.append(h)
        
        # Bottleneck
        h = self.middle_block(h, emb, encoder_hidden_states)
        
        # Decoder
        for module in self.output_blocks:
            # Skip-connection
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, encoder_hidden_states)
        
        h = h.type(x.dtype)
        return self.out(h)

        
class UNetModel_SD15(nn.Module):
    def __init__(self):
        super().__init__()

        # SDv1.5の固定パラメータをハードコード
        self.in_channels = 4
        self.model_channels = 320
        self.out_channels = 4
        self.num_res_blocks = 2
        self.attention_resolutions = (4, 2, 1) # ds=4,2,1 (64->16, 32->16, 16->8)のレベルでAttention
        self.dropout = 0.0
        self.channel_mult = (1, 2, 4, 4)
        self.conv_resample = True
        self.dims = 2
        self.use_spatial_transformer = True
        self.transformer_depth = 1
        self.context_dim = 768
        self.num_heads = 8
        self.dtype = th.float16

        # --- 時間埋め込み ---
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # --- ダウンサンプリング (Encoder) ---
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(self.dims, self.in_channels, self.model_channels, 3, padding=1))
        ])

        #------------------------------------

        #------------------------------------
        
        input_block_chans = [self.model_channels]
        ch = self.model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=mult * self.model_channels)]
                ch = mult * self.model_channels
                
                # この解像度でAttention/Transformerを適用するかどうか
                if ds in self.attention_resolutions:
                    dim_head = ch // self.num_heads
                    layers.append(
                        SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim)
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            # 最後のレベル以外ではDownsample
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, self.conv_resample, dims=self.dims))
                )
                ds *= 2
                input_block_chans.append(ch)
        

        # --- 中間ブロック (Bottleneck) ---
        dim_head = ch // self.num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout),
            SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim),
            ResBlock(ch, time_embed_dim, self.dropout),
        )

        # --- アップサンプリング (Decoder) ---
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=self.model_channels * mult)
                ]
                ch = self.model_channels * mult

                if ds in self.attention_resolutions:
                    dim_head = ch // self.num_heads
                    layers.append(
                        SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim)
                    )

                # 最初のレベル以外、かつ各レベルの最後のブロックの後にUpsample
                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample, dims=self.dims))
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # --- 出力 ---
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, self.model_channels, self.out_channels, 3, padding=1)),
        )

    def forward(self, x: th.Tensor, timesteps: th.Tensor, encoder_hidden_states: th.Tensor) -> th.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        hs = []
        # 時間埋め込み
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        emb = self.time_embed(t_emb)

        # Encoder
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, encoder_hidden_states)
            hs.append(h)
        
        # Bottleneck
        h = self.middle_block(h, emb, encoder_hidden_states)
        
        # Decoder
        for module in self.output_blocks:
            # Skip-connection
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, encoder_hidden_states)
            
        h = h.type(x.dtype)
        return self.out(h)


class UNetModel_SD15_T(nn.Module):
    def __init__(self):
        super().__init__()

        # SDv1.5の固定パラメータをハードコード
        self.in_channels = 4
        self.model_channels = 320
        self.out_channels = 4
        self.num_res_blocks = 2
        self.attention_resolutions = (4, 2, 1) # ds=4,2,1 (64->16, 32->16, 16->8)のレベルでAttention
        self.dropout = 0.0
        self.channel_mult = (1, 2, 4, 4)
        self.conv_resample = True
        self.dims = 2
        self.use_spatial_transformer = True
        self.transformer_depth = 1
        self.context_dim = 768
        self.num_heads = 8
        self.dtype = th.float16

        # --- 時間埋め込み ---
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # conv_ndはself.dimsの値によってnn.Conv2dかnn.Conv3dを返す関数と仮定

        # ==============================================================================
        # 1. ダウンサンプリング (Encoder)
        # ==============================================================================
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(self.dims, self.in_channels, self.model_channels, 3, padding=1))
        ])

        input_block_chans = [self.model_channels]
        ch = self.model_channels # ch = 320
        #ds = 1                   # ds = 1
        

        # --- Encoder: Level 0 (mult=1, ch=320) ---
        # in 1
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=320)]
        ch = 320
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        # in 2
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=320)]
        ch = 320
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        # in 3
        self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, self.conv_resample, dims=self.dims)))
        #ds *= 2 # ds = 2
        input_block_chans.append(ch)


        # --- Encoder: Level 1 (mult=2, ch=640) ---
        # in 4
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=640)] # 320 -> 640
        ch = 640
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        # in 5
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=640)]
        ch = 640
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        # in 6
        self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, self.conv_resample, dims=self.dims)))
        #ds *= 2 # ds = 4
        input_block_chans.append(ch)


        # --- Encoder: Level 2 (mult=4, ch=1280) ---
        # in 7
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=1280)] # 640 -> 1280
        ch = 1280
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        # in 8
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=1280)]
        ch = 1280
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        # in 9
        self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, self.conv_resample, dims=self.dims)))
        #ds *= 2 # ds = 8
        input_block_chans.append(ch)

        # --- Encoder: Level 3 (mult=4, ch=1280) ---
        # in 10
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=1280)] # 1280 -> 1280
        ch = 1280
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        # in 11
        layers = [ResBlock(ch, time_embed_dim, self.dropout, out_channels=1280)]
        ch = 1280
        self.input_blocks.append(TimestepEmbedSequential(*layers))
        input_block_chans.append(ch)

        print(self.input_blocks)
        exit()

        # ==============================================================================
        # 2. 中間ブロック (Bottleneck)
        # ==============================================================================
        # Encoderを抜けた時点で ch=1280, ds=8
        dim_head = ch // self.num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
            SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim),
            ResBlock(ch, time_embed_dim, self.dropout, out_channels=ch),
        )


        ch = 1280 # middle_blockを抜けた時点でのチャンネル数
        #ds = 8    # Encoderを抜けた時点でのds

        # ==============================================================================
        # 3. アップサンプリング (Decoder) 
        # ==============================================================================
        self.output_blocks = nn.ModuleList([])

        # out 0
        ich = input_block_chans.pop() # 1280
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=1280)] # (1280+1280)->1280
        ch = 1280
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 1
        ich = input_block_chans.pop() # 1280
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=1280)] # (1280+1280)->1280
        ch = 1280
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out2
        ich = input_block_chans.pop() # 1280
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=1280)] # (1280+1280)->1280
        ch = 1280
        layers.append(Upsample(ch, self.conv_resample, dims=self.dims)) # Upsample追加
        #ds //= 2 # ds = 4
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 3
        ich = input_block_chans.pop() # 1280
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=1280)] # (1280+1280)->1280
        ch = 1280
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 4
        ich = input_block_chans.pop() # 1280
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=1280)] # (1280+1280)->1280
        ch = 1280
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 5
        ich = input_block_chans.pop() # 640
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=1280)] # (1280+640)->1280
        ch = 1280
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        layers.append(Upsample(ch, self.conv_resample, dims=self.dims))
        #ds //= 2 # ds = 2
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 6
        ich = input_block_chans.pop() # 640
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=640)] # (1280+640)->640
        ch = 640
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 7
        ich = input_block_chans.pop() # 640
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=640)] # (640+640)->640
        ch = 640
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 8
        ich = input_block_chans.pop() # 320
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=640)] # (640+320)->640
        ch = 640
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        layers.append(Upsample(ch, self.conv_resample, dims=self.dims))
        #ds //= 2 # ds = 1
        self.output_blocks.append(TimestepEmbedSequential(*layers))


        # out 9
        ich = input_block_chans.pop() # 320
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=320)] # (640+320)->320
        ch = 320
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 10
        ich = input_block_chans.pop() # 320
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=320)] # (320+320)->320
        ch = 320
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.output_blocks.append(TimestepEmbedSequential(*layers))

        # out 11
        ich = input_block_chans.pop() # 320
        layers = [ResBlock(ch + ich, time_embed_dim, self.dropout, out_channels=320)] # (320+320)->320
        ch = 320
        dim_head = ch // self.num_heads
        layers.append(SpatialTransformer(ch, self.num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim))
        self.output_blocks.append(TimestepEmbedSequential(*layers))


        # --- 出力 ---
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, self.model_channels, self.out_channels, 3, padding=1)),
        )


        

    def forward(self, x: th.Tensor, timesteps: th.Tensor, encoder_hidden_states: th.Tensor) -> th.Tensor:

        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        hs = []
        # 時間埋め込み
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False,dtype=self.dtype)
        emb = self.time_embed(t_emb)

        # Encoder
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, encoder_hidden_states)
            hs.append(h)
        
        # Bottleneck
        h = self.middle_block(h, emb, encoder_hidden_states)
        
        # Decoder
        for module in self.output_blocks:
            # Skip-connection
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, encoder_hidden_states)
            
        h = h.type(x.dtype)
        return self.out(h)
"""