#python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=152.2.132.95 --master_port=1232 unigradicon_train_parallel_attention.py --encoder replace

# Same as unigradicon_train_parallel.py, but the 3 residual 3x3x3 conv blocks of the
# coordinate-attention encoder (fpc.SomeDownsampleNoDilationNet) are replaced by transformer
# blocks: self-attention with 3D RoPE, then cross-attention between A and B with no positional
# encoding (so each image's features stay translation-equivariant on their own token grid),
# then an MLP. The conv stem is kept, so its weights load from conv checkpoints.
# Ported from claudevm/unicarl-attention/train_attention.py.
#
# --encoder replace  12 self+cross blocks, width 384: 28.7M encoder params (conv encoder: 36.1M)
# --encoder self     16 self-attention-only blocks: 28.7M, the matched no-cross-attention control
# --encoder hybrid   conv blocks, each followed by a zero-initialised attention block;
#                    identical to the conv model at step 0, for warm starts (--init)
# --center           subtract per-image channel means before the L2 norm (see JointAttentionFeaturizer)

from datetime import datetime
from icon_registration.config import device
import icon_registration.unicarl.fixed_point_carl as fpc
import argparse
import footsteps
import icon_registration as icon
import icon_registration.carl as carl
import icon_registration.data
import icon_registration.networks as networks
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import icon_registration.unicarl
import icon_registration.unicarl.unigradicon_train_parallel as utp
os.environ["OMP_NUM_THREADS"]="8"


class RoPE3D(nn.Module):
    """Rotary embedding over (z, y, x) token indices. head_dim must be divisible by 6."""

    def __init__(self, head_dim, base=100.0):
        super().__init__()
        assert head_dim % 6 == 0
        per_axis = head_dim // 3
        freqs = base ** (-torch.arange(0, per_axis, 2).float() / per_axis)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x, pos):
        # x: [B, heads, N, head_dim], pos: [N, 3] integer token coordinates
        angles = torch.cat([pos[:, i, None].float() * self.freqs for i in range(3)], dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim, heads, rope=None):
        super().__init__()
        self.heads, self.rope = heads, rope
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, ctx, pos_x=None, pos_ctx=None):
        B, N, C = x.shape
        h = self.heads
        q = self.q(x).view(B, N, h, C // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, h, C // h).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q, pos_x), self.rope(k, pos_ctx)
        # No silent fallback to the math kernel: it materialises every attention matrix
        # (~1.1 GB per image per block for padded A in fp32), which would OOM or crawl.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(B, N, C))


class JointBlock(nn.Module):
    """Self-attention (RoPE) -> cross-attention (no PE, symmetric, shared weights) -> MLP."""

    def __init__(self, dim=fpc.MODEL_DIM, heads=8, cross=True, mlp_ratio=4, zero_init=False):
        super().__init__()
        head_dim = dim // heads
        self.norm1, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if cross else None
        self.self_attn = Attention(dim, heads, rope=RoPE3D(head_dim))
        self.cross_attn = Attention(dim, heads) if cross else None
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        if zero_init:
            for lin in [self.self_attn.out, self.mlp[-1]] + ([self.cross_attn.out] if cross else []):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def forward(self, a, b, pos_a, pos_b):
        a = a + self.self_attn(self.norm1(a), self.norm1(a), pos_a, pos_a)
        b = b + self.self_attn(self.norm1(b), self.norm1(b), pos_b, pos_b)
        if self.cross_attn is not None:
            na, nb = self.norm2(a), self.norm2(b)
            # simultaneous update: swapping (A, B) swaps the outputs, as ICON's symmetry expects
            a, b = a + self.cross_attn(na, nb), b + self.cross_attn(nb, na)
        a = a + self.mlp(self.norm3(a))
        b = b + self.mlp(self.norm3(b))
        return a, b


def to_tokens(x):
    # [B, C, D, H, W] -> [B, N, C], plus integer (z, y, x) positions [N, 3]
    B, C, *spatial = x.shape
    pos = torch.stack(torch.meshgrid(*[torch.arange(s, device=x.device) for s in spatial], indexing="ij"), -1)
    return x.flatten(2).transpose(1, 2), pos.reshape(-1, 3), spatial


def from_tokens(t, spatial):
    return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], *spatial)


class JointAttentionEncoder(fpc.SomeDownsampleNoDilationNet):
    """Encodes A and B together. Subclassing keeps the conv parameter names
    (convs0, convs.*, batchnorms.*), so conv checkpoints load with strict=False."""

    def __init__(self, dimension=3, mode="replace", depth=12, cross=True, heads=8, mlp_ratio=4):
        super().__init__(dimension=dimension)
        assert dimension == 3
        assert mode in ("replace", "hybrid")
        self.mode = mode
        if mode == "replace":
            # keep only the stem (convs0, convs.0); the residual trunk is replaced
            self.convs = nn.ModuleList([self.convs[0]])
            self.batchnorms = nn.ModuleList([])
        else:
            assert depth == 3, "hybrid interleaves one attention block per residual conv block"
        self.blocks = nn.ModuleList([
            JointBlock(fpc.MODEL_DIM, heads, cross=cross, mlp_ratio=mlp_ratio, zero_init=(mode == "hybrid"))
            for _ in range(depth)
        ])

    def stem(self, x):
        x = self.avg_pool(self.convs0(x), 2, ceil_mode=True)
        return torch.relu(self.avg_pool(self.convs[0](x), 2, ceil_mode=True))

    def conv_block(self, x, i):
        x = self.batchnorms[i](x)
        y = self.convs[i + 7](torch.relu(self.convs[i + 4](torch.relu(self.convs[i + 1](x)))))
        return x + y

    def forward(self, A, B):
        a, b = self.stem(A), self.stem(B)
        for i, block in enumerate(self.blocks):
            if self.mode == "hybrid":
                a, b = self.conv_block(a, i), self.conv_block(b, i)
            ta, pos_a, sa = to_tokens(a)
            tb, pos_b, sb = to_tokens(b)
            ta, tb = block(ta, tb, pos_a, pos_b)
            a, b = from_tokens(ta, sa), from_tokens(tb, sb)
        return a, b


class JointAttentionFeaturizer(fpc.AttentionFeaturizer):
    """AttentionFeaturizer whose encoder sees both images at once. Normalisation,
    padding/cropping and the coordinate attention are unchanged.

    center=True subtracts each channel's mean over the tokens of each image before the
    L2 normalisation: at init the attention trunks give every token almost the same feature,
    so the coordinate softmax is flat and every point maps to the mean coordinate; centering
    removes that shared component. The mean is translation-invariant, so equivariance is unaffected."""

    def __init__(self, net, dimension=2, center=False):
        super().__init__(net, dimension=dimension)
        self.center = center

    def forward(self, A, B):
        p = self.padding
        pad = lambda x: F.pad(x, [p] * (2 * self.dimension))
        ft_A, ft_B = self.net(pad(A), pad(B))
        if self.center:
            spatial = tuple(range(2, 2 + self.dimension))
            ft_A = ft_A - ft_A.mean(dim=spatial, keepdim=True)
            ft_B = ft_B - ft_B.mean(dim=spatial, keepdim=True)
        norm = lambda x: 4 * x / (0.001 + torch.sqrt(torch.sum(x**2, dim=1, keepdims=True)))
        return self.registerer(norm(ft_A), self.crop(norm(ft_B)))


def make_net(dimension, input_shape, encoder="replace", center=False, rm=utp.RandomMatrix()):
    """utp.make_net(dimension, input_shape, False) with the attention encoder."""
    kwargs = {"replace": dict(mode="replace", depth=12, cross=True),
              "self": dict(mode="replace", depth=16, cross=False),
              "hybrid": dict(mode="hybrid", depth=3, cross=True)}[encoder]
    ts = JointAttentionFeaturizer(JointAttentionEncoder(dimension=dimension, **kwargs), dimension=dimension, center=center)

    ts = icon.network_wrappers.DownsampleNet(ts, dimension)
    for _ in range(3):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 21),
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension)), dimension)
         )
    for _ in range(2):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 11),
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension)), dimension)
         )
    ts = icon.network_wrappers.DownsampleNet(ts, dimension)

    for _ in range(1):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 8),
             carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension))
         )
    net = icon.losses.GradientICONSparse(ts, utp.SquaredLNCC(sigma=4), lmbda=1.5)
    net.assign_identity_map(input_shape)
    net = utp.augmentify(net, rm=rm)
    net.assign_identity_map(input_shape)
    net.train()
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["replace", "self", "hybrid"], default="replace")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--init", default=None,
                        help="regis_net checkpoint to start from (strict=False: conv weights fill the stem, "
                             "or the whole conv encoder in hybrid mode; attention blocks stay fresh)")
    parser.add_argument("--steps", type=int, default=450000)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest  # train_batchfunction parses --local-rank itself

    import datasets
    from datasets import input_shape

    net = make_net(3, input_shape, encoder=args.encoder, center=args.center)
    if args.init:
        missing, unexpected = net.regis_net.load_state_dict(torch.load(args.init, map_location="cpu"), strict=False)
        print(f"--init {args.init}: {len(missing)} new tensors, {len(unexpected)} checkpoint tensors unused")

    BATCH_SIZE = 4

    utp.train_batchfunction(
        net,
        utp.make_make_pair(datasets),
        steps=args.steps
    )
