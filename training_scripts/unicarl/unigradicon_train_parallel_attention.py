# FOOTSTEPS_NAME=<run name> torchrun --standalone --nproc_per_node=4 unigradicon_train_parallel_attention.py
# (see RUNGUIDE_attention.md; batch 20 per GPU needs ~72 GiB per GPU)

# Same as unigradicon_train_parallel.py, but the 3 residual 3x3x3 conv blocks of the
# coordinate-attention encoder (fpc.SomeDownsampleNoDilationNet) are replaced by transformer
# blocks (12 of them, width 384: 28.7M encoder params vs the conv encoder's 36.1M):
# self-attention with 3D RoPE, then cross-attention between A and B with no positional
# encoding (so each image's features stay translation-equivariant on their own token grid),
# then an MLP. The conv stem is kept, so its weights load from conv checkpoints.
# Ported from claudevm/unicarl-attention/train_attention.py (--encoder replace --center).

from datetime import datetime
from icon_registration.config import device
import icon_registration.unicarl.fixed_point_carl as fpc
import footsteps
import icon_registration as icon
import icon_registration.carl as carl
import icon_registration.data
import icon_registration.networks as networks
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch._functorch.config
from torch.nn.attention import SDPBackend, sdpa_kernel
import icon_registration.unicarl
import icon_registration.unicarl.unigradicon_train_parallel as utp
import fast_training
os.environ["OMP_NUM_THREADS"]="8"
# every step has the same shapes, so let cuDNN pick the fastest conv algorithms once
torch.backends.cudnn.benchmark = True


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

    def __init__(self, dim=fpc.MODEL_DIM, heads=8, mlp_ratio=4):
        super().__init__()
        head_dim = dim // heads
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads, rope=RoPE3D(head_dim))
        self.cross_attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, a, b, pos_a, pos_b):
        a = a + self.self_attn(self.norm1(a), self.norm1(a), pos_a, pos_a)
        b = b + self.self_attn(self.norm1(b), self.norm1(b), pos_b, pos_b)
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
    (convs0, convs.0), so the stem loads from conv checkpoints with strict=False."""

    def __init__(self, dimension=3, depth=12, heads=8, mlp_ratio=4):
        super().__init__(dimension=dimension)
        assert dimension == 3
        # keep only the stem (convs0, convs.0); the residual trunk is replaced
        self.convs = nn.ModuleList([self.convs[0]])
        self.batchnorms = nn.ModuleList([])
        self.blocks = nn.ModuleList([JointBlock(fpc.MODEL_DIM, heads, mlp_ratio=mlp_ratio) for _ in range(depth)])

    def stem(self, x):
        x = self.avg_pool(self.convs0(x), 2, ceil_mode=True)
        return torch.relu(self.avg_pool(self.convs[0](x), 2, ceil_mode=True))

    def forward(self, A, B):
        ta, pos_a, sa = to_tokens(self.stem(A))
        tb, pos_b, sb = to_tokens(self.stem(B))
        # Mixed precision for the transformer blocks only (the coordinate attention stays fp32):
        # ~4.6x faster fwd+bwd than fp32. bf16 (Ampere+) also enables flash attention and needs no
        # loss scaling; fp16 is only there so the efficient kernel runs on older GPUs for testing.
        # LayerNorm runs in fp32 under autocast, so the residual stream stays fp32.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(A.device)[0] >= 8 else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            for block in self.blocks:
                # Activation checkpointing: the blocks' activations were ~3.9 GiB per sample, the
                # largest share of training memory (8.4 -> 5.2 GiB per sample, +12% step time on a
                # Quadro RTX 6000; saving the attention outputs and recomputing the rest was no
                # faster). No BatchNorm or randomness here, so the result is the same.
                if torch.is_grad_enabled():
                    ta, tb = fast_training.checkpoint(block, ta, tb, pos_a, pos_b)
                else:
                    ta, tb = block(ta, tb, pos_a, pos_b)
        return from_tokens(ta.float(), sa), from_tokens(tb.float(), sb)


class JointAttentionFeaturizer(fpc.AttentionFeaturizer):
    """AttentionFeaturizer whose encoder sees both images at once. Normalisation and the
    coordinate attention are unchanged.

    Only A is padded: the coordinate attention needs A's features on the padded grid (so B's
    points can map outside A's field of view), but crops B's back to the image anyway. Encoding
    B unpadded (10^3 tokens instead of 18^3) makes the encoder ~2.5x faster; B's tokens just
    no longer see the zero border as context.

    Each image's features have their per-channel mean (over its tokens) subtracted before
    the L2 normalisation. Without this, at init the attention trunks give every token almost
    the same feature (cosine similarity ~1.0 on a real MR/CT pair), so the coordinate softmax
    is flat and every point maps to the mean coordinate: the first stage collapses the image
    and the later stages train against a constant warped image. Centered, the start is random
    but well-spread (cosine ~0.23). The mean is translation-invariant, so equivariance is
    unaffected, and a component shared by every token carries no correspondence anyway."""

    def forward(self, A, B):
        p = self.padding
        pad = lambda x: F.pad(x, [p] * (2 * self.dimension))
        ft_A, ft_B = self.net(pad(A), B)
        spatial = tuple(range(2, 2 + self.dimension))
        ft_A = ft_A - ft_A.mean(dim=spatial, keepdim=True)
        ft_B = ft_B - ft_B.mean(dim=spatial, keepdim=True)
        norm = lambda x: 4 * x / (0.001 + torch.sqrt(torch.sum(x**2, dim=1, keepdims=True)))
        return self.registerer(norm(ft_A), norm(ft_B))


def make_net(dimension, input_shape, rm=utp.RandomMatrix()):
    """utp.make_net(dimension, input_shape, False) with the attention encoder."""
    ts = JointAttentionFeaturizer(JointAttentionEncoder(dimension=dimension), dimension=dimension)

    ts = icon.network_wrappers.DownsampleNet(ts, dimension)
    for _ in range(3):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 21),
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(fast_training.checkpoint_unet(networks.tallUNet2(dimension=dimension))), dimension)
         )
    for _ in range(2):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 11),
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(fast_training.checkpoint_unet(networks.tallUNet2(dimension=dimension))), dimension)
         )
    ts = icon.network_wrappers.DownsampleNet(ts, dimension)

    for _ in range(1):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 8),
             carl.RotationFunctionFromVectorField(fast_training.checkpoint_unet(networks.tallUNet2(dimension=dimension)))
         )
    net = icon.losses.GradientICONSparse(ts, utp.SquaredLNCC(sigma=4), lmbda=1.5)
    net.assign_identity_map(input_shape)
    net = utp.augmentify(net, rm=rm)
    net.assign_identity_map(input_shape)
    net.train()
    return net


if __name__ == "__main__":

    import datasets
    from datasets import input_shape

    BATCH_SIZE = 20  # per GPU
    utp.BATCH_SIZE = BATCH_SIZE  # read by utp.make_make_pair

    net = make_net(3, input_shape)

    # Inductor recomputes cheap ops in the compiled regions instead of saving them: with the
    # checkpointing above, 3.5 GiB per sample (as eager) instead of 4.5, at ~8% faster than eager.
    torch._functorch.config.activation_memory_budget = 0.5
    fast_training.compile_training_forward(net)

    utp.train_batchfunction(
        net,
        fast_training.prefetching(utp.make_make_pair(datasets)),
        steps=450000
    )
