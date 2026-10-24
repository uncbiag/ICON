"""Speed and memory machinery shared by the uniCARL training scripts in this directory.
Nothing here changes what is computed (see RUNGUIDE_attention.md for the checks)."""

import inspect
import queue
import threading

import icon_registration.networks as networks
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


# Activation checkpointing of the encoder blocks and UNets: ~3.5 instead of ~8.4 GiB per
# sample for the attention model, but ~1/3 slower.
CHECKPOINT = True


def checkpoint(fn, *args):
    return _checkpoint_eager(fn, *args) if CHECKPOINT else fn(*args)


@torch.compiler.disable
def _checkpoint_eager(fn, *args):
    # Runs eagerly even inside torch.compile: inductor schedules the recompute of checkpointed
    # regions inside one big backward graph, which in torch 2.7.1 peaked higher than not
    # checkpointing at all (batch 4 OOMed on 24 GB).
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)


def _unet2_forward_pure(unet, x, y):
    """networks.UNet2.forward with each BatchNorm applied with batch statistics (what training
    mode does) and without touching the running statistics; returns those batch statistics
    instead, so the caller can update the running statistics once. No side effects, so it can
    be checkpointed (also under torch.compile)."""
    x = torch.cat([x, y], 1)
    skips = []
    stats = []
    for depth in range(unet.num_layers):
        skips.append(x)
        y = unet.downConvs[depth](F.leaky_relu(x))
        x = y + networks.pad_or_crop(unet.avg_pool(x, 2, ceil_mode=True), y.size(), unet.dimension)
    for depth in reversed(range(unet.num_layers)):
        y = unet.upConvs[depth](F.leaky_relu(x))
        x = y + F.interpolate(
            networks.pad_or_crop(x, y.size(), unet.dimension),
            scale_factor=2, mode=unet.interpolate_mode, align_corners=False,
        )
        bn = unet.batchNorms[depth]
        reduce_dims = [0] + list(range(2, x.dim()))
        var, mean = torch.var_mean(x.detach(), dim=reduce_dims, unbiased=True)
        stats += [mean, var]
        x = F.batch_norm(x, None, None, bn.weight, bn.bias, True, 0.0, bn.eps)
        x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3], : skips[depth].size()[4]]
        x = torch.cat([x, skips[depth]], 1)
    x = unet.lastConv(x)
    return (x / 10, *stats)


def checkpoint_unet(unet):
    """Activation checkpointing for a tallUNet2 (~2.3 GiB per sample across the 6 UNets). The
    checkpointed part is side-effect free; BatchNorm's running-statistics update is applied
    once, outside it, with BatchNorm's own formula (same result up to float rounding)."""
    assert isinstance(unet, networks.UNet2) and unet.dimension == 3
    forward = unet.forward

    def checkpointed(x, y):
        if not (unet.training and torch.is_grad_enabled()):
            return forward(x, y)
        out, *stats = checkpoint(_unet2_forward_pure, unet, x, y)
        with torch.no_grad():
            for depth in reversed(range(unet.num_layers)):
                bn = unet.batchNorms[depth]
                mean, var = stats[2 * (unet.num_layers - 1 - depth): 2 * (unet.num_layers - depth)]
                bn.num_batches_tracked.add_(1)
                momentum = 1.0 / float(bn.num_batches_tracked) if bn.momentum is None else bn.momentum
                bn.running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
                bn.running_var.mul_(1 - momentum).add_(var, alpha=momentum)
        return out
    unet.forward = checkpointed
    return unet


def checkpoint_unets(net):
    """checkpoint_unet for every tallUNet2 (networks.UNet2) in net."""
    for m in net.modules():
        if type(m) is networks.UNet2:
            checkpoint_unet(m)
    return net


def prefetching(make_batch, depth=2):
    """make_batch, run in a background thread up to `depth` batches ahead: at batch 20 it costs
    ~0.5 s of CPU per step, which would otherwise leave the GPU idle. Same sampling code, so the
    same data distribution."""
    batches = queue.Queue(maxsize=depth)

    def worker():
        while True:
            batches.put(make_batch())
    threading.Thread(target=worker, daemon=True).start()
    return batches.get


def compile_training_forward(net):
    # Compile the training forward only; the validation pass every 300 steps (eval mode, batch 1,
    # and empty batches past BATCH_SIZE) stays eager rather than costing more compiles.
    # net stays a plain module, so checkpoints and attribute access are unchanged.
    # Modules whose forward tags their identity_map with .isIdentity (a shortcut ICON checks with
    # hasattr). Tagging them before each compiled call matches what eager has done by then, and keeps
    # dynamo's hasattr guards stable: otherwise the eager validation pass adds tags the first compile
    # did not see, and the training graph recompiles.
    tagging_modules = [m for m in net.modules()
                       if "identity_map.isIdentity = True" in inspect.getsource(type(m).forward)]
    compiled_forward = torch.compile(net.forward)
    eager_forward = net.forward

    def forward(*args):
        if not net.training:
            return eager_forward(*args)
        for m in tagging_modules:
            m.identity_map.isIdentity = True
        return compiled_forward(*args)
    net.forward = forward
    return net
