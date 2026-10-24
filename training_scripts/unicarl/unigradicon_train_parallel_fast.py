# FOOTSTEPS_NAME=<run name> torchrun --standalone --nproc_per_node=4 unigradicon_train_parallel_fast.py
# (see RUNGUIDE_fast.md)

# unigradicon_train_parallel.py's network (utp.make_net(3, input_shape, False): the conv
# coordinate-attention encoder), trained the same way, but faster and in less memory:
# torch.compile, activation checkpointing of the UNets, batches prepared in a background
# thread, and the library changes on this branch (matmul blurs with closed-form backwards,
# a leaner 3D warp, GradICON noise drawn on the GPU). None of these change what is computed;
# see fast_training.py and RUNGUIDE_fast.md.

import os
import icon_registration.unicarl.unigradicon_train_parallel as utp
import fast_training
import torch
import torch._functorch.config
import torch._inductor.config
os.environ["OMP_NUM_THREADS"]="8"
# every step has the same shapes, so let cuDNN pick the fastest conv algorithms once
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    import datasets
    from datasets import input_shape

    BATCH_SIZE = 20  # per GPU
    utp.BATCH_SIZE = BATCH_SIZE  # read by utp.make_make_pair

    net = utp.make_net(3, input_shape, False)
    # Checkpointing the 6 UNets plus inductor recomputing cheap ops in the compiled regions:
    # 3.6 GiB per sample instead of 7.1 (~75 GiB per GPU at batch 20), and still ~12% faster per
    # sample than the original eager script (torch 2.11, Quadro RTX 6000).
    fast_training.checkpoint_unets(net)
    torch._functorch.config.activation_memory_budget = 0.5
    # inductor's tiling heuristic hits an internal assertion on one of this network's graphs
    # (torch 2.11); it only chooses kernel tilings, so turning it off cannot change results
    torch._inductor.config.triton.coalesce_tiling_analysis = False
    fast_training.compile_training_forward(net)

    utp.train_batchfunction(
        net,
        fast_training.prefetching(utp.make_make_pair(datasets)),
        steps=450000
    )
