# Run guide: uniCARL attention encoder, 4 × 90 GB GPUs, batch 20 per GPU

Script: `training_scripts/unicarl/unigradicon_train_parallel_attention.py` on branch
`shard-datasets-attention` (based on `shard-datasets`). It has no flags. Batch size, steps and the
memory/speed settings are constants in the script.

## 1. Get the branch

The branch has not been pushed. It lives in `/home/tgreer/claudevm/ICON` on the VM, and there is a
bundle at `/home/tgreer/claudevm/shard-datasets-attention.bundle`. In your checkout (for example
`/playpen-raid1/tgreer/icon3/ICON`, which already has `training_scripts/unicarl/unicarl_public_train/`):

```
git fetch /path/to/shard-datasets-attention.bundle shard-datasets-attention:shard-datasets-attention
git checkout shard-datasets-attention
```

`datasets.py` loads the per-rank caches from `unicarl_public_train/` relative to the working directory.
The `*_rank{0..3}_of_4_cached_dataset.trch` caches match 4 GPUs. With a fresh clone, symlink that directory into
`training_scripts/unicarl/`.

## 2. Environment (torch 2.11)

```
python3 -m venv venv211
venv211/bin/pip install torch==2.11.0 torchvision tensorboard tqdm matplotlib "footsteps>=0.1.6" itk girder_client==3.1.8
venv211/bin/pip install -e .          # from the ICON checkout, so the branch's src/ is what runs
```

The library changes on this branch (blurs, warping, `spacing`, the padded identity map) are needed.
An ICON install from PyPI or from another branch will not compile.

## 3. Launch

```
cd training_scripts/unicarl
FOOTSTEPS_NAME=attn_bs20 ../../venv211/bin/torchrun --standalone --nproc_per_node=4 unigradicon_train_parallel_attention.py
```

`torchrun` replaces `torch.distributed.launch`. Use `CUDA_VISIBLE_DEVICES` to choose the GPUs. Results,
tensorboard logs and `network_weights_<step>` (every 5000 steps, same `regis_net` state-dict format as
before) go to `results/attn_bs20/`.

## 4. What to expect

- **Host RAM:** each rank loads its quarter of the caches (~29 GB of the 116 GB) before step 0.
- **Step 0:** each rank compiles for about 3–5 minutes, then the first validation runs eagerly. After that,
  no recompiles. The inductor cache in `/tmp/torchinductor_$USER` makes restarts faster.
- **GPU memory:** about 2.3 GiB + 3.5 GiB × 20 ≈ 72 GiB per GPU. This is measured on a 24 GB Quadro RTX 6000 at batch
  1/2/4 (linear) and extrapolated.
- **Speed:** about 1.14 s per sample on the RTX 6000, so about 23 s per step at batch 20 there. Hopper should be several times faster. The
  untuned starting point was 1.61 s per sample.
- **Validation output:** every 300 steps it prints 8 per-image losses. These were NaN beyond the batch size when the batch was below 8 (library
  quirk, harmless); at batch 20 all 8 are real.

## 5. If it doesn't fit, or for more speed

The knobs are in the script's `__main__` and `CHECKPOINT`:

| setting | GiB/sample | s/sample (RTX 6000) |
|---|---|---|
| shipped: `CHECKPOINT = True`, `activation_memory_budget = 0.5`, compiled | 3.5 | 1.11–1.14 |
| `activation_memory_budget = 0.1` | 3.2 | 1.14 |
| `CHECKPOINT = False` (no `activation_memory_budget`), compiled | 9.5 | 0.86 (fastest; at most batch 8 on 90 GB) |

If it runs out of memory, lower `BATCH_SIZE` or use the 0.1 budget. If you see fragmentation OOMs,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps.

## 6. Not tested here (the VM has one Turing GPU)

- **Multi-GPU DDP:** tested only as `torchrun --nproc_per_node=1`. The 4-rank NCCL path is the library's
  unchanged `train_batchfunction`.
- **bf16:** on Ampere/Hopper the transformer blocks use bf16 (flash attention); here they ran fp16. Watch the
  first few hundred steps' losses.
- **Real data through the full loop:** the smoke tests used random volumes. `make_pair` timing was checked on the
  real HAN-Seg cache.

## What changed versus `unigradicon_train_parallel.py`, and why the result is the same

- **Transformer coordinate-attention encoder** (the point of the script). It uses mixed precision, centred features,
  and B encoded without padding.
- **3D blurs:** matrix products with a hand-written backward. They equal the conv blurs to 1e-16 in float64,
  including gradients, and save nothing for the backward.
- **GradICON noise:** drawn on the GPU at the sampled points only. Same distribution.
- **3D warping:** saves the coordinates instead of a reordered grid copy. Bit-identical forward,
  and gradients within `grid_sample`'s own atomic-add noise.
- **torch.compile:** losses agree with eager to 6–7 digits, gradients to 0.17% (fp32 rounding).
- **Activation checkpointing** (encoder blocks, UNets): gradients and BatchNorm running statistics are within
  run-to-run noise of no checkpointing, and `num_batches_tracked` is identical.
- **cuDNN autotuning, prefetching:** same math and same sampling.
