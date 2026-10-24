# Run guide: uniCARL (original conv architecture), fast build, 4 × 90 GB GPUs, batch 20 per GPU

Script: `training_scripts/unicarl/unigradicon_train_parallel_fast.py` on branch `shard-datasets-attention`.
It trains the same network as `unigradicon_train_parallel.py` (`utp.make_net(3, input_shape, False)`),
with the same loss, data and optimizer, faster and in less memory. It has no flags; batch size and steps
are constants in the script. Shared machinery is in `fast_training.py`.

## 1. Get the branch

The branch has not been pushed. It is in `/home/tgreer/claudevm/ICON` on the VM, and there is a
bundle at `/home/tgreer/claudevm/shard-datasets-attention.bundle`. The bundle needs commit `5c84f94`,
which `/playpen-raid1/tgreer/icon3/ICON` has. In that checkout:

```
git fetch /path/to/shard-datasets-attention.bundle shard-datasets-attention:shard-datasets-attention
git checkout shard-datasets-attention
```

`datasets.py` loads the per-rank caches from `unicarl_public_train/` relative to the working directory.
The `*_rank{0..3}_of_4_*` caches match 4 GPUs. With a fresh clone, symlink that directory into
`training_scripts/unicarl/`.

## 2. Environment: torch 2.11

```
python3 -m venv venv211
venv211/bin/pip install torch==2.11.0 torchvision tensorboard tqdm matplotlib "footsteps>=0.1.6" itk girder_client==3.1.8
venv211/bin/pip install -e .          # from the ICON checkout: the branch's src/ must be what runs
```

torch 2.11 rather than 2.14: for this configuration 2.14 was 13% slower per sample and compiled in
16 min instead of 8. The 2.7.1 venv in `results/yeetle` cannot compile this setup.

## 3. Launch

```
cd training_scripts/unicarl
FOOTSTEPS_NAME=fast_bs20 ../../venv211/bin/torchrun --standalone --nproc_per_node=4 unigradicon_train_parallel_fast.py
```

Use `CUDA_VISIBLE_DEVICES` to choose GPUs. Results, tensorboard logs and `network_weights_<step>` (every
5000 steps, same `regis_net` state-dict format as before, so evaluation code is unchanged) go to
`results/fast_bs20/`.

## 4. What to expect

- **Host RAM:** each rank loads its quarter of the caches (~29 GB) before step 0.
- **Step 0:** each rank compiles for about 8 min, then the first validation runs eagerly. No recompiles after that. The
  inductor cache (`/tmp/torchinductor_$USER`) makes restarts faster.
- **GPU memory:** about 2.2 GiB + 3.6 GiB × 20 ≈ 75 GiB per GPU (measured at batch 2/4 on a 24 GB Quadro RTX 6000,
  linear, extrapolated). This model runs in fp32 throughout, so the figure should carry over to Hopper.
- **Speed (RTX 6000, torch 2.11):**

  | | s/sample | GiB/sample |
  |---|---|---|
  | original `unigradicon_train_parallel.py` (shard-datasets library, eager) | 1.23–1.37 | 7.1 |
  | this script | 1.08–1.11 | 3.6 |
  | this script with `checkpoint_unets` and the budget removed (most it fits is batch 10 on 90 GB) | 0.92 | 7.1 |

- **Validation output:** every 300 steps it prints 8 per-image losses; at batch 20 all 8 are real. Below batch 8 the
  extra ones were NaN, a harmless library quirk.

## 5. If it doesn't fit

- Lower `BATCH_SIZE`.
- Set `activation_memory_budget = 0.1` (about 0.3 GiB less per sample, a few % slower).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with fragmentation OOMs.

## 6. What changed, and why the result is the same

All of these were checked against the original code:

- **3D blurs** (`fpc.Blur`, LNCC): matrix products with a hand-written transposed backward. They equal the conv
  blurs to 1e-16 in float64, forward and gradient, and save nothing for the backward.
- **GradICON noise:** drawn on the GPU, only at the subsampled points it keeps. Same distribution.
- **3D warping:** saves the coordinates and rebuilds the sampling grid in the backward. The forward is bit-identical;
  gradients differ only by `grid_sample`'s own atomic-add nondeterminism.
- **`spacing`** is stored as a tuple of floats and the padded identity map is built once. Both are needed for
  `torch.compile`, and the values are identical.
- **torch.compile:** losses agree with eager to 6–7 digits, and gradients to 0.17% (fp32 rounding).
- **UNet activation checkpointing:** the checkpointed UNet is side-effect free, and BatchNorm's running statistics are
  updated once per step outside the checkpoint. Gradients and running statistics are within run-to-run noise of
  no checkpointing, and `num_batches_tracked` is identical.
- **cuDNN autotuning and batch prefetching in a background thread:** same math, same sampling.

## 7. Not tested here (the VM has one Turing GPU)

- **4-rank DDP:** tested only as `torchrun --nproc_per_node=1`. The DDP code is the library's unchanged
  `train_batchfunction`.
- **Real data through the full loop:** the smoke tests used random volumes. `make_pair` was timed on the real
  HAN-Seg cache.
