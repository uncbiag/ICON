# Run guide: uniGradICON → yeetle ablation

`ablation.py` trains a chain of 8 arms. Each arm changes one thing from the one before it. Every arm gets
yeetle's budget: 15k steps, batch 4 per GPU on 4 GPUs. Each arm is scored on Gold Atlas T2 MR → native
CT bladder Dice (the raw setting, IO-50, all 17 patients), using the same protocol as `prostate_bench/bench.py --setting raw --io 50`.

| arm | change from the previous arm |
|---|---|
| `1_ugi` | uniGradICON recipe: its UNet pyramid, LNCC σ5, lr 5e-5, flip/permute + affine-noise augmentation, two-phase schedule (final full-res step added for the last 20%). Data from the private zip: COPDGene insp/exp pairs (`dirlab_clamped`), OAI knee, HCP T1 (stripped), AbdomenCT-1K |
| `2_attn` | attention coarse stage in place of the coarsest UNet |
| `3_public_unpaired` | yeetle's public datasets, unpaired ones only |
| `4_public_all` | + the same-patient paired datasets, which gives yeetle's data |
| `5_augmentify` | yeetle's augmentation (random rotation + shear around the network) |
| `6_sqlncc` | SquaredLNCC σ4 |
| `7_lr` | lr 2e-5 |
| `8_yeetle` | yeetle's network (CARL rotation fields, blurs, its step layout, final step from the start), which makes it yeetle |

Arm 1 is expected to score about 0: without the attention stage nothing can find the raw-setting offset of up to 200 mm.
It anchors the chain, and arm 2 measures how much of the gap the attention stage explains. Later changes are
credited in chain order, so an interaction lands on whichever arm comes later.

**Not ablatable from the zips:** both zips store volumes already foreground-cropped, p1–p99 clipped and resized
to 160³. Every arm therefore uses yeetle's preprocessing at 160³, where uniGradICON used the whole volume at 175³. Also, for arm 1:

- **Lung:** uniGradICON used COPD lung-ROI-only images; the HU-clamped `dirlab_clamped` is the closest available.
- **Abdomen:** it used Learn2Reg AbdomenCTCT, which isn't in the zips; AbdomenCT-1K stands in.

## Setup (on the training server)

1. **Code:** check out branch `shard-datasets-attention` (bundle: `/home/tgreer/claudevm/shard-datasets-attention.bundle`,
   applies on top of `5c84f94`), then `pip install -e .` into a torch 2.11 venv, as in `RUNGUIDE_fast.md` §2.
2. **Data:** in `training_scripts/unicarl/`:
   ```
   unzip unicarl_public_train.zip      # -> unicarl_public_train/   (the 4-rank shards)
   unzip unicarl_private.zip           # -> results/unicarl_private/
   ```
3. **Gold Atlas:** copy `prostate_bench/data/<pid>/{t2,ct_raw,t2_bladder,ct_raw_bladder}.nii.gz` for the 17 patients,
   and set `GOLD_ATLAS_DATA` to that directory.

## Run

```
cd training_scripts/unicarl
PYTHON=/path/to/venv211/bin/python GOLD_ATLAS_DATA=/path/to/gold_atlas bash run_ablation.sh
```

This trains each arm on 4 GPUs, evaluates it on `EVAL_GPU` (default 0), then writes the summary. It is restartable:
arms whose training or evaluation already finished are skipped. Single steps:

```
torchrun --standalone --nproc_per_node=4 ablation.py train 2_attn
python ablation.py eval 2_attn results/ablation_2_attn/network_weights_final
python ablation.py summary
```

## Outputs: footsteps runs under `results/`

- `ablation_<arm>/`: `info.txt` (command, host, git hash, uncommitted diff), a copy of `ablation.py`,
  `package_versions.txt`, `config.json` (the arm's settings), `tensorboard/`, `network_weights_{5000,10000,15000,final}`.
- `ablation_<arm>_eval/`: the same footsteps record, plus `config.json` (with the path of the weights it scored) and
  `gold_atlas_raw_io50.csv` (per-patient Dice).
- `ablation_summary/summary.txt`: mean/median/min per arm, and which eval run each number came from.

Reruns never overwrite: footsteps adds a `-N` suffix, and the launcher and summary use the newest complete run.

## Cost

- **Host RAM:** arms 1–2 load about 13 GB per rank; arms 3–8 load yeetle's 29 GB public shard per rank.
- **GPU time:** yeetle took 15.5 h on biag-gpu6 for 15k steps; expect less on a faster server, and the uniGradICON-network arms are smaller.
- **Evaluation:** IO-50 takes about 1.5 min per patient on a Quadro RTX 6000, so about 25 min per arm.

## Caveats

- **One seed per arm:** run-to-run noise isn't measured. If two adjacent arms differ by a few hundredths, treat that as unresolved.
- **IO-50:** IO changes the result substantially. All arms use exactly the same IO settings (50 steps, the arm's own loss and λ=1.5).
