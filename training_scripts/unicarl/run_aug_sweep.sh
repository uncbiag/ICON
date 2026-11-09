#!/bin/bash
# Augmentation-scale sweep: yeetle (ablation arm 8) trained for 15k steps at each RandomMatrix scale in
# ablation.AUG_SCALES (0 = no augmentation ... 0.125; yeetle's is 0.1), each scored on Gold Atlas (raw, IO-50)
# and, when ANATOMIX_EVAL_PACK (default ./anatomix_eval_pack.trch) exists, on anatomix EPE; then the summary.
# Footsteps runs under results/ (ablation_aug<s>/, ablation_aug<s>_eval/, ablation_aug<s>_anatomix/). Finished
# steps are skipped, so the script can be restarted.
# PYTHON: the torch 2.11 venv's python. EVAL_GPU: the GPU for evaluation (default 0).
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
for arm in $($PYTHON -c "import ablation; print(' '.join(ablation.AUG_SWEEP))"); do
  if [ -z "$($PYTHON ablation.py latest ablation_$arm network_weights_final)" ]; then
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 ablation.py train $arm
  fi
  run=$($PYTHON ablation.py latest ablation_$arm network_weights_final)
  if [ -z "$($PYTHON ablation.py latest ablation_${arm}_eval gold_atlas_raw_io50.csv)" ]; then
    CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} $PYTHON ablation.py eval $arm $run/network_weights_final
  fi
  if [ -f "${ANATOMIX_EVAL_PACK:-anatomix_eval_pack.trch}" ] && [ -z "$($PYTHON ablation.py latest ablation_${arm}_anatomix anatomix_epe.csv)" ]; then
    CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} $PYTHON ablation.py eval_anatomix $arm $run/network_weights_final
  fi
done
$PYTHON ablation.py summary
