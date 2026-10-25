#!/bin/bash
# Train and score every ablation arm in order, then summarize. Each step is a footsteps run under
# results/ (ablation_<arm>/, ablation_<arm>_eval/, ablation_summary/). Arms whose training or
# evaluation already finished are skipped, so the script can be restarted.
# PYTHON: the torch 2.11 venv's python. EVAL_GPU: the GPU for evaluation (default 0).
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
for arm in $($PYTHON -c "import ablation; print(' '.join(ablation.ARMS))"); do
  if [ -z "$($PYTHON ablation.py latest ablation_$arm network_weights_final)" ]; then
    $PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 ablation.py train $arm
  fi
  run=$($PYTHON ablation.py latest ablation_$arm network_weights_final)
  if [ -z "$($PYTHON ablation.py latest ablation_${arm}_eval gold_atlas_raw_io50.csv)" ]; then
    CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} $PYTHON ablation.py eval $arm $run/network_weights_final
  fi
done
$PYTHON ablation.py summary
