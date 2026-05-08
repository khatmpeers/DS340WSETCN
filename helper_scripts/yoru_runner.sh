#!/usr/bin/env bash
set -u

PYTHON_BIN="$PWD/venv/bin/python"
TRAIN_SCRIPT="train.py"

DATE_COL="time_unix"
EPOCHS=10
BATCH_SIZE=64
LR=1e-4
MAX_ROWS=2000
LOG_EVERY=500

mkdir -p logs outputs

run_job() {
  local name="$1"
  shift

  echo "=================================================="
  echo "Starting: $name"
  echo "Time: $(date)"
  echo "Command: $PYTHON_BIN $TRAIN_SCRIPT $*"
  echo "=================================================="

  "$PYTHON_BIN" "$TRAIN_SCRIPT" "$@" \
    > "logs/${name}.log" 2>&1

  local status=$?
  if [ $status -eq 0 ]; then
    echo "[OK] $name finished at $(date)"
  else
    echo "[FAIL] $name exited with status $status at $(date)"
  fi

  return 0
}

run_job "top5_setcn_10ep" \
  --model setcn \
  --train_dir sampdata_sampled_top5/train \
  --val_dir sampdata_sampled_top5/val \
  --test_dir sampdata_sampled_top5/test \
  --date_column "$DATE_COL" \
  --feature_columns \
    traction_tractionForce \
    odometry_vehicleSpeed \
    odometry_wheelSpeed_fr \
    odometry_wheelSpeed_ml \
    odometry_wheelSpeed_mr \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --max_rows_per_trip "$MAX_ROWS" \
  --log_every_batches "$LOG_EVERY" \
  --output_dir outputs/top5_setcn_10ep

# 2) Temporal sparsity run: 2x sparse dataset
run_job "sparse2x_setcn_10ep" \
  --model setcn \
  --train_dir sampdata_sampled_2xs/train \
  --val_dir sampdata_sampled_2xs/val \
  --test_dir sampdata_sampled_2xs/test \
  --date_column "$DATE_COL" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --max_rows_per_trip "$MAX_ROWS" \
  --log_every_batches "$LOG_EVERY" \
  --output_dir outputs/sparse2x_setcn_10ep

echo "All queued experiments finished at $(date)"
