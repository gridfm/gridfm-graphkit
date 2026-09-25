#!/usr/bin/env bash
# LSF submitter for the GENCO paper PF runtime matrix.
# Four jobs, PF only. The paper OPF GENCO curves reuse these PF timings.
#
# Paper jobs 954945–954948 (2026-07-12) were the same four launchers, submitted
# as inline bsub: 1x exclusive H100, -M 128G, -n 40, span[hosts=1]. That inline
# command did not pin hosts; the jobs landed on cccxc713 (in-memory IEEE) and
# cccxc708 (the other three). This script adds the CCC 7xx host filter used for
# the PowerModels matrix and the GENCO environment dump.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${LSB_LOG_DIR:-$HOME/.lsbatch}"
PYTHON="${GENCO_PYTHON:-python}"
# Processed graphs: <data-path>/<network>/processed/data_index_*.pt
# Download: https://huggingface.co/datasets/gridfm/reproducibility-genco-pf-processed
DATA_PATH="${GENCO_DATA_PATH:-}"

HOST_SELECT="select[hname=='cccxc702' || hname=='cccxc703' || hname=='cccxc704' || hname=='cccxc705' || hname=='cccxc706' || hname=='cccxc707' || hname=='cccxc708' || hname=='cccxc709' || hname=='cccxc710' || hname=='cccxc711' || hname=='cccxc712' || hname=='cccxc713' || hname=='cccxc714' || hname=='cccxc715' || hname=='cccxc716']"

mkdir -p "$LOG_DIR"

if [[ -z "$DATA_PATH" || ! -d "$DATA_PATH" ]]; then
  echo "error: GENCO_DATA_PATH is not a directory (${DATA_PATH:-unset})." >&2
  echo "hf download gridfm/reproducibility-genco-pf-processed --repo-type dataset --local-dir /path/to/pf" >&2
  echo "export GENCO_DATA_PATH=/path/to/pf" >&2
  exit 1
fi

submit() {
  local name="$1"
  local launcher="$2"
  bsub -q normal \
    -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3" \
    -R "span[hosts=1] $HOST_SELECT" \
    -M 128G \
    -n 40 \
    -J "$name" \
    -o "$LOG_DIR/${name}_%J.out" \
    "cd '$REPO_ROOT' && export PYTHONUNBUFFERED=1 && '$PYTHON' '$launcher' --data-path '$DATA_PATH'"
}

submit genco_pf_in_memory_ieee "$SCRIPT_DIR/run_pf_in_memory_ieee.py"
submit genco_pf_in_memory_goc "$SCRIPT_DIR/run_pf_in_memory_goc.py"
submit genco_pf_from_disk_ieee "$SCRIPT_DIR/run_pf_from_disk_ieee.py"
submit genco_pf_from_disk_goc "$SCRIPT_DIR/run_pf_from_disk_goc.py"
