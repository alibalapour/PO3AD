#!/bin/bash
#SBATCH --job-name=ASN_OffsetHead_NormAS_Ablation
#SBATCH --account=def-fhach
#SBATCH --mail-user=alibalapour93.ab@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --array=0-89                  # 2 sets × 3 hidden dims × 15 categories = 90 jobs
#SBATCH --output=logs/asn_offset_norm_as_ablation_%A_%a.out
#SBATCH --error=logs/asn_offset_norm_as_ablation_%A_%a.err


set -euo pipefail

# =============================================================================
# ABLATION STUDY: OFFSET HEAD ARCHITECTURES (Norm-AS only)
# ON ANOMALYSHAPENET DATASET
# =============================================================================
#
# This script compares offset head architectures (baseline vs multi_head)
# using the Norm-AS (original normal-displacement) anomaly synthesis module.
#
# 2 sets of analysis:
#   Set 0: baseline   + Norm-AS
#   Set 1: multi_head + Norm-AS
#
# Hidden dimensions tested: 32, 64, 128
#
# Categories (15 with index 0):
#   ashtray0, bag0, bottle0, bowl0, bucket0, cap0, cup0, eraser0,
#   headset0, helmet0, jar0, microphone0, shelf0, tap0, vase0
#
# Total experiments: 2 sets × 3 hidden dims × 15 categories = 90 jobs
#
# Runtime is reported at training end and evaluation end using $SECONDS.
#
# =============================================================================

echo "=== Job started at $(date) on $(hostname) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi || true

# -------------------------
# Modules (adjust as needed for your cluster)
# -------------------------
module purge
module load StdEnv/2020
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.9
module load flexiblas

# -------------------------
# Activate virtualenv
# -------------------------
VENV_PATH="${SLURM_VENV_PATH:-$HOME/envs/me-torch19-py39}"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
python --version

# -------------------------
# Configure threading for deterministic performance
# -------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# =============================================================================
# EXPERIMENT SETS (2 total — Norm-AS only)
# =============================================================================

declare -a SET_VARIANTS=(
    "baseline"
    "multi_head"
)

declare -a SET_LABELS=(
    "norm_AS"
    "norm_AS"
)

NUM_SETS=${#SET_VARIANTS[@]}

# =============================================================================
# HIDDEN DIMENSIONS (3 total)
# =============================================================================

declare -a HIDDEN_DIMS=(
    32
    64
    128
)

# =============================================================================
# CATEGORIES (15 total — AnomalyShapeNet classes with index 0)
# =============================================================================

declare -a CATEGORIES=(
    "ashtray0"
    "bag0"
    "bottle0"
    "bowl0"
    "bucket0"
    "cap0"
    "cup0"
    "eraser0"
    "headset0"
    "helmet0"
    "jar0"
    "microphone0"
    "shelf0"
    "tap0"
    "vase0"
)

# =============================================================================
# VOXEL SIZES FOR EVALUATION
# =============================================================================

declare -a VOXEL_SIZES=(
    0.03
)

NUM_HIDDEN_DIMS=${#HIDDEN_DIMS[@]}
NUM_CATEGORIES=${#CATEGORIES[@]}

# =============================================================================
# MAP SLURM ARRAY TASK ID → (SET, HIDDEN_DIM, CATEGORY)
# =============================================================================

TASK_ID=${SLURM_ARRAY_TASK_ID}

JOBS_PER_SET=$((NUM_HIDDEN_DIMS * NUM_CATEGORIES))       # 3 × 15 = 45
JOBS_PER_HIDDEN=$((NUM_CATEGORIES))                       # 15

SET_IDX=$((TASK_ID / JOBS_PER_SET))
REMAINING=$((TASK_ID % JOBS_PER_SET))
HIDDEN_DIM_IDX=$((REMAINING / JOBS_PER_HIDDEN))
CATEGORY_IDX=$((REMAINING % JOBS_PER_HIDDEN))

if [ $SET_IDX -ge $NUM_SETS ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($TASK_ID) exceeds number of experiments ($((NUM_SETS * NUM_HIDDEN_DIMS * NUM_CATEGORIES)))"
    exit 1
fi

VARIANT="${SET_VARIANTS[$SET_IDX]}"
SET_LABEL="${SET_LABELS[$SET_IDX]}"
HIDDEN_DIM="${HIDDEN_DIMS[$HIDDEN_DIM_IDX]}"
CATEGORY="${CATEGORIES[$CATEGORY_IDX]}"

# =============================================================================
# NORM-AS TRAINING PARAMETERS
# =============================================================================
EPOCHS_VAR=300
OPTIMIZER_VAR="Adam"
LR_VAR=0.0008
SEED_ARG="--manual_seed 42"

DATASET="AnomalyShapeNet"
BASE_LOGPATH="./log/asn_offset_norm_as_ablation/${CATEGORY}/${SET_LABEL}/${VARIANT}/hidden_${HIDDEN_DIM}"

echo "=============================================="
echo "Offset Head Ablation Study — Norm-AS (AnomalyShapeNet)"
echo "=============================================="
echo "  Set:          ${SET_IDX} (${VARIANT} + ${SET_LABEL})"
echo "  Variant:      ${VARIANT}"
echo "  Hidden Dim:   ${HIDDEN_DIM}"
echo "  Category:     ${CATEGORY}"
echo "  Dataset:      ${DATASET}"
echo "  Task ID:      ${TASK_ID} (Set ${SET_IDX}, Hidden ${HIDDEN_DIM_IDX}, Category ${CATEGORY_IDX})"
echo "  Log Path:     ${BASE_LOGPATH}"
echo "=============================================="

# =============================================================================
# OFFSET HEAD SPECIFIC ARGUMENTS
# =============================================================================

OFFSET_ARGS="--offset_head_variant ${VARIANT} --offset_hidden_dim ${HIDDEN_DIM} --offset_num_layers 3 --offset_dropout 0.0"

# =============================================================================
# RUN TRAINING (with runtime measurement)
# =============================================================================

echo ""
echo "=== Starting training at $(date) ==="
SECONDS=0

python train.py \
  --dataset AnomalyShapeNet --category ${CATEGORY} \
  --optimizer ${OPTIMIZER_VAR} --logpath "${BASE_LOGPATH}/" --gpu_id 0 \
  --epochs ${EPOCHS_VAR} \
  --lr ${LR_VAR} \
  --data_repeat 600 \
  --batch_size 32 \
  --num_works 16 \
  --voxel_size 0.03 \
  --save_freq 50 \
  --mask_num 64 \
  ${SEED_ARG} \
  ${OFFSET_ARGS}

TRAIN_SECONDS=$SECONDS
TRAIN_H=$((TRAIN_SECONDS / 3600))
TRAIN_M=$(( (TRAIN_SECONDS % 3600) / 60 ))
TRAIN_S=$((TRAIN_SECONDS % 60))

echo "=== Training completed at $(date) ==="
echo "=== Training runtime: ${TRAIN_H}h ${TRAIN_M}m ${TRAIN_S}s (${TRAIN_SECONDS}s total) ==="
echo "Results saved to: ${BASE_LOGPATH}/"

# =============================================================================
# RUN EVALUATION ON ALL CHECKPOINTS WITH MULTIPLE VOXEL SIZES
# =============================================================================

echo ""
echo "=== Starting evaluation on all checkpoints ==="
SECONDS=0

CKPT_DIR="${BASE_LOGPATH}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "Warning: Checkpoint directory does not exist: ${CKPT_DIR}"
    echo "Skipping evaluation."
else
    ALL_CKPTS=$(find "${CKPT_DIR}" -name "*.pth" -type f 2>/dev/null | sort)

    if [ -z "${ALL_CKPTS}" ]; then
        echo "Warning: No checkpoints found in ${CKPT_DIR}"
        echo "Skipping evaluation."
    else
        echo "Found checkpoints:"
        echo "${ALL_CKPTS}"

        RESULTS_DIR="${BASE_LOGPATH}/eval_results"
        mkdir -p "${RESULTS_DIR}"

        for CKPT_PATH in ${ALL_CKPTS}; do
            CKPT_NAME=$(basename "${CKPT_PATH}" .pth)
            echo ""
            echo "========================================"
            echo "Evaluating checkpoint: ${CKPT_NAME}"
            echo "========================================"

            CHECKPOINT_NAME="${CKPT_PATH#./}"

            for VOXEL_SIZE in "${VOXEL_SIZES[@]}"; do
                echo ""
                echo "--- Testing with voxel_size=${VOXEL_SIZE} ---"

                EVAL_OUTPUT="${RESULTS_DIR}/${CKPT_NAME}_voxel${VOXEL_SIZE}.txt"

                echo "Running evaluation..."
                python eval.py \
                    --dataset AnomalyShapeNet \
                    --category ${CATEGORY} \
                    --checkpoint_name "${CHECKPOINT_NAME}" \
                    --logpath ./ \
                    --voxel_size ${VOXEL_SIZE} \
                    --offset_head_variant ${VARIANT} \
                    --offset_hidden_dim ${HIDDEN_DIM} \
                    2>&1 | tee "${EVAL_OUTPUT}"

                echo "Results saved to: ${EVAL_OUTPUT}"
            done
        done

        echo ""
        echo "=== All evaluations completed at $(date) ==="
        echo "Results saved to: ${RESULTS_DIR}/"
    fi
fi

EVAL_SECONDS=$SECONDS
EVAL_H=$((EVAL_SECONDS / 3600))
EVAL_M=$(( (EVAL_SECONDS % 3600) / 60 ))
EVAL_S=$((EVAL_SECONDS % 60))

TOTAL_SECONDS=$((TRAIN_SECONDS + EVAL_SECONDS))
TOTAL_H=$((TOTAL_SECONDS / 3600))
TOTAL_M=$(( (TOTAL_SECONDS % 3600) / 60 ))
TOTAL_S=$((TOTAL_SECONDS % 60))

# =============================================================================
# RUNTIME SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "=== Job Summary ==="
echo "=============================================="
echo "  Set:            ${SET_IDX} (${VARIANT} + ${SET_LABEL})"
echo "  Variant:        ${VARIANT}"
echo "  Anomaly Module: ${SET_LABEL}"
echo "  Hidden Dim:     ${HIDDEN_DIM}"
echo "  Category:       ${CATEGORY}"
echo "  Dataset:        ${DATASET}"
echo "  Log Path:       ${BASE_LOGPATH}"
echo "----------------------------------------------"
echo "  Training Time:    ${TRAIN_H}h ${TRAIN_M}m ${TRAIN_S}s (${TRAIN_SECONDS}s)"
echo "  Evaluation Time:  ${EVAL_H}h ${EVAL_M}m ${EVAL_S}s (${EVAL_SECONDS}s)"
echo "  Total Time:       ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s (${TOTAL_SECONDS}s)"
echo "  Completed:        $(date)"
echo "=============================================="
