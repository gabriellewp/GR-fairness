#!/usr/bin/env bash
#SBATCH --job-name=grid-independent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/independent_%j.out


# Run the verified standalone GRID pipeline using only this repository's code
# and generated artifacts. Set GRID_PYTHON to choose your environment.

set -euo pipefail
set -o pipefail

if [[ -n "${FAIRDIVERSE_GRID_ROOT:-}" ]]; then
    GRID_ROOT="$(cd "${FAIRDIVERSE_GRID_ROOT}" && pwd)"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Slurm executes a spool copy of this file, so BASH_SOURCE points below
    # /var/spool. Submit from GRID/ or set FAIRDIVERSE_GRID_ROOT explicitly.
    GRID_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
else
    GRID_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "${GRID_ROOT}/.project-root" ]]; then
    echo "ERROR: GRID_ROOT is not the embedded GRID directory: ${GRID_ROOT}" >&2
    echo "Set FAIRDIVERSE_GRID_ROOT to the absolute embedded GRID path." >&2
    exit 1
fi
if [[ -n "${GRID_PYTHON:-}" ]]; then
    PYTHON_BIN="${GRID_PYTHON}"
else
    PYTHON_BIN="python"
fi

TOKENIZER="${GRID_TOKENIZER:-rkmeans}"
DATASET="${GRID_DATASET:-beauty}"
# The release uses complete metadata for every dataset and tokenizer.
METADATA_TEXT_MODE="${GRID_METADATA_TEXT_MODE:-complete_metadata}"
EMBEDDING_PATH_OVERRIDE="${GRID_EMBEDDING_PATH:-}"
EMBEDDING_NAME="${GRID_EMBEDDING_NAME:-}"
DATA_DIR_OVERRIDE="${GRID_DATA_DIR_OVERRIDE:-}"
OUTPUT_RUN_NAME="${GRID_OUTPUT_RUN_NAME:-}"
TIGER_CKPT_OVERRIDE="${GRID_TIGER_CKPT:-}"
TIGER_MAX_STEPS="${GRID_TIGER_MAX_STEPS:-}"
TIGER_VAL_CHECK_INTERVAL="${GRID_TIGER_VAL_CHECK_INTERVAL:-}"
TIGER_EARLY_STOP_PATIENCE="${GRID_TIGER_EARLY_STOP_PATIENCE:-}"
TIGER_SELECTION_METRIC="${GRID_TIGER_SELECTION_METRIC:-}"
TIGER_RUN_TAG="${GRID_TIGER_RUN_TAG:-}"
TIGER_EVAL_WORKERS="${GRID_TIGER_EVAL_WORKERS:-}"
TIGER_EVAL_TIMEOUT="${GRID_TIGER_EVAL_TIMEOUT:-}"
TIGER_STRATEGY="${GRID_TIGER_STRATEGY:-}"
PREPROCESS_ONLY="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --dataset requires beauty, toys, sports, or pet-supplies" >&2
                exit 2
            fi
            DATASET="$2"
            shift 2
            ;;
        --tokenizer)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tokenizer requires rkmeans, rvq, or rqvae" >&2
                exit 2
            fi
            TOKENIZER="$2"
            shift 2
            ;;
        --metadata-text-mode)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --metadata-text-mode requires complete_metadata" >&2
                exit 2
            fi
            METADATA_TEXT_MODE="$2"
            shift 2
            ;;
        --embedding-path)
            EMBEDDING_PATH_OVERRIDE="$2"
            shift 2
            ;;
        --embedding-name)
            EMBEDDING_NAME="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --output-run-name)
            OUTPUT_RUN_NAME="$2"
            shift 2
            ;;
        --tiger-ckpt)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-ckpt requires a checkpoint path" >&2
                exit 2
            fi
            TIGER_CKPT_OVERRIDE="$2"
            shift 2
            ;;
        --tiger-max-steps)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-max-steps requires a positive integer" >&2
                exit 2
            fi
            TIGER_MAX_STEPS="$2"
            shift 2
            ;;
        --tiger-val-check-interval)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-val-check-interval requires a positive number of training batches" >&2
                exit 2
            fi
            TIGER_VAL_CHECK_INTERVAL="$2"
            shift 2
            ;;
        --tiger-early-stop-patience)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-early-stop-patience requires a positive number of validation checks" >&2
                exit 2
            fi
            TIGER_EARLY_STOP_PATIENCE="$2"
            shift 2
            ;;
        --tiger-selection-metric)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-selection-metric requires val/ndcg@10 or val/recall@5" >&2
                exit 2
            fi
            TIGER_SELECTION_METRIC="$2"
            shift 2
            ;;
        --tiger-run-tag)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-run-tag requires a tag" >&2
                exit 2
            fi
            TIGER_RUN_TAG="$2"
            shift 2
            ;;
        --tiger-eval-workers)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-eval-workers requires a positive integer" >&2
                exit 2
            fi
            TIGER_EVAL_WORKERS="$2"
            shift 2
            ;;
        --tiger-eval-timeout)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-eval-timeout requires a positive number of seconds" >&2
                exit 2
            fi
            TIGER_EVAL_TIMEOUT="$2"
            shift 2
            ;;
        --tiger-strategy)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tiger-strategy requires auto or ddp" >&2
                exit 2
            fi
            TIGER_STRATEGY="$2"
            shift 2
            ;;
        --preprocess-only)
            PREPROCESS_ONLY="true"
            shift
            ;;
        --help|-h)
            echo "Usage: sbatch $0 [--dataset beauty|toys|sports|pet-supplies] [--tokenizer rkmeans|rvq|rqvae]"
            echo "                [--metadata-text-mode complete_metadata] [--preprocess-only]"
            echo "                [--embedding-path FILE --embedding-name NAME] [--data-dir DIR]"
            echo "                [--output-run-name NAME]"
            echo "                [--tiger-ckpt FILE] [--tiger-max-steps STEPS]"
            echo "                [--tiger-val-check-interval BATCHES] [--tiger-early-stop-patience CHECKS]"
            echo "                [--tiger-selection-metric val/ndcg@10|val/recall@5]"
            echo "                [--tiger-run-tag NAME] (separate TIGER training and inference artifacts)"
            echo "                [--tiger-eval-workers N] [--tiger-eval-timeout SECONDS]"
            echo "                [--tiger-strategy auto|ddp]"
            echo "Environment alternatives: GRID_DATASET, GRID_TOKENIZER, GRID_METADATA_TEXT_MODE,"
            echo "GRID_EMBEDDING_PATH, GRID_EMBEDDING_NAME, GRID_DATA_DIR_OVERRIDE,"
            echo "GRID_OUTPUT_RUN_NAME, GRID_TIGER_CKPT, GRID_TIGER_MAX_STEPS,"
            echo "GRID_TIGER_VAL_CHECK_INTERVAL, GRID_TIGER_EARLY_STOP_PATIENCE,"
            echo "GRID_TIGER_SELECTION_METRIC, and GRID_TIGER_RUN_TAG"
            echo "GRID_TIGER_EVAL_WORKERS, GRID_TIGER_EVAL_TIMEOUT, and GRID_TIGER_STRATEGY"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "${TOKENIZER}" in
    rkmeans|kmeans)
        TOKENIZER="rkmeans"
        TOKENIZER_TRAIN_EXPERIMENT="rkmeans_train_flat"
        TOKENIZER_INFERENCE_EXPERIMENT="rkmeans_inference_flat"
        ;;
    rvq)
        TOKENIZER_TRAIN_EXPERIMENT="rvq_train_flat"
        TOKENIZER_INFERENCE_EXPERIMENT="rvq_inference_flat"
        ;;
    rqvae)
        TOKENIZER_TRAIN_EXPERIMENT="rqvae_train_flat"
        TOKENIZER_INFERENCE_EXPERIMENT="rqvae_inference_flat"
        ;;
    *)
        echo "ERROR: Unsupported tokenizer '${TOKENIZER}'. Choose rkmeans, rvq, or rqvae." >&2
        exit 2
        ;;
esac

if [[ "${METADATA_TEXT_MODE}" == "complete" ]]; then
    echo "WARNING: metadata mode 'complete' is deprecated; using 'complete_metadata'." >&2
    METADATA_TEXT_MODE="complete_metadata"
fi
case "${METADATA_TEXT_MODE}" in
    complete_metadata) ;;
    *)
        echo "ERROR: This release requires complete_metadata; got '${METADATA_TEXT_MODE}'." >&2
        exit 2
        ;;
esac

SPLIT_STRATEGY="${SPLIT_STRATEGY:-temporal_loo}"
case "${DATASET}" in
    beauty)
        DATASET_SLUG="beauty"
        DEFAULT_RAW_DATA_DIR="${AMAZON_2014_DIR:-${GRID_ROOT}/../data/raw/amazon-mcauley-2014}"
        DEFAULT_METADATA_FILE="meta_Beauty_5core.json.gz"
        DEFAULT_INTERACTIONS_FILE="reviews_Beauty_5.json.gz"
        ;;
    toys)
        DATASET_SLUG="toys"
        DEFAULT_RAW_DATA_DIR="${AMAZON_2014_DIR:-${GRID_ROOT}/../data/raw/amazon-mcauley-2014}"
        DEFAULT_METADATA_FILE="meta_Toys_and_Games_5core.json.gz"
        DEFAULT_INTERACTIONS_FILE="reviews_Toys_and_Games_5.json.gz"
        ;;
    sports)
        DATASET_SLUG="sports"
        DEFAULT_RAW_DATA_DIR="${AMAZON_2014_DIR:-${GRID_ROOT}/../data/raw/amazon-mcauley-2014}"
        DEFAULT_METADATA_FILE="meta_Sports_and_Outdoors_5core.json.gz"
        DEFAULT_INTERACTIONS_FILE="reviews_Sports_and_Outdoors_5.json.gz"
        ;;
    pet-supplies|pet_supplies)
        DATASET="pet-supplies"
        DATASET_SLUG="pet_supplies"
        DEFAULT_RAW_DATA_DIR="${AMAZON_2023_DIR:-${GRID_ROOT}/../data/raw/amazon-mcauley-2023}"
        DEFAULT_METADATA_FILE="meta_Pet_Supplies_5core.jsonl.gz"
        DEFAULT_INTERACTIONS_FILE="Pet_Supplies.csv.gz"
        ;;
    *)
        echo "ERROR: Unsupported dataset '${DATASET}'. Choose beauty, toys, sports, or pet-supplies." >&2
        exit 2
        ;;
esac
RAW_DATA_DIR="${RAW_DATA_DIR:-${DEFAULT_RAW_DATA_DIR}}"
METADATA_FILE="${METADATA_FILE:-${DEFAULT_METADATA_FILE}}"
INTERACTIONS_FILE="${INTERACTIONS_FILE:-${DEFAULT_INTERACTIONS_FILE}}"
DATA_ROOT="${GRID_DATA_ROOT:-${GRID_ROOT}/data/amazon_${DATASET_SLUG}_5core_tfrecords}"
OUTPUT_ROOT="${GRID_OUTPUT_ROOT:-${GRID_ROOT}/outputs/${DATASET_SLUG}_5core}"
RUN_NAME="${GRID_RUN_NAME:-${METADATA_TEXT_MODE}}"
if [[ -n "${DATA_DIR_OVERRIDE}" ]]; then
    if [[ "${DATA_DIR_OVERRIDE}" = /* ]]; then
        DATA_DIR="${DATA_DIR_OVERRIDE}"
    else
        DATA_DIR="${GRID_ROOT}/../${DATA_DIR_OVERRIDE}"
    fi
else
    DATA_DIR="${DATA_ROOT}/${SPLIT_STRATEGY}/${RUN_NAME}"
fi
if [[ -n "${EMBEDDING_PATH_OVERRIDE}" && -z "${EMBEDDING_NAME}" ]]; then
    echo "ERROR: --embedding-name is required with --embedding-path." >&2
    exit 2
fi
ARTIFACT_RUN_NAME="${OUTPUT_RUN_NAME:-${RUN_NAME}}"
RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/${SPLIT_STRATEGY}/${ARTIFACT_RUN_NAME}"
COMMON_OUTPUT_DIR="${RUN_OUTPUT_ROOT}/common"
OUTPUT_DIR="${RUN_OUTPUT_ROOT}/${TOKENIZER}"
case "/${DATA_DIR}/${OUTPUT_DIR}/${EMBEDDING_PATH_OVERRIDE}/${TIGER_CKPT_OVERRIDE}/" in
    */description/*)
        echo "ERROR: This release requires complete_metadata artifacts; description-only paths are excluded." >&2
        exit 2
        ;;
esac
LOG_DIR="${GRID_LOG_DIR:-${GRID_ROOT}/logs/${DATASET_SLUG}_5core/${SPLIT_STRATEGY}/${ARTIFACT_RUN_NAME}/${TOKENIZER}}"
# The offline tokenizer learns three semantic levels. TIGER predicts those
# levels plus the deterministic collision-disambiguation suffix emitted by SID
# inference. Keep these knobs separate so setting TIGER to four can never train
# an accidental fourth clustering/codebook layer.
TOKENIZER_HIERARCHIES="${TOKENIZER_HIERARCHIES:-3}"
TIGER_HIERARCHIES="${TIGER_HIERARCHIES:-4}"
TIGER_SEQUENCE_LENGTH="${TIGER_SEQUENCE_LENGTH:-160}"
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"
EMBEDDING_DIM="${EMBEDDING_DIM:-2048}"
GRID_USE_WANDB="${GRID_USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-fairdiverse-grid}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${DATASET_SLUG}-5core-${SPLIT_STRATEGY}-${ARTIFACT_RUN_NAME}-${TOKENIZER}}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-grid-${DATASET_SLUG}-5core-${SPLIT_STRATEGY}-${ARTIFACT_RUN_NAME}-${TOKENIZER}-${SLURM_JOB_ID:-local}}"
GRID_WANDB_LOG_MODEL="${GRID_WANDB_LOG_MODEL:-false}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-${GRID_ROOT}/.cache/huggingface}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${GRID_ROOT}/.cache/wandb}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export TORCH_HOME="${TORCH_HOME:-${GRID_ROOT}/.cache/torch}"

if [[ "${SPLIT_STRATEGY}" == "temporal_loo" ]]; then
    MIN_SEQUENCE_LENGTH=4
else
    MIN_SEQUENCE_LENGTH=2
fi

if [[ ! "${TOKENIZER_HIERARCHIES}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${TIGER_HIERARCHIES}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${TIGER_SEQUENCE_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: hierarchy counts and TIGER_SEQUENCE_LENGTH must be positive integers." >&2
    exit 1
fi
if (( TIGER_HIERARCHIES != TOKENIZER_HIERARCHIES && \
      TIGER_HIERARCHIES != TOKENIZER_HIERARCHIES + 1 )); then
    echo "ERROR: TIGER_HIERARCHIES must equal TOKENIZER_HIERARCHIES or add exactly one suffix token." >&2
    exit 1
fi
if (( TIGER_SEQUENCE_LENGTH % TIGER_HIERARCHIES != 0 )); then
    echo "ERROR: TIGER_SEQUENCE_LENGTH must be divisible by TIGER_HIERARCHIES." >&2
    exit 1
fi
if [[ -n "${TIGER_MAX_STEPS}" && ! "${TIGER_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --tiger-max-steps must be a positive integer, got '${TIGER_MAX_STEPS}'." >&2
    exit 2
fi
if [[ -n "${TIGER_VAL_CHECK_INTERVAL}" && ! "${TIGER_VAL_CHECK_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --tiger-val-check-interval must be a positive integer, got '${TIGER_VAL_CHECK_INTERVAL}'." >&2
    exit 2
fi
if [[ -n "${TIGER_EARLY_STOP_PATIENCE}" && ! "${TIGER_EARLY_STOP_PATIENCE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --tiger-early-stop-patience must be a positive integer, got '${TIGER_EARLY_STOP_PATIENCE}'." >&2
    exit 2
fi
if [[ -n "${TIGER_SELECTION_METRIC}" && "${TIGER_SELECTION_METRIC}" != "val/ndcg@10" && "${TIGER_SELECTION_METRIC}" != "val/recall@5" ]]; then
    echo "ERROR: --tiger-selection-metric must be val/ndcg@10 or val/recall@5." >&2
    exit 2
fi
if [[ -n "${TIGER_RUN_TAG}" && ! "${TIGER_RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "ERROR: --tiger-run-tag must contain only letters, digits, underscores, or dashes and start with a letter or digit." >&2
    exit 2
fi
if [[ -n "${TIGER_EVAL_WORKERS}" && ! "${TIGER_EVAL_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --tiger-eval-workers must be a positive integer." >&2
    exit 2
fi
if [[ -n "${TIGER_EVAL_TIMEOUT}" && ! "${TIGER_EVAL_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --tiger-eval-timeout must be a positive integer." >&2
    exit 2
fi
if [[ -n "${TIGER_STRATEGY}" && "${TIGER_STRATEGY}" != "auto" && "${TIGER_STRATEGY}" != "ddp" ]]; then
    echo "ERROR: --tiger-strategy must be auto or ddp." >&2
    exit 2
fi
if [[ -n "${TIGER_CKPT_OVERRIDE}" ]]; then
    if [[ ! -f "${TIGER_CKPT_OVERRIDE}" ]]; then
        echo "ERROR: TIGER resume checkpoint does not exist: ${TIGER_CKPT_OVERRIDE}" >&2
        exit 2
    fi
    TIGER_CKPT_OVERRIDE="$(readlink -f "${TIGER_CKPT_OVERRIDE}")"
fi

mkdir -p "${DATA_ROOT}" "${COMMON_OUTPUT_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}" \
    "${HF_HOME}" "${WANDB_CACHE_DIR}"
cd "${GRID_ROOT}"

echo "GRID root: ${GRID_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Dataset: ${DATASET}"
echo "Run name: ${RUN_NAME}"
echo "Artifact run name: ${ARTIFACT_RUN_NAME}"
echo "Tokenizer: ${TOKENIZER}"
echo "Embedding source: ${EMBEDDING_NAME:-t5xl}"
echo "Metadata text mode: ${METADATA_TEXT_MODE}"
echo "Tokenizer hierarchies: ${TOKENIZER_HIERARCHIES}"
echo "TIGER hierarchies: ${TIGER_HIERARCHIES}"
echo "TIGER sequence length: ${TIGER_SEQUENCE_LENGTH} tokens ($((TIGER_SEQUENCE_LENGTH / TIGER_HIERARCHIES)) items)"
echo "TIGER resume checkpoint: ${TIGER_CKPT_OVERRIDE:-none}"
echo "TIGER maximum steps: ${TIGER_MAX_STEPS:-experiment default}"
echo "TIGER validation interval (training batches): ${TIGER_VAL_CHECK_INTERVAL:-experiment default}"
echo "TIGER early-stopping patience (checks): ${TIGER_EARLY_STOP_PATIENCE:-experiment default}"
echo "TIGER selection metric: ${TIGER_SELECTION_METRIC:-experiment default}"
echo "TIGER run tag: ${TIGER_RUN_TAG:-none}"
echo "TIGER evaluation workers: ${TIGER_EVAL_WORKERS:-experiment default}"
echo "TIGER evaluation timeout: ${TIGER_EVAL_TIMEOUT:-experiment default}"
echo "TIGER strategy: ${TIGER_STRATEGY:-experiment default}"
echo "Slurm job: ${SLURM_JOB_ID:-direct-shell-run}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "W&B enabled: ${GRID_USE_WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"

case "${GRID_USE_WANDB,,}" in
    true|1|yes)
        GRID_USE_WANDB="true"
        if ! "${PYTHON_BIN}" -c 'import wandb' >/dev/null 2>&1; then
            echo "ERROR: W&B logging is enabled but the wandb package is unavailable." >&2
            echo "Install wandb in ${PYTHON_BIN}'s environment or set GRID_USE_WANDB=false." >&2
            exit 1
        fi
        ;;
    false|0|no)
        GRID_USE_WANDB="false"
        ;;
    *)
        echo "ERROR: GRID_USE_WANDB must be true or false, got '${GRID_USE_WANDB}'." >&2
        exit 2
        ;;
esac

# Populated before each Hydra command. CSV remains enabled; this adds W&B as
# a second Lightning logger with a distinct run for every pipeline stage.
WANDB_OVERRIDES=()
set_wandb_stage() {
    local stage="$1"
    WANDB_OVERRIDES=()
    if [[ "${GRID_USE_WANDB}" == "true" ]]; then
        WANDB_OVERRIDES=(
            "+logger.wandb._target_=lightning.pytorch.loggers.wandb.WandbLogger"
            "+logger.wandb.project=${WANDB_PROJECT}"
            "+logger.wandb.name=${WANDB_RUN_PREFIX}-${stage}"
            "+logger.wandb.group=${WANDB_GROUP}"
            "+logger.wandb.job_type=${stage}"
            "+logger.wandb.save_dir=${OUTPUT_DIR}"
            "+logger.wandb.log_model=${GRID_WANDB_LOG_MODEL}"
        )
        if [[ -n "${WANDB_ENTITY}" ]]; then
            WANDB_OVERRIDES+=("+logger.wandb.entity=${WANDB_ENTITY}")
        fi
    fi
}

latest_checkpoint() {
    local checkpoint_dir="$1"
    # A fresh run has no checkpoint directory yet. Empty output with a zero
    # status lets the caller start training under `set -e` and `pipefail`.
    if [[ ! -d "${checkpoint_dir}" ]]; then
        return 0
    fi
    find "${checkpoint_dir}" -name 'checkpoint_*.ckpt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | sed -n '1s/^[^ ]* //p'
}

archive_stale_inference_files() {
    local pickle_dir="$1"
    local archive_dir="${pickle_dir}/stale_$(date +%Y%m%dT%H%M%S)"
    local stale_files=()
    local candidate
    shopt -s nullglob
    stale_files=("${pickle_dir}"/predictions_*.pkl)
    shopt -u nullglob
    for candidate in \
        "${pickle_dir}/merged_predictions.pkl" \
        "${pickle_dir}/merged_predictions_tensor.pt"; do
        if [[ -e "${candidate}" ]]; then
            stale_files+=("${candidate}")
        fi
    done
    if (( ${#stale_files[@]} > 0 )); then
        mkdir -p "${archive_dir}"
        mv "${stale_files[@]}" "${archive_dir}/"
        echo "Archived stale inference files to ${archive_dir}"
    fi
}

validate_tensor_shape() {
    local tensor_path="$1"
    local expected_rows="$2"
    local expected_columns="$3"
    local label="$4"
    "${PYTHON_BIN}" -c '
import sys
import torch

path, rows, columns, label = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
value = torch.load(path, map_location="cpu", weights_only=False)
expected = (rows, columns)
if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
    shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
    raise ValueError(f"{label} has shape {shape}, expected {expected}: {path}")
if not torch.isfinite(value).all():
    raise ValueError(f"{label} contains NaN or infinite values: {path}")
print(f"Verified {label} shape: {expected}")
' "${tensor_path}" "${expected_rows}" "${expected_columns}" "${label}"
}

if [[ ! -d "${DATA_DIR}/training" ]]; then
    "${PYTHON_BIN}" preprocess_amazon_beauty.py \
        --input-dir "${RAW_DATA_DIR}" \
        --output-dir "${DATA_DIR}" \
        --metadata-file "${METADATA_FILE}" \
        --metadata-text-mode "${METADATA_TEXT_MODE}" \
        --interactions-file "${INTERACTIONS_FILE}" \
        --min-seq-length "${MIN_SEQUENCE_LENGTH}" \
        --split-strategy "${SPLIT_STRATEGY}" \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --seed 42 2>&1 | tee "${LOG_DIR}/step0_preprocess.log"
else
    RECORDED_METADATA_TEXT_MODE="$("${PYTHON_BIN}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("metadata_text_mode", "")
print(value)
' "${DATA_DIR}/split_config.json" 2>/dev/null || true)"
    if [[ -z "${RECORDED_METADATA_TEXT_MODE}" ]]; then
        echo "ERROR: Existing data does not record metadata_text_mode: ${DATA_DIR}/split_config.json" >&2
        echo "Refusing to guess from the directory name. Regenerate it or select a data directory with explicit provenance." >&2
        exit 1
    fi
    if [[ "${RECORDED_METADATA_TEXT_MODE}" != "${METADATA_TEXT_MODE}" ]]; then
        echo "ERROR: Existing data uses metadata_text_mode=${RECORDED_METADATA_TEXT_MODE}, requested ${METADATA_TEXT_MODE}: ${DATA_DIR}" >&2
        exit 1
    fi
    "${PYTHON_BIN}" verify_split_strategy.py "${DATA_DIR}"
fi

for required_dir in training validation testing items; do
    if ! find "${DATA_DIR}/${required_dir}" -name '*.tfrecord.gz' -print -quit | grep -q .; then
        echo "ERROR: Missing TFRecords in ${DATA_DIR}/${required_dir}" >&2
        exit 1
    fi
done
for required_file in item_id_mapping.json split_config.json validation_items.json test_items.json; do
    if [[ ! -s "${DATA_DIR}/${required_file}" ]]; then
        echo "ERROR: Missing preprocessing output ${DATA_DIR}/${required_file}" >&2
        exit 1
    fi
done

# Verify the bytes consumed by the embedding model, not merely the requested
# option or directory name. No downstream artifact may be created otherwise.
"${PYTHON_BIN}" verify_metadata_text.py "${DATA_DIR}" "${METADATA_TEXT_MODE}"

if [[ "${PREPROCESS_ONLY}" == "true" ]]; then
    printf 'GRID preprocessing complete.\nDataset: %s\nData: %s\n' "${DATASET}" "${DATA_DIR}"
    exit 0
fi

EMBED_DIR="${COMMON_OUTPUT_DIR}/step1_embeddings"
mkdir -p "${EMBED_DIR}"
ITEM_COUNT="$(${PYTHON_BIN} -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "${DATA_DIR}/item_id_mapping.json")"
if [[ -n "${EMBEDDING_PATH_OVERRIDE}" ]]; then
    if [[ "${EMBEDDING_PATH_OVERRIDE}" = /* ]]; then
        EMBEDDING_PATH="${EMBEDDING_PATH_OVERRIDE}"
    else
        EMBEDDING_PATH="${GRID_ROOT}/../${EMBEDDING_PATH_OVERRIDE}"
    fi
    if [[ ! -f "${EMBEDDING_PATH}" ]]; then
        echo "ERROR: Custom embedding file does not exist: ${EMBEDDING_PATH}" >&2
        exit 1
    fi
    EMBEDDING_DIM="$(${PYTHON_BIN} -c 'import sys, torch; x=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(x.shape[-1])' "${EMBEDDING_PATH}")"
    EMBEDDING_COUNT="$(${PYTHON_BIN} -c 'import sys, torch; x=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(x.shape[0])' "${EMBEDDING_PATH}")"
    if [[ "${EMBEDDING_COUNT}" -ne "${ITEM_COUNT}" ]]; then
        echo "ERROR: ${EMBEDDING_NAME} has ${EMBEDDING_COUNT} rows but the GRID item mapping has ${ITEM_COUNT} items." >&2
        exit 1
    fi
    echo "Using custom ${EMBEDDING_NAME} embeddings: ${EMBEDDING_PATH} (${EMBEDDING_COUNT} x ${EMBEDDING_DIM})"
else
    EMBEDDING_PATH="$(find "${EMBED_DIR}" -name merged_predictions_tensor.pt -print -quit)"
fi
if [[ -z "${EMBEDDING_PATH}" ]]; then
    mkdir -p "${EMBED_DIR}/pickle"
    archive_stale_inference_files "${EMBED_DIR}/pickle"
    set_wandb_stage "embedding-inference"
    "${PYTHON_BIN}" -m src.inference \
        experiment=sem_embeds_inference_flat \
        data_dir="${DATA_DIR}" \
        hydra.run.dir="${EMBED_DIR}" \
        hydra.job.chdir=False \
        "${WANDB_OVERRIDES[@]}" 2>&1 | tee "${LOG_DIR}/step1_embeddings.log"
    EMBEDDING_PATH="$(find "${EMBED_DIR}" -name merged_predictions_tensor.pt -print -quit)"
else
    if [[ -z "${EMBEDDING_PATH_OVERRIDE}" ]]; then
        echo "Reusing shared item embeddings: ${EMBEDDING_PATH}"
    fi
fi
test -n "${EMBEDDING_PATH}"
validate_tensor_shape "${EMBEDDING_PATH}" "${ITEM_COUNT}" "${EMBEDDING_DIM}" "item embeddings"

SID_TRAIN_DIR="${OUTPUT_DIR}/step2_tokenizer_training"
mkdir -p "${SID_TRAIN_DIR}/metadata"
SID_CKPT="$(latest_checkpoint "${SID_TRAIN_DIR}/checkpoints")"
if [[ -z "${SID_CKPT}" ]]; then
    RESTART_METADATA="${SID_TRAIN_DIR}/metadata/restart_metadata.json"
    if [[ -f "${RESTART_METADATA}" ]]; then
        FAILED_METADATA="${RESTART_METADATA%.json}.failed.$(date +%Y%m%dT%H%M%S).json"
        mv "${RESTART_METADATA}" "${FAILED_METADATA}"
        echo "Archived stale retry metadata: ${FAILED_METADATA}"
    fi
    echo "Starting ${TOKENIZER} tokenizer training with ${TOKENIZER_TRAIN_EXPERIMENT}"
    set_wandb_stage "tokenizer-training"
    "${PYTHON_BIN}" -m src.train \
        experiment="${TOKENIZER_TRAIN_EXPERIMENT}" \
        data_dir="${DATA_DIR}" \
        embedding_path="${EMBEDDING_PATH}" \
        embedding_dim="${EMBEDDING_DIM}" \
        num_hierarchies="${TOKENIZER_HIERARCHIES}" \
        codebook_width="${CODEBOOK_WIDTH}" \
        hydra.run.dir="${SID_TRAIN_DIR}" \
        hydra.job.chdir=False \
        trainer.devices=1 \
        "${WANDB_OVERRIDES[@]}" 2>&1 | tee "${LOG_DIR}/step2_semantic_ids.log"
    SID_CKPT="$(latest_checkpoint "${SID_TRAIN_DIR}/checkpoints")"
fi
test -n "${SID_CKPT}"

SID_INFER_DIR="${OUTPUT_DIR}/step2b_sid_inference"
mkdir -p "${SID_INFER_DIR}/metadata" "${SID_INFER_DIR}/pickle"
CLEAN_SIDS="${SID_INFER_DIR}/pickle/merged_predictions_tensor_suffix${TIGER_HIERARCHIES}.pt"
RAW_SIDS="$(find "${SID_INFER_DIR}" -name merged_predictions_tensor.pt -print -quit)"
if [[ -z "${RAW_SIDS}" ]]; then
    archive_stale_inference_files "${SID_INFER_DIR}/pickle"
    set_wandb_stage "sid-inference"
    "${PYTHON_BIN}" -m src.inference \
        experiment="${TOKENIZER_INFERENCE_EXPERIMENT}" \
        data_dir="${DATA_DIR}" \
        embedding_path="${EMBEDDING_PATH}" \
        embedding_dim="${EMBEDDING_DIM}" \
        num_hierarchies="${TOKENIZER_HIERARCHIES}" \
        codebook_width="${CODEBOOK_WIDTH}" \
        "ckpt_path='${SID_CKPT}'" \
        hydra.run.dir="${SID_INFER_DIR}" \
        hydra.job.chdir=False \
        trainer.devices=1 \
        "${WANDB_OVERRIDES[@]}" 2>&1 | tee "${LOG_DIR}/step2b_sid_inference.log"
    RAW_SIDS="$(find "${SID_INFER_DIR}" -name merged_predictions_tensor.pt -print -quit)"
    test -n "${RAW_SIDS}"
fi
validate_tensor_shape "${RAW_SIDS}" "${TIGER_HIERARCHIES}" "${ITEM_COUNT}" "raw semantic IDs"
if [[ ! -f "${CLEAN_SIDS}" ]]; then
    "${PYTHON_BIN}" clean_semantic_ids.py "${RAW_SIDS}" "${CLEAN_SIDS}" \
        --tokenizer-hierarchies "${TOKENIZER_HIERARCHIES}" \
        --tiger-hierarchies "${TIGER_HIERARCHIES}" \
        --codebook-size "${CODEBOOK_WIDTH}"
fi
validate_tensor_shape "${CLEAN_SIDS}" "${TIGER_HIERARCHIES}" "${ITEM_COUNT}" "clean semantic IDs"

# The standalone config names its validation input "evaluation".
if [[ ! -e "${DATA_DIR}/evaluation" ]]; then
    ln -s "${DATA_DIR}/validation" "${DATA_DIR}/evaluation"
fi

TIGER_STAGE_SUFFIX=""
if [[ -n "${TIGER_RUN_TAG}" ]]; then
    TIGER_STAGE_SUFFIX="_${TIGER_RUN_TAG}"
fi
TIGER_DIR="${OUTPUT_DIR}/step3_training_suffix${TIGER_HIERARCHIES}${TIGER_STAGE_SUFFIX}"
mkdir -p "${TIGER_DIR}/metadata"
set_wandb_stage "tiger-training"
TIGER_TRAIN_OVERRIDES=()
if [[ -n "${TIGER_CKPT_OVERRIDE}" ]]; then
    # Keep the quotes in the argv value so Hydra treats characters such as
    # the '=' signs in Lightning checkpoint filenames as literal path text.
    TIGER_TRAIN_OVERRIDES+=("ckpt_path='${TIGER_CKPT_OVERRIDE}'")
fi
if [[ -n "${TIGER_MAX_STEPS}" ]]; then
    TIGER_TRAIN_OVERRIDES+=("trainer.max_steps=${TIGER_MAX_STEPS}")
fi
if [[ -n "${TIGER_VAL_CHECK_INTERVAL}" ]]; then
    TIGER_TRAIN_OVERRIDES+=("trainer.val_check_interval=${TIGER_VAL_CHECK_INTERVAL}")
fi
if [[ -n "${TIGER_EARLY_STOP_PATIENCE}" ]]; then
    TIGER_TRAIN_OVERRIDES+=("callbacks.early_stopping.patience=${TIGER_EARLY_STOP_PATIENCE}")
fi
if [[ -n "${TIGER_SELECTION_METRIC}" ]]; then
    TIGER_TRAIN_OVERRIDES+=("callbacks.model_checkpoint.monitor=${TIGER_SELECTION_METRIC}")
    TIGER_TRAIN_OVERRIDES+=("callbacks.early_stopping.monitor=${TIGER_SELECTION_METRIC}")
fi
if [[ -n "${TIGER_EVAL_WORKERS}" ]]; then
    TIGER_TRAIN_OVERRIDES+=("data_loading.val_dataloader_config.dataloader.num_workers=${TIGER_EVAL_WORKERS}")
    TIGER_TRAIN_OVERRIDES+=("data_loading.test_dataloader_config.dataloader.num_workers=${TIGER_EVAL_WORKERS}")
fi
if [[ -n "${TIGER_EVAL_TIMEOUT}" ]]; then
    TIGER_TRAIN_OVERRIDES+=("data_loading.val_dataloader_config.dataloader.timeout=${TIGER_EVAL_TIMEOUT}")
    TIGER_TRAIN_OVERRIDES+=("data_loading.test_dataloader_config.dataloader.timeout=${TIGER_EVAL_TIMEOUT}")
fi
if [[ -n "${TIGER_STRATEGY}" ]]; then
    TIGER_TRAIN_OVERRIDES+=("trainer.strategy=${TIGER_STRATEGY}")
fi
"${PYTHON_BIN}" -m src.train \
    experiment=tiger_train_flat \
    data_dir="${DATA_DIR}" \
    semantic_id_path="${CLEAN_SIDS}" \
    num_hierarchies="${TIGER_HIERARCHIES}" \
    sequence_length="${TIGER_SEQUENCE_LENGTH}" \
    hydra.run.dir="${TIGER_DIR}" \
    hydra.job.chdir=False \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    "${TIGER_TRAIN_OVERRIDES[@]}" \
    "${WANDB_OVERRIDES[@]}" 2>&1 | tee "${LOG_DIR}/step3_training${TIGER_STAGE_SUFFIX}.log"

TIGER_CKPT="$(latest_checkpoint "${TIGER_DIR}/checkpoints")"
test -n "${TIGER_CKPT}"
INFER_DIR="${OUTPUT_DIR}/step4_inference_suffix${TIGER_HIERARCHIES}${TIGER_STAGE_SUFFIX}"
mkdir -p "${INFER_DIR}/metadata"
set_wandb_stage "tiger-inference"
"${PYTHON_BIN}" -m src.inference \
    experiment=tiger_inference_flat \
    data_dir="${DATA_DIR}" \
    semantic_id_path="${CLEAN_SIDS}" \
    num_hierarchies="${TIGER_HIERARCHIES}" \
    sequence_length="${TIGER_SEQUENCE_LENGTH}" \
    "ckpt_path='${TIGER_CKPT}'" \
    hydra.run.dir="${INFER_DIR}" \
    hydra.job.chdir=False \
    trainer.devices=1 \
    "${WANDB_OVERRIDES[@]}" 2>&1 | tee "${LOG_DIR}/step4_inference${TIGER_STAGE_SUFFIX}.log"

printf 'GRID pipeline complete.\nData: %s\nSIDs: %s\nCheckpoint: %s\n' \
    "${DATA_DIR}" "${CLEAN_SIDS}" "${TIGER_CKPT}"
