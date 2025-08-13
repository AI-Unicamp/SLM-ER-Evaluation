#!/bin/bash
#SBATCH --job-name=emotion2vec_feat_extract # Job name for Slurm
#SBATCH --output=/home/lucas.ueda/slurm/emotion2vec_feat_extract_%j.out # Standard output file
#SBATCH --error=/home/lucas.ueda/slurm/emotion2vec_feat_extract_%j.err   # Standard error file
#SBATCH --ntasks=1                   # Run on a single CPU core/task
#SBATCH --time=0-23:00:00            # Maximum 8 hours for feature extraction (adjust as needed)
#SBATCH --mem=64G                    # Request 64GB of memory (adjust based on dataset size)
#SBATCH --partition=l40s             # Specify your Slurm partition (e.g., 'l40s', 'h100')
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mail-user=l156368@dac.unicamp.br # Your email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications for job events

# Load Miniconda and activate your environment
source ~/miniconda3/bin/activate
conda activate emotion2vec 

# --- Define Project and Data Paths ---
HOME_DIR=$(echo ~)

# Model directory from training
PROJECT_ROOT="${HOME_DIR}/github/SLM-ER-Evaluation/ser"

EMOTION2VEC_MODEL_DIR="${PROJECT_ROOT}/upstream"
EMOTION2VEC_CHECKPOINT="${PROJECT_ROOT}/pretrained_model/emotion2vec_base/emotion2vec_base.pt"
FEATURES_DIR="${PROJECT_ROOT}/scripts/esd_features"     # Directory where extracted .npy and .lengths files will be saved
MANIFESTS_DIR="${PROJECT_ROOT}/scripts/esd_manifests"    # Directory where your .tsv, .lengths, .emo manifests are located (from generate_esd_manifests.py)

# Add the project root to PYTHONPATH so custom Fairseq modules (models, tasks) can be found
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Create the features save directory if it doesn't exist
mkdir -p "$FEATURES_DIR"
echo "Created features save directory: $FEATURES_DIR"

echo "--- Starting emotion2vec feature extraction job ---"

# List of data splits to process
SPLITS=("train" "val" "test")

# Path to the feature extraction Python script
PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/extract_features_from_tsv.py"

# Loop through each split and run the feature extraction script
for SPLIT in "${SPLITS[@]}"; do
    echo "Extracting features for split: ${SPLIT}"
    python "$PYTHON_SCRIPT" \
        --data "$MANIFESTS_DIR" \
        --model "$EMOTION2VEC_MODEL_DIR" \
        --split "$SPLIT" \
        --checkpoint "$EMOTION2VEC_CHECKPOINT" \
        --save-dir "$FEATURES_DIR" \
        --layer 11 # Always extract from layer 11 for emotion2vec base features
done

# Check the exit code of the last command (feature extraction loop)
FEAT_EXTRACT_EXIT_CODE=$?
echo "Feature extraction job exited with code: $FEAT_EXTRACT_EXIT_CODE"

if [ $FEAT_EXTRACT_EXIT_CODE -eq 0 ]; then
    echo "Emotion2vec feature extraction completed successfully!"
    echo "Features saved at: $FEATURES_DIR"
else
    echo "Emotion2vec feature extraction failed with exit code $FEAT_EXTRACT_EXIT_CODE"
    exit $FEAT_EXTRACT_EXIT_CODE # Exit with error code if something went wrong
fi