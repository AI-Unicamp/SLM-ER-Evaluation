#!/bin/bash
#SBATCH --job-name=emotion2vec_esd_train # Job name for Slurm
#SBATCH --output=/home/lucas.ueda/slurm/emotion2vec_esd_train_%j.out # Standard output file
#SBATCH --error=/home/lucas.ueda/slurm/emotion2vec_esd_train_%j.err   # Standard error file
#SBATCH --ntasks=1                   # Run on a single CPU core/task
#SBATCH --time=2-00:00:00            # Maximum 2 days (adjust as needed for full training)
#SBATCH --mem=128G                   # Request 128GB of memory (adjust based on dataset size and GPU memory)
#SBATCH --partition=l40s             # Specify your Slurm partition (e.g., 'l40s', 'h100')
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mail-user=l156368@dac.unicamp.br # Your email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications for job events

# Load Miniconda and activate your environment
source ~/miniconda3/bin/activate
conda activate emotion2vec  

# --- Define Project and Data Paths ---
# <--- IMPORTANT: Adjust these paths to match your actual directory structure --->
PROJECT_ROOT="/home/lucas.ueda/github/SLM-ER-Evaluation/ser"
ESD_FILES_ROOT="/home/lucas.ueda/expressive_datasets/esd/files" # Base directory for raw WAV files
EMOTION2VEC_MODEL_DIR="${PROJECT_ROOT}/upstream" # Directory containing Fairseq model definitions
EMOTION2VEC_CHECKPOINT="${PROJECT_ROOT}/pretrained_model/emotion2vec_base/emotion2vec_base.pt" # Path to your pre-trained model checkpoint
FEATURES_DIR="${PROJECT_ROOT}/scripts/esd_features" # Directory for extracted .npy, .lengths, .emo files
MANIFESTS_DIR="${PROJECT_ROOT}/scripts/esd_manifests" # Directory for generated .tsv files

# Add the project root to PYTHONPATH so custom Fairseq modules (models, tasks) can be found
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "--- Training the downstream model on ESD dataset ---"
PYTHON_TRAIN_SCRIPT="${PROJECT_ROOT}/bin/train.py"

python "$PYTHON_TRAIN_SCRIPT" \
    dataset.feat_path="${FEATURES_DIR}" \
    dataset.batch_size=128 \
    optimization.epoch=100 \
    optimization.lr=5e-4

TRAINING_EXIT_CODE=$?
echo "Training job exited with code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Full emotion2vec ESD training pipeline completed successfully!"
    echo "Trained model checkpoint (best_model.pth) will be in ${PROJECT_ROOT}/esd_downstream_model"
else
    echo "Full emotion2vec ESD training pipeline failed with exit code $TRAINING_EXIT_CODE"
    exit $TRAINING_EXIT_CODE
fi