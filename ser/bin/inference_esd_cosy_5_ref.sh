#!/bin/bash
#SBATCH --job-name=emotion2vec_esd_infer # Job name for Slurm
#SBATCH --output=/home/lucas.ueda/slurm/emotion2vec_esd_infer_%j.out # Standard output file
#SBATCH --error=/home/lucas.ueda/slurm/emotion2vec_esd_infer_%j.err   # Standard error file
#SBATCH --ntasks=1                   # Run on a single CPU core/task
#SBATCH --time=0-04:00:00            # Maximum 4 hours for inference (adjust as needed)
#SBATCH --mem=32G                    # Request 32GB of memory (adjust based on dataset size)
#SBATCH --partition=l40s             # Specify your Slurm partition (e.g., 'l40s', 'h100')
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mail-user=l156368@dac.unicamp.br # Your email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications for job events

# Load Miniconda and activate your environment
source ~/miniconda3/bin/activate
conda activate emotion2vec  # <--- IMPORTANT: Replace 'emotion2vec' with your actual Conda environment name

# --- Define Project and Model Paths ---
# <--- IMPORTANT: Adjust these paths to match your actual directory structure --->
PROJECT_ROOT="/home/lucas.ueda/github/SLM-ER-Evaluation/ser"
INFERENCE_SCRIPT="${PROJECT_ROOT}/bin/inference.py"
EMOTION2VEC_CHECKPOINT="${PROJECT_ROOT}/pretrained_model/emotion2vec_base/emotion2vec_base.pt"
DOWNSTREAM_CHECKPOINT="${PROJECT_ROOT}/bin/outputs/2025-07-14/22-49-47/esd_downstream_model/best_model.pth" # Path to your trained downstream model
UPSTREAM_MODEL_DIR="${PROJECT_ROOT}/upstream" # Path to the 'upstream' directory for Fairseq models

# Add the project root to PYTHONPATH so custom Fairseq modules can be found
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "--- Starting emotion2vec ESD Inference Job ---"


# Option 2: Infer on a directory of WAV files and save results to CSV
INPUT_WAV_DIR="/hadatasets/pedro.correa/samples_cosyvoice/5_ref/audio" # Example directory with WAVs
OUTPUT_CSV_PATH="${PROJECT_ROOT}/inferences/cosy_5_ref.csv" # Path to save the CSV results

# INPUT_WAV_DIR="/hadatasets/pedro.correa/samples_styletts2/5_ref/audio"
# OUTPUT_CSV_PATH="${PROJECT_ROOT}/inferences/styletts2_5_ref.csv" 

python "$INFERENCE_SCRIPT" \
    --input_path "$INPUT_WAV_DIR" \
    --emotion2vec_checkpoint "$EMOTION2VEC_CHECKPOINT" \
    --downstream_checkpoint "$DOWNSTREAM_CHECKPOINT" \
    --model_dir "$UPSTREAM_MODEL_DIR" \
    --output_csv "$OUTPUT_CSV_PATH"


# Check the exit code of the inference script
INFERENCE_EXIT_CODE=$?
echo "Inference job exited with code: $INFERENCE_EXIT_CODE"

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "Emotion2vec ESD inference completed successfully!"
    if [ -n "$OUTPUT_CSV_PATH" ]; then # Check if OUTPUT_CSV_PATH was set (meaning directory processing)
        echo "Inference results saved to: $OUTPUT_CSV_PATH"
    fi
else
    echo "Emotion2vec ESD inference failed with exit code $INFERENCE_EXIT_CODE"
    exit $INFERENCE_EXIT_CODE # Exit with error code if something went wrong
fi