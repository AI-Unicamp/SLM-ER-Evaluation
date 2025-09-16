#!/bin/bash
#SBATCH --job-name=styletts
#SBATCH --output=/home/joao.lima/slurm/styletts.out
#SBATCH --error=/home/joao.lima/slurm/styletts.err
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=70G

BIND_PATHS=$(paste -sd, <<EOF
/dev/shm:/dev/shm
$HOME:/home/$USER
EOF
)

apptainer exec --fakeroot \
    --no-home \
    --cleanenv \
    --writable \
    --bind "$BIND_PATHS" \
    --nv \
    ~/images/espnet_sandbox \
    bash -c "
    export HOME=/home/$USER ; \
    source /opt/miniconda/bin/activate ; \
    conda activate styletts2 && \
    python3 batch_inference.py
"