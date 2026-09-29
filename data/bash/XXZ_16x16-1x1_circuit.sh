#!/bin/bash -l

#SBATCH --job-name=tensor-sr
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=data/outputs/XXZ_16x16-1x1_circuit.txt
#SBATCH --error=data/outputs/XXZ_16x16-1x1_circuit.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 64 \
    --H XXZ_16x16 \
    --method circuit \
    --L 1 \
    --R 1 \
    > "data/outputs/XXZ_16x16-1x1_circuit.txt" 2>&1
