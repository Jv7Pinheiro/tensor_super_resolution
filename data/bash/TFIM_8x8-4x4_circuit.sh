#!/bin/bash -l

#SBATCH --job-name=tensor-sr
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=data/outputs/TFIM_8x8-4x4_circuit.txt
#SBATCH --error=data/outputs/TFIM_8x8-4x4_circuit.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 64 \
    --H TFIM_8x8 \
    --method circuit \
    --L 4 \
    --R 4 \
    > "data/outputs/TFIM_8x8-4x4_circuit.txt" 2>&1
