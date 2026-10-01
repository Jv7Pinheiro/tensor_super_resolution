#!/bin/bash -l

#SBATCH --job-name=TFIM_16x16-16x16_circuit
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --output=data/outputs/TFIM_16x16-16x16_circuit.txt
#SBATCH --error=data/outputs/TFIM_16x16-16x16_circuit.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 128 \
    --H TFIM_16x16 \
    --method circuit \
    --L 16 \
    --R 16 \
    > "data/outputs/TFIM_16x16-16x16_circuit.txt" 2>&1
