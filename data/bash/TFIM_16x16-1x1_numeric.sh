#!/bin/bash -l

#SBATCH --job-name=TFIM_16x16-1x1_numeric
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --output=data/outputs/TFIM_16x16-1x1_numeric.txt
#SBATCH --error=data/outputs/TFIM_16x16-1x1_numeric.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 128 \
    --H TFIM_16x16 \
    --method numeric \
    --L 1 \
    --R 1 \
    > "data/outputs/TFIM_16x16-1x1_numeric.txt" 2>&1
