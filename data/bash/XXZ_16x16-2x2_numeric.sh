#!/bin/bash -l

#SBATCH --job-name=XXZ_16x16-2x2_numeric
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --output=data/outputs/XXZ_16x16-2x2_numeric.txt
#SBATCH --error=data/outputs/XXZ_16x16-2x2_numeric.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 128 \
    --H XXZ_16x16 \
    --method numeric \
    --L 2 \
    --R 2 \
    > "data/outputs/XXZ_16x16-2x2_numeric.txt" 2>&1
