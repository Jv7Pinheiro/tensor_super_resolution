#!/bin/bash -l

#SBATCH --job-name=XXZ_8x8-4x4_numeric
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --time=08:00:00
#SBATCH --output=data/outputs/XXZ_8x8-4x4_numeric.txt
#SBATCH --error=data/outputs/XXZ_8x8-4x4_numeric.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 26 \
    --H XXZ_8x8 \
    --method numeric \
    --L 4 \
    --R 4 \
    > "data/outputs/XXZ_8x8-4x4_numeric.txt" 2>&1
