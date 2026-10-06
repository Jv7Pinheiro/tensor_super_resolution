#!/bin/bash -l

#SBATCH --job-name=TFIM_8x8-1x1_numeric
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=data/outputs/TFIM_8x8-1x1_numeric.txt
#SBATCH --error=data/outputs/TFIM_8x8-1x1_numeric.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 64 \
    --H TFIM_8x8 \
    --method numeric \
    --L 1 \
    --R 1 \
    > "data/outputs/TFIM_8x8-1x1_numeric.txt" 2>&1
