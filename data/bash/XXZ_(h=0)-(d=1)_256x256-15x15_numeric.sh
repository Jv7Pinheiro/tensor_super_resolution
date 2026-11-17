#!/bin/bash -l

#SBATCH --job-name=XXZ_(h=0)-(d=1)_256x256-15x15_numeric
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=data/outputs/XXZ_(h=0)-(d=1)_256x256-15x15_numeric.txt
#SBATCH --error=data/outputs/XXZ_(h=0)-(d=1)_256x256-15x15_numeric.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 python/QFAMES_SV.py \
    --workers 64 \
    --H "XXZ_(h=0)-(d=1)_256x256" \
    --method numeric \
    --L 15 \
    --R 15 \
    --perturbations "None,Small,Big,Rand"
