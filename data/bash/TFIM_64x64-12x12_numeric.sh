#!/bin/bash -l

#SBATCH --job-name=TFIM_64x64-12x12_numeric
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=data/outputs/TFIM_64x64-12x12_numeric.txt
#SBATCH --error=data/outputs/TFIM_64x64-12x12_numeric.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 QFAMES_SV.py \
    --workers 64 \
    --H "TFIM_(g=0.5)_64x64" \
    --method numeric \
    --L 12 \
    --R 12 \
    --perturbations "None,Small,Big,Rand"
