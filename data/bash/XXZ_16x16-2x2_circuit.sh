#!/bin/bash -l

#SBATCH --job-name=XXZ_16x16-2x2_circuit
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=data/outputs/XXZ_16x16-2x2_circuit.txt
#SBATCH --error=data/outputs/XXZ_16x16-2x2_circuit.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \
    --workers 64 \
    --H "XXZ_(h=0)-(d=-1)_16x16" \
    --method circuit \
    --L 2 \
    --R 2 \
    --algorithms "QFAMES"

python3 main.py \
    --workers 64 \
    --H "XXZ_(h=1)-(d=-1)_16x16" \
    --method circuit \
    --L 2 \
    --R 2 \
    --algorithms "QFAMES"

python3 main.py \
    --workers 64 \
    --H "XXZ_(h=0)-(d=0)_16x16" \
    --method circuit \
    --L 2 \
    --R 2 \
    --algorithms "QFAMES"

python3 main.py \
    --workers 64 \
    --H "XXZ_(h=0)-(d=1)_16x16" \
    --method circuit \
    --L 2 \
    --R 2 \
    --algorithms "QFAMES"

python3 main.py \
    --workers 64 \
    --H "XXZ_(h=0)-(d=2)_16x16" \
    --method circuit \
    --L 2 \
    --R 2 \
    --algorithms "QFAMES"
