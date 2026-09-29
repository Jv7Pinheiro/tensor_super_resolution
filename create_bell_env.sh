#!/usr/bin/env bash
set -euo pipefail

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list 2>/dev/null | awk '{print $1}' | grep -qx 'tensor-super-resolution'; then
  echo "Conda environment 'tensor-super-resolution' already exists."
else
  echo "Creating conda environment 'tensor-super-resolution' from project config..."
  conda env create -f environment.yml
fi

echo "Environment ready:"
echo "  module load conda/2025.09"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate tensor-super-resolution"
