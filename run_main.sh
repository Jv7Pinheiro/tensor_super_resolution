#!/bin/bash -l
#SBATCH --job-name=tensor-sr
#SBATCH --account=csit
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=output.txt
#SBATCH --error=output.txt
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

status=0
python3 main.py --workers 64 >> output.txt 2>&1 || status=$?

# Auto-save repo state only after a successful test run.
if [ "$status" -eq 0 ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git config user.name "Jv7Pinheiro"
    git config user.email "deolivj@purdue.edu"

    git add -A

    if ! git diff --cached --quiet; then
        git commit -m "Auto-commit from cluster run $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        git push origin HEAD || git push
    else
        echo "No repo changes to commit."
    fi
elif [ "$status" -ne 0 ]; then
    echo "Python job failed (exit code $status); skipping auto-commit/push."
else
    echo "Not inside a git repository; skipping auto-commit/push."
fi

exit "$status"