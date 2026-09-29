from pathlib import Path


###################
## Configuration ##
###################

ACCOUNT = "csit"
PARTITION = "cpu"

OUTPUT_DIR = Path("data/bash")

HAMILTONIANS = ["XXZ", "TFIM"]
METHODS = ["numeric", "circuit"]

MIN_SIZE = 8
MAX_SIZE = 16

LR_VALUES = [1, 2, 4, 8, 16]


#####################
## Resource policy ##
#####################

def get_resources(size, method):
    """
    Return (cores, qos, walltime) for a given experiment.

    walltime is in Slurm's HH:MM:SS format.
    """

    if method == "numeric":
        if size == 8:
            return 32, "standby", "03:00:00"

        elif size == 16:
            # Estimated runtime is around 4 hours, so don't use
            # standby because its maximum walltime is 4 hours.
            return 64, "normal", "08:00:00"

    elif method == "circuit":
        if size == 8:
            return 64, "normal", "08:00:00"

        elif size == 16:
            return 64, "normal", "24:00:00"

    raise ValueError(
        f"No resource policy defined for size={size}, method={method}"
    )


#############################
## SLURM script generation ##
#############################

def make_slurm_script(job_name, ham, size, lr, method, cores, qos, walltime):
    output_file = f"{job_name}.txt"

    return f"""#!/bin/bash -l

#SBATCH --job-name=tensor-sr
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --time={walltime}
#SBATCH --output=data/outputs/{output_file}
#SBATCH --error=data/outputs/{output_file}
#SBATCH --mail-user=deolivj@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tensor-super-resolution

cd "$HOME/tensor_super_resolution" || exit 1

python3 main.py \\
    --workers {cores} \\
    --H {ham}_{size}x{size} \\
    --method {method} \\
    --L {lr} \\
    --R {lr} \\
    > "data/outputs/{output_file}" 2>&1
"""


##########
## Main ##
##########

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0

    for ham in HAMILTONIANS:
        for method in METHODS:
            cores, qos, walltime = get_resources(MAX_SIZE if False else MIN_SIZE, method)

            for size in range(MIN_SIZE, MAX_SIZE + 1):
                # Only use powers of two.
                if size & (size - 1):
                    continue

                # LR values up to the Hamiltonian size.
                for lr in LR_VALUES:
                    if lr > size:
                        continue

                    job_name = f"{ham}_{size}x{size}-{lr}x{lr}_{method}"

                    cores, qos, walltime = get_resources(size, method)

                    script = make_slurm_script(
                        job_name=job_name,
                        ham=ham,
                        size=size,
                        lr=lr,
                        method=method,
                        cores=cores,
                        qos=qos,
                        walltime=walltime,
                    )

                    output_path = OUTPUT_DIR / f"{job_name}.sh"

                    output_path.write_text(script)

                    generated += 1

                    print(f"\tCreated {output_path} [{cores} cores, {qos}, {walltime}]")

    print(f"Generated {generated} SLURM scripts.")


if __name__ == "__main__":
    main()