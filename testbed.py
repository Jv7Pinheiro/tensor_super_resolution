from pathlib import Path


###################
## Configuration ##
###################

ACCOUNT = "csit"
PARTITION = "cpu"

OUTPUT_DIR = Path("data/bash")

HAMILTONIANS = ["XXZ", "TFIM"]
METHODS = ["numeric", "circuit"]

TFIM_params = [0.5, 1, 1.5]
XXZ_params = [[-1, 0], [-1, 1], [0, 0], [1, 0], [2, 0]]

MIN_SIZE = 8
MAX_SIZE = 16

LR_VALUES = [1, 2, 4, 8, 16]


#####################
## Resource policy ##
#####################

def get_resources(lr, size, method):
    """
    Return (cores, qos, walltime) for a given experiment.

    walltime is in Slurm's HH:MM:SS format.
    """

    if size == 8:
        if lr == 1:
            return 64, "standby", "01:00:00"
        elif lr == 2:
            return 64, "standby", "03:00:00"
        elif lr == 4:
            return 64, "normal", "24:00:00"
        elif lr == 8:
            return 64, "normal", "48:00:00"
    if size == 16:
        if lr == 1:
            return 64, "standby", "01:00:00"
        elif lr == 2:
            return 64, "standby", "03:00:00"
        elif lr == 4:
            return 64, "normal", "24:00:00"
        elif lr == 8:
            return 64, "normal", "48:00:00"
        elif lr == 16:
            return 64, "normal", "48:00:00"

    raise ValueError(
        f"No resource policy defined for size={size}, method={method}, L=R={lr}"
    )


#############################
## Hamiltonian configurations
#############################

def get_hamiltonian_names(ham, size):
    """
    Return all parameterized Hamiltonian names for a given size.

    TFIM_params contains g values.
    XXZ_params contains [delta, h] pairs.
    """

    if ham == "TFIM":
        return [
            f"TFIM_(g={g})_{size}x{size}"
            for g in TFIM_params
        ]

    elif ham == "XXZ":
        return [
            f"XXZ_(h={h})-(d={delta})_{size}x{size}"
            for delta, h in XXZ_params
        ]

    raise ValueError(f"Unknown Hamiltonian: {ham}")


#############################
## SLURM script generation ##
#############################

def make_slurm_script(
    job_name, ham, size, lr, method, cores, qos, walltime
):
    output_file = f"{job_name}.txt"

    # Generate one main.py call for each parameter configuration.
    hamiltonian_names = get_hamiltonian_names(ham, size)

    python_calls = []

    for hamiltonian_name in hamiltonian_names:
        python_calls.append(
            f'python3 main.py \\\n'
            f'    --workers {cores} \\\n'
            f'    --H "{hamiltonian_name}" \\\n'
            f'    --method {method} \\\n'
            f'    --L {lr} \\\n'
            f'    --R {lr} \\\n'
            f'    --algorithms "QFAMES"'
        )

    # Run all parameter configurations sequentially in one Slurm job.
    commands = "\n\n".join(python_calls)

    return f"""#!/bin/bash -l

#SBATCH --job-name={job_name}
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

{commands}
"""


##########
## Main ##
##########

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    generated = 0

    for ham in HAMILTONIANS:
        for method in METHODS:
            for size in range(MIN_SIZE, MAX_SIZE + 1):

                # Only use powers of two.
                if size & (size - 1):
                    continue

                # LR values up to the Hamiltonian size.
                for lr in LR_VALUES:
                    if lr > size:
                        continue

                    job_name = f"{ham}_{size}x{size}-{lr}x{lr}_{method}"

                    cores, qos, walltime = get_resources(
                        lr, size, method
                    )

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

                    print(
                        f"\tCreated {output_path} "
                        f"[{cores} cores, {qos}, {walltime}; "
                        f"{len(get_hamiltonian_names(ham, size))} runs]"
                    )

    print(f"Generated {generated} SLURM scripts.")


if __name__ == "__main__":
    main()
