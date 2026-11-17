from pathlib import Path


###################
## Configuration ##
###################

ACCOUNT = "csit"
PARTITION = "cpu"

OUTPUT_DIR = Path("data/bash")

HAMILTONIANS = ["XXZ", "TFIM"]
METHODS = ["numeric"]

TFIM_params = [0.5]
XXZ_params = [[0, 1], [0, 2]]

GENERAL = False

# if GENERAL = True, use the configuration below
MIN_SIZE = 16
MAX_SIZE = 4098

LR_VALUES = [6, 8, 10, 12, 16, 32]

# elif GENERAL = False, use the configuration below
# Keys are exact Hamiltonian names; values are lists of LR values to test.
lr_dict = {
    # "XXZ_(h=0)-(d=1)_16x16":   [4, 6, 8],
    # "XXZ_(h=0)-(d=1)_32x32":   [8, 10, 12],
    # "XXZ_(h=0)-(d=1)_64x64":   [10, 20, 25],
    # "XXZ_(h=0)-(d=1)_128x128": [15, 25, 35],
    "XXZ_(h=0)-(d=1)_256x256": [15],
    "XXZ_(h=0)-(d=1)_512x512": [15],
    "XXZ_(h=0)-(d=1)_1024x1024": [15],
    # "XXZ_(h=0)-(d=2)_16x16":   [9, 11, 13],
    # "XXZ_(h=0)-(d=2)_32x32":   [14, 16, 18],
    "XXZ_(h=0)-(d=2)_256x256": [15],
    "XXZ_(h=0)-(d=2)_512x512": [15],
    "XXZ_(h=0)-(d=2)_1024x1024": [15],
    # "TFIM_(g=0.5)_16x16":      [16],
    "TFIM_(g=0.5)_32x32":      [15],
    "TFIM_(g=0.5)_64x64":      [15],
    "TFIM_(g=0.5)_128x128":      [15],
    "TFIM_(g=0.5)_256x256":      [15],
    "TFIM_(g=0.5)_512x512":      [15],
    "TFIM_(g=0.5)_1024x1024":      [15],
}


#####################
## Resource policy ##
#####################

def get_resources(lr, size, method):
    """
    Return (cores, qos, walltime) for a given experiment.

    walltime is in Slurm's HH:MM:SS format.
    """

    if lr <= 15:
        quality = "standby"
    else:
        quality = "normal"

    if quality == "standby":
        hours = "04:00:00"
    elif quality == "normal":
        if lr <= 16:
            hours = "05:30:00"
        elif lr < 30:
            hours = "32:00:00"
        else:
            hours = "48:00:00"

    return 64, quality, hours


#############################
## Hamiltonian configurations
#############################

def get_hamiltonian_names(ham, size):
    """
    Return all parameterized Hamiltonian names for a given size.

    TFIM_params contains g values.
    XXZ_params contains [h, delta] pairs.
    """

    if ham == "TFIM":
        return [
            f"TFIM_(g={g})_{size}x{size}"
            for g in TFIM_params
        ]

    elif ham == "XXZ":
        return [
            f"XXZ_(h={h})-(d={delta})_{size}x{size}"
            for h, delta in XXZ_params
        ]

    raise ValueError(f"Unknown Hamiltonian: {ham}")


def parse_ham_name(ham_name):
    """
    Extract (ham_type, size) from a Hamiltonian name string.
    e.g. "XXZ_(h=0)-(d=1)_16x16" -> ("XXZ", 16)
         "TFIM_(g=0.5)_16x16"    -> ("TFIM", 16)
    """
    if ham_name.startswith("XXZ"):
        ham_type = "XXZ"
    elif ham_name.startswith("TFIM"):
        ham_type = "TFIM"
    else:
        raise ValueError(f"Cannot parse Hamiltonian type from: {ham_name}")

    # Size is always the last token after the final underscore, e.g. "16x16"
    size_str = ham_name.rsplit("_", 1)[-1]  # "16x16"
    size = int(size_str.split("x")[0])
    return ham_type, size


#############################
## SLURM script generation ##
#############################

def make_slurm_script(job_name, hamiltonian_names, lr, method, cores, qos, walltime):
    output_file = f"{job_name}.txt"

    python_calls = []
    for hamiltonian_name in hamiltonian_names:
        python_calls.append(
            f'python3 python/QFAMES_SV.py \\\n'
            f'    --workers {cores} \\\n'
            f'    --H "{hamiltonian_name}" \\\n'
            f'    --method {method} \\\n'
            f'    --L {lr} \\\n'
            f'    --R {lr} \\\n'
            f'    --perturbations "None,Small,Big,Rand"'
        )

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

    if GENERAL:
        # ── Original behaviour: sweep all sizes × LR values × Hamiltonians ──
        for ham in HAMILTONIANS:
            for method in METHODS:
                for size in range(MIN_SIZE, MAX_SIZE + 1):

                    # Only use powers of two.
                    if size & (size - 1):
                        continue

                    for lr in LR_VALUES:
                        if lr > size:
                            continue

                        job_name = f"{ham}_{size}x{size}-{lr}x{lr}_{method}"
                        hamiltonian_names = get_hamiltonian_names(ham, size)
                        cores, qos, walltime = get_resources(lr, size, method)

                        script = make_slurm_script(
                            job_name=job_name,
                            hamiltonian_names=hamiltonian_names,
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
                            f"{len(hamiltonian_names)} run(s)]"
                        )

    else:
        # ── Specific behaviour: one script per (ham_name, lr) in lr_dict ────
        for method in METHODS:
            for ham_name, lr_values in lr_dict.items():
                ham_type, size = parse_ham_name(ham_name)

                for lr in lr_values:
                    job_name = f"{ham_name}-{lr}x{lr}_{method}"
                    cores, qos, walltime = get_resources(lr, size, method)

                    # Each script runs only the single named Hamiltonian.
                    script = make_slurm_script(
                        job_name=job_name,
                        hamiltonian_names=[ham_name],
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
                        f"[{cores} cores, {qos}, {walltime}]"
                    )

    print(f"\nGenerated {generated} SLURM script(s).")


if __name__ == "__main__":
    main()