import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(linewidth=200)

import TSRHSE
import algorithms
import aux_functions
import hamiltonians
import par_comp

from scipy.signal import find_peaks


# TODO:
# Mess with TAU
# Get the bar figures
# Somehow test T_Total scaling
# Take a look at Zhang's notes

PERTURBATION_OPTIONS = ("None", "Small", "Big", "Rand", "Stand")
ALGORITHM_OPTIONS = ("QPE", "KQPE", "QMEGS", "Jennrich_V1", "Jennrich_V2", "CP_ALS", "QFAMES")

def parse_csv_options(value, valid_options, argument_name):
    user_inputs = [option.strip() for option in value.split(",")]

    if not value.strip() or any(not option for option in user_inputs):
        raise argparse.ArgumentTypeError(f"{argument_name} must be a comma-separated list of valid options")

    invalid_options = sorted(set(user_inputs) - set(valid_options))
    if invalid_options:
        valid = ", ".join(valid_options)
        invalid = ", ".join(invalid_options)
        raise argparse.ArgumentTypeError(f"{argument_name} contains invalid option(s): {invalid}. Valid options are: {valid}")

    duplicate_options = sorted(option for option in set(user_inputs) if user_inputs.count(option) > 1)
    if duplicate_options:
        duplicates = ", ".join(duplicate_options)
        raise argparse.ArgumentTypeError(f"{argument_name} contains duplicate option(s): {duplicates}")

    return user_inputs

def parse_args():
    parser = argparse.ArgumentParser(description="Run TSR/QFAMES benchmark suite")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel Z-tensor generation (default: 4)",
    )
    parser.add_argument(
        "--H",
        type=str,
        default="belldiagonal4x4",
        help="Hamiltonian name to use for benchmarking (default: belldiagonal4x4)",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=0,
        help="Number of left coefficients for QFAMES and TSRHSE (default: 0 => same length as Hamiltonian)",
    )
    parser.add_argument(
        "--R",
        type=int,
        default=0,
        help="Number of right coefficients for QFAMES and TSRHSE (default: 0 => same length as Hamiltonian)",
    )
    parser.add_argument(
        "--perturbations",
        type=str,
        default=",".join(PERTURBATION_OPTIONS),
        help="Comma-separated perturbations to test "
        f"(choices: {', '.join(PERTURBATION_OPTIONS)})",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(ALGORITHM_OPTIONS),
        help="Comma-separated algorithms to test "
        f"(choices: {', '.join(ALGORITHM_OPTIONS)})",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="circuit",
        help="Z-tensor's generation method for QFAMES and TSRHSE (default: circuit)",
    )
    parser.add_argument(
        "--genG",
        type=int,
        default=0,
        help="Comuptes G signaling tensor from G tensor (default: True)",
    )
    args = parser.parse_args()
    args.perturbations = parse_csv_options(args.perturbations, PERTURBATION_OPTIONS, "--perturbations")
    args.algorithms = parse_csv_options(args.algorithms, ALGORITHM_OPTIONS, "--algorithms")
    return args

def get_errors(true, estimate, target):
    true = np.asarray(true, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()

    # Single estimate case: keep the original behavior for scalar outputs.
    if estimate.size == 1:
        if true[target] < 0:
            return abs(true[target] + estimate[0])
        else:
            return abs(true[target] - estimate[0])

    # Pairwise absolute errors: cost[i, j] = |estimate_i - true_j|
    cost = np.abs(estimate[:, None] - true[None, :])

    n_est = estimate.size
    n_true = true.size

    used_est = np.zeros(n_est, dtype=bool)
    used_true = np.zeros(n_true, dtype=bool)
    estimate_index_for_true = {}

    # Greedy matching: repeatedly choose the smallest remaining pair while
    # excluding already-used estimates and true eigenvalues.
    for _ in range(min(n_est, n_true)):
        best_cost = np.inf
        best_i = None
        best_j = None

        for i in range(n_est):
            if used_est[i]:
                continue
            for j in range(n_true):
                if used_true[j]:
                    continue
                if cost[i, j] < best_cost:
                    best_cost = cost[i, j]
                    best_i = i
                    best_j = j

        if best_i is None or best_j is None:
            break

        used_est[best_i] = True
        used_true[best_j] = True
        estimate_index_for_true[best_j] = best_i

    # Return errors in the SAME ORDER as the true eigenvalues array.
    errors = np.full(n_true, np.nan, dtype=float)
    for j, i in estimate_index_for_true.items():
        errors[j] = abs(estimate[i] - true[j])

    return errors.tolist()

def normalize_output_energy(output_energy, sort=False):
    values = np.asarray(output_energy, dtype=object).ravel().tolist()
    missing_count = sum(value is None for value in values)
    values = [value for value in values if value is not None]
    energies = np.real(np.asarray(values)).tolist()
    return sorted(energies) if sort else energies

def record_result(name, df, test_type, param_value, perturb, algorithm, shots, torN, T_max, T_total, eval_estimate, errors, multiplicities):
    results_row = {
        "test_type": test_type,
        "param_value": param_value,
        "perturb": perturb,
        "algorithm": algorithm,
        "shots": shots,
        "t or N": torN,
        "T_max": T_max,
        "T_total": T_total,
        "eval(s)": eval_estimate,
        "errors": errors,
        "multiplicities": multiplicities
    }

    df = pd.concat([df, pd.DataFrame([results_row])], ignore_index=True)
    df.to_csv(f"data/dataframes/{name}.csv", index=False)

    return df

def plot_G(thetas, G, eigenvalues, tau, LR, path, perturb, peak_height=0.5, title=""):
    """
    Left panel : ||G(theta)||_F^2 versus theta, one vertical dashed line per
                 eigenvalue, and an orange dotted line at each detected peak.
    Right panels: one panel per detected peak (up to LR), arranged in rows of
                  3 columns, showing singular values of G(theta*_k) as bars
                  with the threshold tau as a red dashed horizontal line.

    Parameters
    ----------
    thetas : (J,) array     the theta grid G was evaluated on
    G : (L, R, J) array     output of par_comp.generate_G_tensor
    eigenvalues : array     eigenvalues of M (any order; complex is fine, the real part is used)
    tau : float             singular-value threshold (the one QFAMES_setup returns)
    path : str              where to save the figure
    peak_height : float     a peak of ||G||_F^2 must be at least this tall to count.
                            This is an absolute number, so it depends on how G is
                            normalized; lower it if no peak is found.
    title : str             optional title for the left panel

    Returns
    -------
    theta_stars : list of float      locations of all detected peaks
    svs : list of arrays             singular values of G(theta*_k) for each peak, largest first
    """
    import matplotlib.gridspec as gridspec

    norm2 = np.sum(np.abs(G) ** 2, axis=(0, 1))

    peaks, _ = find_peaks(norm2, height=peak_height)

    if len(peaks) == 0:
        peak_indices = [int(np.argmax(norm2))]
    else:
        peaks_sorted = sorted(peaks, key=lambda i: norm2[i], reverse=True)
        peak_indices = [int(i) for i in peaks_sorted[:LR]]
        peak_indices.sort()

    n_panels = len(peak_indices)
    svs = [np.linalg.svd(G[:, :, idx], compute_uv=False) for idx in peak_indices]

    n_sv_cols = 3
    n_sv_rows = max(1, (n_panels + n_sv_cols - 1) // n_sv_cols)

    fig = plt.figure(figsize=(13, 3.5 + 3.5 * (n_sv_rows - 1)))
    if title:
        fig.suptitle(title, fontsize=16)

    gs_left = gridspec.GridSpec(1, 1, left=0.05, right=0.28, top=0.88, bottom=0.12)
    gs_right = gridspec.GridSpec(
        n_sv_rows, n_sv_cols,
        left=0.35, right=0.97,
        top=0.88, bottom=0.12,
        hspace=0.55, wspace=0.4
    )

    ax1 = fig.add_subplot(gs_left[0, 0])
    sv_axes = [fig.add_subplot(gs_right[i // n_sv_cols, i % n_sv_cols]) for i in range(n_panels)]

    ax1.plot(thetas, norm2, lw=2)
    ax1.set_title(rf"$\|G(\theta)\|_F^2$")
    add_eigenvalue_lines(ax1, eigenvalues, LR, perturb)
    ax1.set_xlabel(rf"$\theta$")
    for idx in peak_indices:
        ax1.axvline(thetas[idx], color='orange', linestyle=':', linewidth=1.5)

    for ax2, idx, sv in zip(sv_axes, peak_indices, svs):
        x = np.arange(1, len(sv) + 1)
        ax2.axhline(tau, color='red', linestyle='--', linewidth=2, label=r'$\tau$')
        ax2.bar(x, sv, width=0.4, color='blue', label=rf"$\sigma(G(\theta^\star))$")
        ax2.set_xticks(x)
        ax2.legend(fontsize=10)
        ax2.set_title(rf"SV of $G$ at $\lambda \approx {thetas[idx]:.4f}$")

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

    theta_stars = [thetas[idx] for idx in peak_indices]
    return theta_stars, svs

def add_eigenvalue_lines(ax, eigenvalues, L, perturb):
    red_count = L if perturb in ["None", "Small", "Big"] else len(eigenvalues)

    for eigenvalue in eigenvalues[:red_count]:
        ax.axvline(
            eigenvalue,
            color="red",
            linestyle="--",
            linewidth=1
        )

    for eigenvalue in eigenvalues[red_count:]:
        ax.axvline(
            eigenvalue,
            color="gray",
            linestyle="--",
            linewidth=0.5
        )

def main():
    # Parse Arguments
    args = parse_args()
    method = args.method
    workers = args.workers

    # Set seed
    seed = 82304
    rng = np.random.default_rng(seed)

    ############################
    ## Initialize Hamiltonian ##
    ############################
    try:
        Ham = hamiltonians.get_hamiltonian(args.H)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    
    # Name, initialize, and normalize hamiltonian
    name = args.H
    M = (np.pi / (4 * np.linalg.norm(Ham))) * Ham
    n = M.shape[0]

    # Set length of L and R parameters
    L = args.L
    R = args.R
    rows, cols = M.shape
    if (L > rows or R > cols) or (L < 0 or R < 0):
        raise ValueError(f"Inputs L ({L}) and R ({R}) must be positive integers less than or equal to matrix size [{rows}, {cols}]")
    if L == 0: L = M.shape[0]
    if R == 0: R = M.shape[1]

    # Obtain information about matrix
    is_unitary = aux_functions.is_matrix_unitary(M)
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Choose Init State: Targeted eigenvalue for QPE, KQPE, and QMEGS
    lambda_i = 0 # This is the index of the INIT state, # If 1 then QPE and KQPE need a scaling factor greater than ||M||
    eigenvectors = np.real(eigenvectors)
    eigenvalues = np.real(eigenvalues)
    eigenvalue = eigenvalues[lambda_i]
    
    # Obtain eigenvalue multiplicities up to floating point error
    sorted_eigenvalues = np.sort(np.real(eigenvalues))
    tol = 1e-10
    groups = []
    counts = []
    i = 0
    while i < len(sorted_eigenvalues):
        j = i + 1
        while j < len(sorted_eigenvalues) and abs(sorted_eigenvalues[j] - sorted_eigenvalues[i]) < tol:
            j += 1
        groups.append(np.mean(sorted_eigenvalues[i:j]))  # use mean of the group as representative
        counts.append(j - i)
        i = j
    unique_evals = np.array(groups)
    true_multiplicities = np.array(counts)
    dict_multiplicities = dict(zip(unique_evals.tolist(), true_multiplicities.tolist()))

    # Print Information about my matrix
    name = f"{name}-{L}x{R}_{method}"
    print(name)
    print(f"My Matrix: \n{M}\n")
    print(f"L = {L}, R = {R}")
    print(f"is_unitary: {is_unitary}")
    print(f"Norm of my matrix: {np.linalg.norm(M)}")
    print(f"Eigenvalues: {sorted_eigenvalues}")
    print(f"Eigenvalues: {unique_evals}")
    print(f"Evals Mult:  {true_multiplicities}")
    print(f"Target eigenvalue: {eigenvalue}")
    # print(f"Eigenvectors: \n{eigenvectors}")


    #########################
    ## Set Test Parameters ##
    #########################
    # Keep the existing grouped execution for the four Z-tensor algorithms.
    non_z_tensor_algorithms = ["QPE", "KQPE", "QMEGS"]
    z_tensor_methods = ["Jennrich_V1", "Jennrich_V2", "CP_ALS", "QFAMES"]
    algorithms_array = [algorithm for algorithm in non_z_tensor_algorithms if algorithm in args.algorithms]
    Z_tensor_methods = [algorithm for algorithm in z_tensor_methods if algorithm in args.algorithms]
    if Z_tensor_methods: algorithms_array.append("Z_Tensor_Methods")

    # Verbosity parameters
    verbosity = 0
    verbose = True if verbosity > 0 else False

    # Number of perturbations and their strength
    all_perturbation_params = {
        "None": {"range": None, "scale": None},
        "Small": {"range": 1, "scale": 0.5},
        "Big": {"range": 3, "scale": 1},
        "Rand": {"range": None, "scale": None},
        "Stand": {"range": None, "scale": None},
    }
    perturbation_params = {perturbation: all_perturbation_params[perturbation] for perturbation in args.perturbations}

    # Test configurations: iterate through eps_array and T_max_array separately
    # eps_array = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    T_max_array = np.array([1000]) # , 200, 400, 800, 1200, 1600, 2000, 3200
    test_configs = {
        # "eps": {"array": eps_array, "name": "eps"},
        "T_max": {"array": T_max_array, "name": "T_max"}
    }

    # Number of shots for each main category of algorithm
    # Z_tensor_shots_array is not applicable when command line argument --method is set to "numeric"
    # TODO: Edit QMEGS framework such that it can create numeric Z array
    qpe_shots = 1000 # Applies to QPE and KQPE
    Z_array_shots_array = np.array([1500]) # Applies to QMEGS
    Z_tensor_shots_array = np.array([1500]) # Applies to QFAMES and TSRHSE


    ################
    ## Begin Test ##
    ################
    # Create output directory if it doesn't exist
    os.makedirs("data/dataframes", exist_ok=True)

    results_df = pd.DataFrame()
    true_evals_row = { # Add row with true eigenvalues for reference
        "test_type": "true eigenvalues",
        "param_value": None,
        "perturb": None,
        "algorithm": None,
        "shots": None,
        "t or N": None,
        "T_max": None,
        "T_total": None,
        "eval(s)": unique_evals.tolist(),
        "errors": None,
        "multiplicities": str(dict_multiplicities)
    }
    results_df = pd.concat([results_df, pd.DataFrame([true_evals_row])], ignore_index=True)

    # Iterate through the two different tests being performed: scaling epsilon and T_max
    for test_type, config in test_configs.items():
        param_array = config["array"]
        print(f"\n\ntest_type = {test_type}, param_array = {param_array}")

        # Iterate through the scaling of eps and T_max
        for param_value in param_array:
            # Set eps and T_max: while one is active, the other must be None
            if test_type == "eps":
                eps = param_value
                T_max = None
            else:
                eps = None
                T_max = param_value

            # In each test_type and parameter we want to study the different perturbations
            for perturb in perturbation_params.keys():
                # Obtain current test's perturbation 
                perturb_range = perturbation_params[perturb]["range"] # Ontain current perturbation's range
                perturb_scale = perturbation_params[perturb]["scale"] # Ontain current perturbation's scale

                # Apply perturbation if needed
                if perturb == "Rand":
                    PHI = rng.standard_normal((n, n))
                    PHI = PHI / np.linalg.norm(PHI, axis=0, keepdims=True)
                elif perturb == "Stand":
                    PHI = np.eye(n, dtype=eigenvectors.dtype)
                elif perturb == "None":
                    PHI = eigenvectors
                else: # "Small" and "Large"
                    # Create the new Init state and U_list unitaries
                    PHI = eigenvectors + np.random.uniform(-perturb_range, perturb_range) * perturb_scale

                U_list = PHI[:, 0:L]
                V_list = PHI[:, 0:R]
                Init = PHI[:, lambda_i]

                # Print information about current test
                print(f"Test: {test_type} = {param_value}, perturb = {perturb}")

                # Perform the above test for each selected algorithm
                for alg in algorithms_array:
                    ##################
                    ## QPE and KQPE ##
                    ##################
                    if alg == "QPE" or alg == "KQPE":
                        function = getattr(algorithms, alg)

                        # Run Algorithm
                        start_time = time.perf_counter()
                        _, phases, my_eigenvalue, _, T_max_alg, T_total, torN = function(M, Init=Init, eps=eps, T_max=T_max, shots=qpe_shots, is_unitary=is_unitary, verbosity=verbosity)
                        end_time = time.perf_counter()
                        print(f"\tfinished {alg} in {end_time - start_time:.6f} seconds")

                        # Update Data Frame
                        results_df = record_result(name, results_df, test_type, param_value, perturb, alg, qpe_shots, torN, T_max_alg, T_total, my_eigenvalue, get_errors(eigenvalues, my_eigenvalue, lambda_i), None)
                    

                    ###########
                    ## QMEGS ##
                    ###########
                    elif alg == "QMEGS":
                        for shots in Z_array_shots_array:
                            # Generate Z_array and other QMEGS required arguments
                            data_start_time = time.perf_counter()
                            Z, dx, t_list, K, T_max_alg, T_total, torN = aux_functions.QMEGS_setup(M, Init, shots=shots, eps=eps, T_max=T_max, is_unitary=is_unitary)
                            data_end_time = time.perf_counter()
                            print(f"\tfinished Z array creation in {data_end_time - data_start_time:.6f} seconds")

                            # Run Algorithm
                            start_time = time.perf_counter()
                            output_energy = algorithms.QMEGS(Z, dx, t_list, K, T_max_alg)
                            end_time = time.perf_counter()
                            print(f"\tfinished {alg} in {end_time - start_time:.6f} seconds")

                            # Update Data Frame
                            my_eigenvalue = normalize_output_energy(output_energy, sort=True)
                            results_df = record_result(name, results_df, test_type, param_value, perturb, alg, shots, torN, T_max_alg, T_total, my_eigenvalue, get_errors(eigenvalues, my_eigenvalue, lambda_i), None)


                    ####################################
                    ## QFAMES and all TSRHSE variants ##
                    ####################################
                    else:
                        # Obtain t_list and other QFAMES required arguments
                        dx, tau, t_list, T_max_alg, T_total, torN = aux_functions.QFAMES_setup(M, min(L, R), U_list, V_list, eps=eps, T_max=T_max, is_unitary=is_unitary, q=0.05, verbose=verbose)
                        # This function runs all selected Z_tensor_methods
                        def run_Z_tensor_methods(shots):
                            nonlocal results_df

                            ###############################
                            ## TSRHSE Jennrich version 1 ##
                            ###############################
                            if "Jennrich_V1" in Z_tensor_methods:
                                # Run Algorithm
                                start_time = time.perf_counter()
                                output_energy = algorithms.TSRHSE(Z, t_list)
                                end_time = time.perf_counter()
                                print(f"\tfinished TSRHSE Jennrich version 1 in {end_time - start_time:.6f} seconds")
                                
                                # Update Data Frame
                                my_eigenvalue = normalize_output_energy(output_energy, sort=True)
                                results_df = record_result(name, results_df, test_type, param_value, perturb, "Jennrich_V1", shots, torN, T_max_alg, T_total, my_eigenvalue, get_errors(eigenvalues, my_eigenvalue, lambda_i), None)
                            

                            ###############################
                            ## TSRHSE Jennrich version 2 ##
                            ###############################
                            if "Jennrich_V2" in Z_tensor_methods:
                                # Run Algorithm
                                start_time = time.perf_counter()
                                output_energy = TSRHSE.jennrich_ladder(Z, t_list, min(L, R))
                                end_time = time.perf_counter()
                                print(f"\tfinished TSRHSE Jennrich version 2 in {end_time - start_time:.6f} seconds")
                                
                                # Update Data Frame
                                my_eigenvalue = normalize_output_energy(output_energy, sort=True)
                                results_df = record_result(name, results_df, test_type, param_value, perturb, "Jennrich_V2", shots, torN, T_max_alg, T_total, my_eigenvalue, get_errors(eigenvalues, my_eigenvalue, lambda_i), None)


                            ###################
                            ## TSRHSE CP ALS ##
                            ###################
                            if "CP_ALS" in Z_tensor_methods:
                                # Run Algorithm
                                start_time = time.perf_counter()
                                output_energy, _, _ = TSRHSE.cp_eigenphases(Z, t_list, min(L, R), verbosity=verbosity)
                                end_time = time.perf_counter()
                                print(f"\tfinished TSRHSE CP ALS in {end_time - start_time:.6f} seconds")
                                
                                # Update Data Frame
                                my_eigenvalue = normalize_output_energy(output_energy, sort=True)
                                results_df = record_result(name, results_df, test_type, param_value, perturb, "CP_ALS", shots, torN, T_max_alg, T_total, my_eigenvalue, get_errors(eigenvalues, my_eigenvalue, lambda_i), None)

                            
                            ############
                            ## QFAMES ##
                            ############
                            if "QFAMES" in Z_tensor_methods:
                                # Run Algorithm
                                start_time = time.perf_counter()
                                clusters = algorithms.QFAMES(Z, dx, t_list, min(L, R), T_max_alg, tau, alpha=0.0005, verbose=False, true_multiplicities=dict_multiplicities)
                                end_time = time.perf_counter()
                                print(f"\tfinished QFAMES in {end_time - start_time:.6f} seconds")


                                # Update Data Frame
                                my_eigenvalue = [c.eigenvalue for c in clusters] # already sorted by eigenvalue
                                multiplicities = str({c.eigenvalue: c.multiplicity for c in clusters})
                                results_df = record_result(name, results_df, test_type, param_value, perturb, "QFAMES", shots, torN, T_max_alg, T_total, my_eigenvalue, get_errors(eigenvalues, my_eigenvalue, lambda_i), multiplicities)

                        if method == "numeric":
                            # Generate Z_Tensor
                            data_start_time = time.perf_counter()
                            Z = par_comp.generate_Z_tensor(M, torN, U_list, V_list, L, R, t_list, is_unitary=is_unitary, shots=-1, workers=workers, method="numeric")
                            data_end_time = time.perf_counter()
                            print(f"\tfinished Z tensor numeric creation in {data_end_time - data_start_time:.6f} seconds; shape of Z is {Z.shape}")
                            run_Z_tensor_methods(-1)

                        elif method == "circuit":
                            for shots in Z_tensor_shots_array:
                                # Generate Z_tensor
                                data_start_time = time.perf_counter()
                                Z = par_comp.generate_Z_tensor(M, torN, U_list, V_list, L, R, t_list, shots=shots, is_unitary=is_unitary, workers=workers, method="circuit")
                                data_end_time = time.perf_counter()
                                print(f"\tfinished Z tensor circuit creation in {data_end_time - data_start_time:.6f} seconds; shape of Z is {Z.shape}; {shots} shots")
                                run_Z_tensor_methods(shots)

                        else:
                            raise ValueError(f"Method {method} is unsurported, choose either \"numeric\" or \"circuit\"")

                        if args.genG:
                            margin = 5 / T_max
                            thetas = thetas = np.arange(eigenvalues.min() - margin, eigenvalues.max() + margin, dx)
                            G = par_comp.generate_G_tensor(Z, t_list, thetas, workers=workers)

                            title = f"{name}\ntau = {tau}, Tmax={T_max}, Perturbation={perturb}"
                            
                            # Plot matrix elements of G
                            fig, axes = plt.subplots(L, R, figsize=(2*L + L, 2*R), squeeze=False)
                            fig.suptitle(title, fontsize=16)
                            for i in range(L):
                                for j in range(R):
                                    ax = axes[i, j]
                                    ax.plot(thetas, np.abs(G[i,j,:]), linewidth=2)
                                    ax.set_title(f"|G_[{i},{j}]|", fontsize=12)
                                    add_eigenvalue_lines(ax, eigenvalues, L, perturb)

                            plt.tight_layout()
                            plt.savefig(f"data/figures/{name}_tau-{tau}_Tmax-{T_max}_perturb-{perturb}_G_elements.pdf")

                            path = f"data/figures/{name}_tau-{tau}_Tmax-{T_max}_perturb-{perturb}_G.pdf"
                            theta_star, sv = plot_G(thetas, G, eigenvalues, tau, L, path, perturb, title=title)

if __name__ == "__main__":
    main()