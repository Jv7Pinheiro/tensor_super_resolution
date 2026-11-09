import argparse
import os
import sys
import time

import pandas as pd
import numpy as np
np.set_printoptions(linewidth=200)

import hamiltonians
import par_comp
import plotting
import aux_functions

from collections import namedtuple
EigenCluster = namedtuple("EigenCluster", ["eigenvalue", "multiplicity"])

PERTURBATION_OPTIONS = ("None", "Small", "Big", "Rand", "Stand")

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
        "--method",
        type=str,
        default="circuit",
        help="Z-tensor's generation method for QFAMES and TSRHSE (default: circuit)",
    )
    parser.add_argument(
        "--perturbations",
        type=str,
        default=",".join(PERTURBATION_OPTIONS),
        help="Comma-separated perturbations to test "
        f"(choices: {', '.join(PERTURBATION_OPTIONS)})",
    )
    args = parser.parse_args()
    args.perturbations = parse_csv_options(args.perturbations, PERTURBATION_OPTIONS, "--perturbations")
    return args

def select_diverse_columns(PHI, eigenvalues, target_count, tol=1e-6):
    """
    Select `target_count` columns from PHI such that the first min(target_count,
    n_unique) columns come from distinct eigenvalues (one per eigenvalue, in the
    order they appear after sorting). Remaining slots are filled by cycling
    through the eigenvalues again in the same order.

    Parameters
    ----------
    PHI         : (n, n) array of eigenvectors (columns), as returned by np.linalg.eig
    eigenvalues : (n,) array of corresponding eigenvalues (same order as PHI columns)
    target_count: int, number of columns to select (L or R)
    tol         : float, tolerance for grouping near-duplicate eigenvalues

    Returns
    -------
    selected : (n, target_count) array
    """
    # Group column indices by eigenvalue (sorted order)
    sorted_idx = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[sorted_idx]

    groups = []  # each entry is a list of column indices sharing one eigenvalue
    i = 0
    while i < len(sorted_evals):
        j = i + 1
        while j < len(sorted_evals) and abs(sorted_evals[j] - sorted_evals[i]) < tol:
            j += 1
        groups.append(sorted_idx[i:j].tolist())
        i = j

    # Build selection by cycling: one from each group, then repeat
    pointers = [0] * len(groups)  # next unused index within each group
    selected_cols = []
    group_cycle = 0
    while len(selected_cols) < target_count:
        g = group_cycle % len(groups)
        col = groups[g][pointers[g] % len(groups[g])]
        selected_cols.append(col)
        pointers[g] += 1
        group_cycle += 1

    return PHI[:, selected_cols]

def QFAMES_WITH_PLOTS(Z, dx, t_list, K, T, tau, alpha, true_multiplicities, path, verbose=False):
    """
    nonorthogonal QMEGS algorithm

    To avoid long classical running time, we first do a rough search
    then do a detailed search around the rough maximal point.
    """
    L = len(Z[:,0,0])
    R = len(Z[0,:,0])
    N = len(Z[0,0,:])
    W = np.zeros((L, R), dtype='complex') #Store data matrix
    num_x=int(2*np.pi/(dx*10))
    num_x_detail=int(2/dx/T)
    x_rough=np.arange(0,num_x)*dx*10-np.pi
    G = np.zeros(len(x_rough), dtype='complex') #Store Frobenius norm of the data matrix (for rough search)
    for l in range(L):
        for r in range(R):
            G += np.abs(Z[l, r, :].dot(np.exp(1j*np.outer(t_list,x_rough)))/N)**2
    Dominant_freq=np.zeros(K,dtype='complex')
    Dominant_num=np.zeros(K,dtype='int')
    Dominant_vector=[]
    num_found = 0
    svs = np.zeros([L, K], dtype=complex)
    for k in range(K):
        max_idx_rough = np.argmax(G) # Rough search
        # print('maxG',max(G))
        if G[max_idx_rough] < 1e-6:
            if verbose: print('No more clusters found, exiting early.')
            break
        num_found += 1
        Dominant_potential=x_rough[max_idx_rough]
        x=np.arange(0,num_x_detail)*dx+Dominant_potential-1/T
        G_detail = np.zeros(len(x), dtype='complex') #Store Frobenius norm of the data matrix (for detail search)
        for l in range(L):
            for r in range(R):
                G_detail += np.abs(Z[l, r, :].dot(np.exp(1j*np.outer(t_list,x)))/N)**2
        max_idx_detail = np.argmax(G_detail) # Detail search
        Dominant_freq[k]=x[max_idx_detail] #one cluster found

        # Calculate the number of eigenvalues in the cluster
        for l in range(L):
            for r in range(R):
                W[l,r] = Z[l, r, :].dot(np.exp(1j*t_list*Dominant_freq[k]))/N
        U, S, V = np.linalg.svd(W)
        svs[:, k] = S
        V = V.conj().T

        Dominant_num[k] = int(np.sum(S > tau))
        if Dominant_num[k]>0: #if there is a cluster, calculate the orthogonal vector space
        #    print('k=',k,'S=',S)
            # Calculate the orthogonal vector space of the cluster
           if verbose:
                if L>R:
                    Dominant_vector.append(U[:,Dominant_num[k]:])
                else:
                    Dominant_vector.append(V[:,Dominant_num[k]:])
        # interval_max=x[max_idx_detail]+alpha/T
        # interval_min=x[max_idx_detail]-alpha/T
        min_exclusion = 6 * dx * 10  # at minimum exclude 3 rough grid steps on each side
        interval_half = max(alpha / T, min_exclusion)
        interval_max = x[max_idx_detail] + interval_half
        interval_min = x[max_idx_detail] - interval_half

        # print('interval_min=',interval_min,'interval_max=',interval_max)
        G=np.multiply(G,x_rough>interval_max)+np.multiply(G,x_rough<interval_min) #eliminate interval

    plotting.plot_G(Dominant_freq, svs, tau, path, true_multiplicities=true_multiplicities)

    # Bundle each eigenvalue with its multiplicity so the pairing is never lost
    clusters = [
        EigenCluster(eigenvalue=float(np.real(Dominant_freq[k])), multiplicity=int(Dominant_num[k]))
        for k in range(K)
        if Dominant_num[k] > 0 or not np.isnan(Dominant_freq[k])
    ]
    clusters.sort(key=lambda c: c.eigenvalue)

    if verbose:
        return clusters, Dominant_vector
    else:
        return clusters

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
    gaps = np.diff(unique_evals) # unique_evals already sorted
    delta_min = float(np.min(gaps)) if len(gaps) > 0 else 1.0

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

    T_max_array = np.array([200]) # scaling T_max
    all_perturbation_params = { # scale perturbations
        "None": {"range": None, "scale": None},
        "Small": {"range": 1, "scale": 0.5},
        "Big": {"range": 3, "scale": 1},
        "Rand": {"range": None, "scale": None},
        "Stand": {"range": None, "scale": None}
    }
    perturbation_params = {perturbation: all_perturbation_params[perturbation] for perturbation in args.perturbations}

    print(f"\n\n")
    for T_max in T_max_array:
        print(f"T_max = {T_max}")

        # Set alpha so the exclusion window is half the minimum gap
        alpha = delta_min * T_max / 4
        for perturb in perturbation_params.keys():
            print(f"\tPerturb = {perturb}")

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

            if perturb in ("None", "Small", "Big"):
                # PHI columns correspond to eigenvalues — pick diverse ones
                U_list = select_diverse_columns(PHI, eigenvalues, L)
                V_list = select_diverse_columns(PHI, eigenvalues, R)
            else:
                # Rand and Stand have no eigenvalue structure — just take first L/R columns
                U_list = PHI[:, 0:L]
                V_list = PHI[:, 0:R]

            #############################
            ## ACTUAL TEST STARTS HERE ##
            #############################
            # Obtain auxiliary parameters
            dx, tau, t_list, T_max_alg, T_total, torN = aux_functions.QFAMES_setup(M, min(L, R), U_list, V_list, T_max=T_max, is_unitary=is_unitary, q=0.05)
            path = f"data/QFAMES_figs/{name}_tau-{tau:g}_alpha={alpha:.2f}_Tmax-{T_max}_perturb-{perturb}"
            # Generate Z tensor
            if method == "numeric":
                # Generate Z_Tensor
                data_start_time = time.perf_counter()
                Z = par_comp.generate_Z_tensor(M, torN, U_list, V_list, L, R, t_list, is_unitary=is_unitary, shots=-1, workers=workers, method="numeric")
                data_end_time = time.perf_counter()
                print(f"\t\tfinished Z tensor numeric creation in {data_end_time - data_start_time:.6f} seconds; shape of Z is {Z.shape}")

            elif method == "circuit":
                # Generate Z_tensor
                data_start_time = time.perf_counter()
                Z = par_comp.generate_Z_tensor(M, torN, U_list, V_list, L, R, t_list, shots=shots, is_unitary=is_unitary, workers=workers, method="circuit")
                data_end_time = time.perf_counter()
                print(f"\t\tfinished Z tensor circuit creation in {data_end_time - data_start_time:.6f} seconds; shape of Z is {Z.shape}; {shots} shots")

            else:
                raise ValueError(f"Method {method} is unsurported, choose either \"numeric\" or \"circuit\"")

            # QFAMES
            start_time = time.perf_counter()
            clusters = QFAMES_WITH_PLOTS(Z, dx, t_list, min(L, R), T_max_alg, tau, alpha, dict_multiplicities, f"{path}_EXP1.png")
            plotting.plot_sv_vs_theta(Z, t_list, T_max_alg, tau, dict_multiplicities, path=f"{path}_EXP2.png")
            end_time = time.perf_counter()

            print(f"\t\tfinished QFAMES with plots in {end_time - start_time:.6f} seconds")

            est_eigenvalue = [c.eigenvalue for c in clusters] # already sorted by eigenvalue
            multiplicities = str({c.eigenvalue: c.multiplicity for c in clusters})

            print(f"\t\t{multiplicities}")
if __name__ == "__main__":
    main()