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

import warnings

warnings.filterwarnings(
    "ignore",
    # message="No artists with labels found to put in legend.*",
    category=UserWarning,
)

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
        "--plotting",
        type=int,
        default="0",
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

def score_multiplicities(estimated_dict, true_dict, unique_evals, tol=0.05):
    """
    Compare estimated multiplicities against ground truth.

    Each estimated eigenvalue is matched to at most one true eigenvalue
    (greedy nearest-neighbour, closest unmatched pair first).  True
    eigenvalues with no estimate within `tol` count as estimate=0.

    Returns
    -------
    n_correct    : int  — eigenvalues whose multiplicity was exact
    total_error  : int  — sum of |estimated - true| over all true eigenvalues
    n_found      : int  — true eigenvalues matched to any estimate
    details      : list of (true_eval, true_mult, est_mult) tuples
    """
    # Work with a mutable copy so we can remove matched entries
    unmatched_est = list(estimated_dict.items())   # [(theta, mult), ...]

    # Build all (distance, true_idx, est_idx) triples, sort by distance,
    # then greedily assign closest pairs first
    true_items = list(true_dict.items())           # [(eval, mult), ...]
    assignments = {}                               # true_idx -> est_mult (or 0)

    # Collect all candidate pairs within tolerance
    candidates = []
    for ti, (te, tm) in enumerate(true_items):
        for ei, (ee, em) in enumerate(unmatched_est):
            d = abs(ee - te)
            if d < tol:
                candidates.append((d, ti, ei))

    candidates.sort()          # closest first

    used_true = set()
    used_est  = set()
    for d, ti, ei in candidates:
        if ti in used_true or ei in used_est:
            continue
        assignments[ti] = unmatched_est[ei][1]   # est_mult
        used_true.add(ti)
        used_est.add(ei)

    details = []
    for ti, (te, tm) in enumerate(true_items):
        em = assignments.get(ti, 0)              # 0 if eigenvalue not found
        details.append((te, tm, em))

    n_correct   = sum(1 for _, tm, em in details if tm == em)
    total_error = sum(abs(tm - em)        for _, tm, em in details)
    n_found     = sum(1 for _, tm, em in details if em > 0)
    return n_correct, total_error, n_found, details

def _make_summary_figure(df, name, LR):
    """
    One figure per perturbation, showing n_correct and total_error
    side-by-side for every (method_label, tolerance) combination.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    perturbations = df["perturbation"].unique()
    n_perturb = len(perturbations)

    fig, axes = plt.subplots(
        n_perturb, 2,
        figsize=(14, 3.5 * n_perturb),
        squeeze=False)

    # colour palette: original=orange, naive=steelblue, robust=seagreen
    def _colour(label):
        if label.startswith("original"):  return "darkorange"
        if label.startswith("naive"):     return "steelblue"
        return "seagreen"

    for row_idx, perturb in enumerate(perturbations):
        sub = df[df["perturbation"] == perturb].copy()
        labels  = sub["method_label"].tolist()
        correct = sub["n_correct"].tolist()
        errors  = sub["total_error"].tolist()
        n_evals = int(sub["n_evals"].iloc[0])
        colours = [_colour(l) for l in labels]
        x       = np.arange(len(labels))
        bar_w   = 0.6

        ax_c = axes[row_idx, 0]
        ax_e = axes[row_idx, 1]

        ax_c.bar(x, correct, width=bar_w, color=colours, edgecolor="white", linewidth=0.5)
        ax_c.axhline(n_evals, color="black", linestyle="--", linewidth=1, label=f"perfect ({n_evals})")
        if LR < n_evals:
            ax_c.axhline(LR, color="grey", linestyle="--", linewidth=1, label=f"best ({LR})")
        ax_c.set_ylim(0, n_evals + 1)
        ax_c.set_ylabel("# multiplicities correct")
        ax_c.set_title(f"Perturbation = {perturb}  |  correct")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax_c.legend(fontsize=8)

        ax_e.bar(x, errors, width=bar_w, color=colours, edgecolor="white", linewidth=0.5)
        ax_e.set_ylabel("total |error| in multiplicity")
        ax_e.set_title(f"Perturbation = {perturb}  |  total error")
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax_e.legend(fontsize=8)

    # legend patches
    patches = [
        mpatches.Patch(color="darkorange", label="original (hard τ)"),
        mpatches.Patch(color="steelblue",  label="naive adaptive"),
        mpatches.Patch(color="seagreen",   label="robust adaptive"),
    ]
    fig.legend(handles=patches, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.suptitle(f"{name} — multiplicity estimation comparison",
                 fontsize=11, y=1.03)
    fig.tight_layout()

    fig_path = f"data/QFAMES_figs/Summaries/{name}_summary.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary figure saved to {fig_path}")

def QFAMES_WITH_PLOTS(Z, dx, t_list, K, T, tau, alpha, true_multiplicities, path, verbose=False, plotting=False):
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

    if plotting:
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

def naive_adaptive_multiplicity(S, gap_threshold=0.3):
    """
    S: singular values in descending order, from np.linalg.svd
    gap_threshold: minimum relative drop to declare the gap
    
    For each consecutive pair, compute the relative drop:
        drop_j = 1 - S[j+1]/S[j]
    The multiplicity is the position of the first drop exceeding gap_threshold.
    """
    if len(S) == 0 or S[0] < 1e-10:
        return 0
    # Normalize by S[0]
    s = S / S[0]
    # Find the first large relative drop
    for j in range(len(s) - 1):
        if s[j+1] / s[j] < (1 - gap_threshold):
            return j + 1
    return len(s)  # all SVs are signal

def naive_adaptive_QFAMES(Z, dx, t_list, K, T, alpha, adaptivity, true_multiplicities, verbose=False):
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

        Dominant_num[k] = naive_adaptive_multiplicity(S, adaptivity)

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

def robust_adaptive_multiplicity(Z, t_list, theta_star, N, L, R, T, relative_floor=0.05):
    """
    Count signal SVs using local maximality at theta_star.
    
    Genuine degenerate SVs peak AT theta_star.
    Aliased SVs from neighbors peak AWAY from theta_star.
    
    Parameters
    ----------
    relative_floor : float
        Minimum σ_j / σ_1 to consider a SV as potentially signal (default 0.05).
        Filters out floating-point noise SVs that might accidentally be local maxima.
    """
    eps = 1.0 / (4.0 * T)   # quarter-resolution step; much smaller than any eigenvalue gap
    
    def compute_W_at(theta):
        W = np.zeros((L, R), dtype='complex')
        for l in range(L):
            for r in range(R):
                W[l, r] = Z[l, r, :].dot(np.exp(1j * t_list * theta)) / N
        return W
    
    S0 = np.linalg.svd(compute_W_at(theta_star),         compute_uv=False)
    Sp = np.linalg.svd(compute_W_at(theta_star + eps),   compute_uv=False)
    Sm = np.linalg.svd(compute_W_at(theta_star - eps),   compute_uv=False)
    
    if S0[0] < 1e-10:
        return 0
    
    count = 0
    for j in range(len(S0)):
        is_local_max = (S0[j] >= Sp[j]) and (S0[j] >= Sm[j])
        is_above_floor = (S0[j] / S0[0]) > relative_floor
        if is_local_max and is_above_floor:
            count += 1
        else:
            break   # SVs are sorted — once we hit non-signal, all subsequent are too
    
    return max(count, 1)   # always at least 1 since we found a G peak

def robust_adaptive_QFAMES(Z, dx, t_list, K, T, alpha, adaptivity, true_multiplicities, verbose=False):
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

        Dominant_num[k] = robust_adaptive_multiplicity(Z, t_list, Dominant_freq[k], N, L, R, T, relative_floor=adaptivity)

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
    args = parse_args()
    method = args.method
    workers = args.workers

    seed = 82304
    rng = np.random.default_rng(seed)

    try:
        Ham = hamiltonians.get_hamiltonian(args.H)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    name = args.H
    M = (np.pi / (4 * np.linalg.norm(Ham))) * Ham
    n = M.shape[0]

    L = args.L
    R = args.R
    rows, cols = M.shape
    if (L > rows or R > cols) or (L < 0 or R < 0):
        raise ValueError(f"Inputs L ({L}) and R ({R}) must be positive integers less than or equal to matrix size [{rows}, {cols}]")
    if L == 0: L = M.shape[0]
    if R == 0: R = M.shape[1]

    is_unitary   = aux_functions.is_matrix_unitary(M)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigenvectors = np.real(eigenvectors)
    eigenvalues  = np.real(eigenvalues)
    eigenvalue   = eigenvalues[0]

    sorted_eigenvalues = np.sort(np.real(eigenvalues))
    tol_group = 1e-10
    groups, counts = [], []
    i = 0
    while i < len(sorted_eigenvalues):
        j = i + 1
        while j < len(sorted_eigenvalues) and abs(sorted_eigenvalues[j] - sorted_eigenvalues[i]) < tol_group:
            j += 1
        groups.append(np.mean(sorted_eigenvalues[i:j]))
        counts.append(j - i)
        i = j
    unique_evals       = np.array(groups)
    true_multiplicities = np.array(counts)
    dict_multiplicities = dict(zip(unique_evals.tolist(), true_multiplicities.tolist()))
    n_evals = len(dict_multiplicities)
    gaps      = np.diff(unique_evals)
    delta_min = float(np.min(gaps)) if len(gaps) > 0 else 1.0

    name = f"{name}-{L}x{R}_{method}"
    print(name)
    print(f"My Matrix: \n{M}\n")
    print(f"L = {L}, R = {R}")
    print(f"is_unitary: {is_unitary}")
    print(f"Norm of my matrix: {np.linalg.norm(M)}")
    print(f"Eigenvalues: {sorted_eigenvalues}")
    print(f"Eigenvalues: {unique_evals}")
    print(f"Evals Mult:  {true_multiplicities}")

    adaptive_tolerance_1 = np.sqrt(L * R) * np.array([0.1, 0.05, 0.01])
    adaptive_tolerance_2 = np.array([0.15, 0.3, 0.5, 0.7, 0.75, 0.8])
    adaptive_tolerance_3 = np.array([0.01, 0.05, 0.1])

    T_max_array = np.array([200])
    all_perturbation_params = {
        "None":  {"range": None, "scale": None},
        "Small": {"range": 1,    "scale": 0.5},
        "Big":   {"range": 3,    "scale": 1},
        "Rand":  {"range": None, "scale": None},
        "Stand": {"range": None, "scale": None},
    }
    perturbation_params = {p: all_perturbation_params[p] for p in args.perturbations}

    # --- accumulate all results here ---
    records = []   # one dict per method-run

    os.makedirs("data/QFAMES_figs", exist_ok=True)
    os.makedirs("data/dataframes",     exist_ok=True)

    print(f"\n\n")
    for T_max in T_max_array:
        print(f"T_max = {T_max}")
        alpha = delta_min * T_max / 4

        for perturb in perturbation_params:
            print(f"\tPerturb = {perturb}")

            perturb_range = perturbation_params[perturb]["range"]
            perturb_scale = perturbation_params[perturb]["scale"]

            if perturb == "Rand":
                PHI = rng.standard_normal((n, n))
                PHI = PHI / np.linalg.norm(PHI, axis=0, keepdims=True)
            elif perturb == "Stand":
                PHI = np.eye(n, dtype=eigenvectors.dtype)
            elif perturb == "None":
                PHI = eigenvectors
            else:
                PHI = eigenvectors + np.random.uniform(-perturb_range, perturb_range) * perturb_scale

            if perturb in ("None", "Small", "Big"):
                U_list = select_diverse_columns(PHI, eigenvalues, L)
                V_list = select_diverse_columns(PHI, eigenvalues, R)
            else:
                U_list = PHI[:, 0:L]
                V_list = PHI[:, 0:R]

            #############################
            ## ACTUAL TEST STARTS HERE ##
            #############################
            dx, tau, t_list, T_max_alg, T_total, torN = aux_functions.QFAMES_setup(M, min(L, R), U_list, V_list, T_max=T_max, is_unitary=is_unitary, q=0.05)
            path = (f"data/QFAMES_figs/{name}_tau-{tau:g}_alpha={alpha:.2f}_Tmax-{T_max}_perturb-{perturb}")
           
            # Generate Z tensor
            if method == "numeric":
                data_start_time = time.perf_counter()
                Z = par_comp.generate_Z_tensor(M, torN, U_list, V_list, L, R, t_list, is_unitary=is_unitary, shots=-1, workers=workers, method="numeric")
                data_end_time = time.perf_counter()
                print(f"\t\tfinished Z tensor numeric creation in {data_end_time - data_start_time:.6f} seconds; shape of Z is {Z.shape}")

            elif method == "circuit":
                data_start_time = time.perf_counter()
                Z = par_comp.generate_Z_tensor(M, torN, U_list, V_list, L, R, t_list, shots=1500, is_unitary=is_unitary, workers=workers, method="circuit")
                data_end_time = time.perf_counter()
                print(f"\t\tfinished Z tensor circuit creation in {data_end_time - data_start_time:.6f} seconds; shape of Z is {Z.shape}; {shots} shots")

            else:
                raise ValueError(f"Method {method} is unsurported, choose either \"numeric\" or \"circuit\"")

            # helper: run, score, print, record
            def _run(label, clusters_fn, tol_val):
                t0 = time.perf_counter()
                clusters = clusters_fn()
                elapsed = time.perf_counter() - t0
                est = {c.eigenvalue: c.multiplicity for c in clusters}
                n_correct, tot_err, n_found, details = score_multiplicities(est, dict_multiplicities, unique_evals)
                
                print(f"\t\t{label}  "
                      f"\t\tcorrect={n_correct}/{n_evals}  "
                      f"total_err={tot_err}  "
                      f"found={n_found}/{n_evals}  "
                      f"({elapsed:.2f}s)")
                
                print(f"\t\t\t{est}")
                
                records.append({
                    "hamiltonian": args.H,
                    "L": L, "R": R,
                    "T_max": T_max,
                    "perturbation": perturb,
                    "method_label": label.strip(),
                    "tolerance": tol_val,
                    "n_correct": n_correct,
                    "total_error": tot_err,
                    "n_found": n_found,
                    "n_evals": n_evals,
                    "result": str(est)
                })

            # --- original QFAMES (hard tau) ---
            for val in adaptive_tolerance_1:
                _run(
                    f"original tau={val:g}\n",
                    lambda v=val: QFAMES_WITH_PLOTS(
                        Z, dx, t_list, min(L, R), T_max_alg, v, alpha,
                        dict_multiplicities, f"{path}_EXP1.png", False,
                        plotting=args.plotting),
                    val)

            # --- naive adaptive ---
            for val in adaptive_tolerance_2:
                _run(
                    f"naive  adaptive tol={val}\n",
                    lambda v=val: naive_adaptive_QFAMES(
                        Z, dx, t_list, min(L, R), T_max_alg, alpha, v,
                        dict_multiplicities),
                    val)

            # --- robust adaptive ---
            for val in adaptive_tolerance_3:
                _run(
                    f"robust adaptive floor={val}\n",
                    lambda v=val: robust_adaptive_QFAMES(
                        Z, dx, t_list, min(L, R), T_max_alg, alpha, v,
                        dict_multiplicities),
                    val)


    df = pd.DataFrame(records)
    csv_path = f"data/dataframes/{name}_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    _make_summary_figure(df, name, min(L, R))

if __name__ == "__main__":
    main()