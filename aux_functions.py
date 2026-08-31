import time
import algorithms

import qiskit as qk
import numpy as np
import scipy as sp

def is_matrix_unitary(matrix):
    # Ensure it's a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Calculate conjugate transpose (Hermitian adjoint)
    hermitian_adjoint = matrix.conj().T
    
    # Multiply matrix by its conjugate transpose
    product = np.dot(matrix, hermitian_adjoint)

    # Create an identity matrix of the same size
    identity = np.eye(matrix.shape[0])
    
    # Check if product is almost equal to identity
    return np.allclose(product, identity)

def generate_t_list(N, T_max, sigma, seed=777):
    """ Generate time samples from truncated Gaussian
    Parameters:
        N : number of samples
        T : variance of Gaussian
        sigma : truncated parameter

    Returns:
        t_list: np.array of time points
    """

    np.random.seed(seed)
    t_list = sp.stats.truncnorm.rvs(-sigma, sigma, loc=0, scale=T_max, size=N)
    return t_list

def generate_Z_array(M, N, t_list, Init, is_unitary=True, shots=1):
    """
    Generates a complex-valued array Z(t) using the Hadamard Test:
        Z(t) = ⟨φ| U(t) |φ⟩ = Re + i Im

    Each entry is estimated via two Hadamard Tests:
        - one for the real part
        - one for the imaginary part

    Parameters:
        M           : Input matrix (unitary or Hermitian)
        N           : Number of samples (not directly used; inferred from t_list)
        t_list      : List of time points
        Init        : State preparation (|φ⟩)
        is_unitary  : Whether M is unitary or Hermitian
        shots       : Number of samples per Hadamard Test

    Returns:
        Z_array : List of complex expectation values Z(t)
    """

    Z_array = np.zeros((N), dtype=complex)
    for n, t in enumerate(t_list):
        # Real part estimation
        qc = algorithms.HadamardTest(M, Init, t, is_unitary=is_unitary)
        state = qk.quantum_info.Statevector.from_instruction(qc)
        counts_real = state.sample_counts(shots, [0])

        # Convert probability to expectation value
        prob_0 = counts_real.get("0", 0) / shots
        real = 2*prob_0 - 1

        # Imaginary part estimation
        qc = algorithms.HadamardTest(M, Init, t=t, img=True, is_unitary=is_unitary)
        state = qk.quantum_info.Statevector.from_instruction(qc)
        counts_img = state.sample_counts(shots, [0])

        prob_0 = counts_img.get("0", 0) / shots
        img = 2*prob_0 - 1

        # Combine into complex expectation value
        Z = real + 1j*img
        Z_array[n] = Z
        
    return np.array(Z_array)

def generate_Z_tensor(M, N, U_list, V_list, L, R, t_list, is_unitary=True, shots=1, verbose=True):
    Z_tensor = np.zeros((L, R, N), dtype=complex)
    
    if verbose: print("Beginning to fill Z tensor")
    Zstart = time.perf_counter()
    for l in range(L):
        for r in range(R):
            start = time.perf_counter()
            for n, t in enumerate(t_list):
                # Real part: W = Identity  (img=False)
                qc_re = algorithms.GeneralizedHadamardTest(M, U_list[l], V_list[r], t, img=False, is_unitary=is_unitary)

                state = qk.quantum_info.Statevector.from_instruction(qc_re)
                counts_real = state.sample_counts(shots, [0])

                prob_0 = counts_real.get("0", 0) / shots
                real = 2*prob_0 - 1


                # Imaginary part: W = S†   (img=True)
                qc_im = algorithms.GeneralizedHadamardTest(M, U_list[l], V_list[r], t, img=True, is_unitary=is_unitary)

                state = qk.quantum_info.Statevector.from_instruction(qc_im)
                counts_real = state.sample_counts(shots, [0])

                prob_0 = counts_real.get("0", 0) / shots
                img = 2*prob_0 - 1

                Z_tensor[l, r, n] = real + 1j*img
            end = time.perf_counter()
            if verbose: print(f"\tfilling my Z[{l},{r}, :] took {end - start:.6f} seconds", flush=True)
    Zend = time.perf_counter()
    if verbose: print(f"Total time to generate Z was {Zend - Zstart:.6f} seconds")
    return Z_tensor

def generate_multiple_Z_tensors(M, N, U_list, V_list, L, R, t_list, is_unitary=True, shots=750, verbose=False):
    Z_tensor_single = np.zeros((L, R, N), dtype=complex)
    Z_tensor_poly = np.zeros((L, R, N), dtype=complex)

    
    if verbose: print("Beginning to fill Z tensor")
    Zstart = time.perf_counter()
    for l in range(L):
        for r in range(R):
            start = time.perf_counter()
            for n, t in enumerate(t_list):
                # Real part: W = Identity  (img=False)
                qc_re = algorithms.GeneralizedHadamardTest(M, U_list[l], V_list[r], t, img=False, is_unitary=is_unitary)

                state = qk.quantum_info.Statevector.from_instruction(qc_re)
                counts_real_single = state.sample_counts(1, [0])
                counts_real_poly = state.sample_counts(shots, [0])

                prob_0_single = counts_real_single.get("0", 0)
                prob_0_poly = counts_real_poly.get("0", 0) / shots

                real_single = 2*prob_0_single - 1
                real_poly = 2*prob_0_poly - 1


                # Imaginary part: W = S†   (img=True)
                qc_im = algorithms.GeneralizedHadamardTest(M, U_list[l], V_list[r], t, img=True, is_unitary=is_unitary)

                state = qk.quantum_info.Statevector.from_instruction(qc_im)
                counts_img_single = state.sample_counts(1, [0])
                counts_img_poly = state.sample_counts(shots, [0])

                prob_0_single = counts_img_single.get("0", 0)
                prob_0_poly = counts_img_poly.get("0", 0) / shots
                img_single = 2*prob_0_single - 1
                img_poly = 2*prob_0_poly - 1

                Z_tensor_single[l, r, n] = real_single + 1j*img_single
                Z_tensor_poly[l, r, n] = real_poly + 1j*img_poly
            end = time.perf_counter()
            if verbose: print(f"\tfilling my Z[{l},{r}, :] took {end - start:.6f} seconds", flush=True)
    Zend = time.perf_counter()
    if verbose: print(f"Total time to generate Z was {Zend - Zstart:.6f} seconds")
    return Z_tensor_single, Z_tensor_poly

def QMEGS_setup(M, Init, eps=None, T_max=None, is_unitary=True, sigma=1, p_min=0.5, p_tail=0.0, D=2, eta=0.1, q=0.05, delta_dom=None):
    """
    Compute QMEGS parameters from error tolerance eps.
    
    p_min     : min overlap of initial state with dominant eigenstates (default 0.5)
    p_tail    : total overlap with non-dominant eigenstates (default 0.0)
    D         : number of dominant eigenvalues to find
    eta       : failure probability
    delta_dom : spectral gap (if known, uses Thm 3.2; else uses Thm 3.1)
    """

    K = len(M)
    gap = p_min - p_tail  # must be > 0

    if T_max is None:
        if delta_dom is None:
            # No-gap regime (Theorem 3.1): T_max = 1/eps, independent of gap
            T_max = int(1.0 / eps)
        else:
            # Gapped regime (Theorem 3.2): shorter T_max but requires knowing gap
            T_max = p_tail / (p_min * eps) if p_tail > 0 else 1.0/eps
            T_max = int(max(T_max, 1.0/delta_dom))  # enforce T >> Delta_dom^-1

    # Set alpha, q, and dx (Theorem 3.1 guidance)
    dx = q / T_max  # grid step, q << alpha, and grid step q/T is the resolution

    # N formula from Theorem 3.1:  N = Omega(1/gap^2 * log((T/q + D)/eta))
    log_factor = np.log2((T_max / q + D) / eta)
    N = int(np.ceil((1.0 / gap**2) * log_factor))
    # N = min(N, 100)  # practical minimum

    t_list = generate_t_list(N, T_max, sigma)
    T_total = sum(np.abs(t_list))

    Z = generate_Z_array(M, N, t_list, Init, is_unitary=is_unitary, shots=1)

    return Z, dx, t_list, K, T_max, T_total, N

def QFAMES_setup(M, U_list, V_list, eps=None, T_max=None, is_unitary=True, shots=1, sigma=1, p_min=0.5, p_tail=0.0, q=0.05, delta=None, verbose=0):
    K = len(M) + 1
    L = R = len(M)

    # T from eps: epsilon = O(p_tail/p_min * 1/T) => T = p_tail/(p_min * eps)
    # If p_tail is unknown/zero, fall back to T = 1/eps
    if p_tail > 0:
        T_max = p_tail / (p_min * eps)
    else:
        if eps is not None:
            T_max = int(1.0 / eps)
        else:
            if T_max == None:
                print("error")
                
    if delta is not None:
        T_max = int(max(T_max, 1.0/delta))  # must resolve gap
    
    # N = Omega(L*R*p_tail^-2) — eq. 25, but if p_tail~0 use practical bound
    N = 0
    tau = 0 # SVD threshold
    if p_tail > 0:
        N = np.ceil(L * R / p_tail**2)
        tau = p_tail
    else:
        N = np.ceil(L * R * 100)  # practical default
        tau = 0.1 * np.sqrt(L * R)

    dx = q / T_max  # grid step, q << alpha, and grid step q/T is the resolution

    t_list = generate_t_list(N, T_max, sigma)
    T_total = L*R*sum(np.abs(t_list))

    return dx, tau, t_list, K, T_max, T_total, N