import numpy as np
import qiskit as qk
import scipy as sp

def normalize_init_operator(Init, label="Init", control=False, ctrl_state="0"):
    """
    Converts various initialization objects into a Gate.

    Supported inputs:
        - numpy array representing a statevector
        - numpy array representing a unitary matrix
        - StatePreparation
        - Gate
        - Instruction
        - QuantumCircuit

    Returns:
        Gate
        Gontrolled Gate if control = True
    """
    gate = None
    # Case 1: numpy array
    if isinstance(Init, np.ndarray):
        # Statevector
        if Init.ndim == 1:
            Init = Init / np.linalg.norm(Init)
            Init = qk.circuit.library.StatePreparation(Init, label=label)

        # Unitary matrix
        elif Init.ndim == 2:
            Init = qk.circuit.library.UnitaryGate(Init, label=label)
            if control == True:
                Init = Init.control(1, ctrl_state=ctrl_state)
            return Init

        else:
            raise ValueError("Invalid ndarray shape for Init.")

    # Case 2: StatePreparation
    if isinstance(Init, qk.circuit.library.StatePreparation) or isinstance(Init, qk.circuit.library.InstructionSet):
        qc = qk.QuantumCircuit(Init.num_qubits)
        qc.append(Init, range(Init.num_qubits))
        Init = qc

    # Case 3: QuantumCircuit
    if isinstance(Init, qk.QuantumCircuit) or isinstance(Init, qk.circuit.Instruction):
        Init = Init.to_gate(label=label)

    # Case 4: Gate
    if isinstance(Init, qk.circuit.Gate):
        Init.name = label
        if control == True:
            Init = Init.control(1, ctrl_state=ctrl_state)

    return Init

def QPE(M, Init=None, eps=None, t=None, T_max=None, T_total=None, shots=1000, scaling=0.0, is_unitary=True, verbosity=1, measure=True):
    """
    Standard Quantum Phase Estimation (QPE).

    Estimates the phase φ such that:
        U |ψ⟩ = e^{2πiφ} |ψ⟩

    If M is unitary:
        U = M

    If M is Hermitian (Hamiltonian):
        U = exp(-i H t0), where t0 = 2π / scaling
        In this case, the returned phases are scaled to approximate eigenvalues.

    Parameters:
        M           : Input matrix (unitary or Hermitian)
        eps         : Desired precision (used to determine number of control qubits and therefore precision)
        t           : Number of control qubits (overrides eps if provided)
        T_max       : Maximum time evolution (is overriden by eps or t)
        T_total     : Total time evolution (is overriden by eps, t, or T_max)
        Init        : StatePreparation gate for initializing eigenvector
        shots       : Number of samples used for estimating phase distribution
        scaling     : Optional scaling factor (||H||₂ if not provided)
        is_unitary  : Whether M is unitary or Hermitian
        verbose     : Print statistics and circuit
        measure     : Whether to include measurement gates in returned circuit

    Returns:
        qc      : QuantumCircuit implementing QPE
        phases  : List of estimated phases (or eigenvalues if Hermitian)
        counts  : Sampled bitstring counts
    """

    # Ensure either precision or number of qubits is specified
    if t is None and eps is None and T_max is None and T_total is None:
        raise ValueError("One of these must be specified: eps, t, T_max, T_total")
    
    # Number of target qubits (log₂ of matrix dimension)
    n = int(np.log2(len(M)))

    # Time scaling parameter (only used for Hermitian case)
    t0 = 0.0
    if not is_unitary:
        # Use spectral norm (largest eigenvalue magnitude) if not provided
        if scaling == 0: 
            scaling = np.linalg.norm(M, ord=2)
        # Choose t0 to avoid phase wrap-around
        t0 = 2 * np.pi / scaling

    # Determine number of control qubits based on desired precision
    if t is None:
        if eps is not None:
            m = np.ceil(n/2) + 1
            t = int(m + np.ceil(np.log2(2+1/(2*eps))))
        elif T_max is not None:
            t = int(np.floor(np.log2(T_max / t0)) + 1)
        elif T_total is not None:
            t = int(np.floor(np.log2(T_total / t0)))


    # Define quantum and classical registers
    control_reg = qk.QuantumRegister(t, name="ctrl")
    target_reg = qk.QuantumRegister(n, name="tgt")
    meas_reg = qk.ClassicalRegister(t, name="meas")

    regs = []
    regs.append(control_reg)
    regs.append(target_reg)
    
    # Create the quantum circuit
    qc = qk.QuantumCircuit(*regs, meas_reg)
    qc.name = "QPE"

    # Apply Hadamards to create superposition over control register
    qc.h(control_reg)

    # Initialize target register to chosen eigenvector (if provided)
    if Init is not None:
        Init = normalize_init_operator(Init, label="Init")
        qc.append(Init, target_reg)

    qc.barrier()

    T_max = t0 * 2**(t-1)
    T_total = 0
    # Apply controlled-U^(2^k) operations
    for k in range(t):
        T_total += t0 * 2**k

        Uk_gate = qk.QuantumCircuit(n)

        if is_unitary:
            # Compute U^(2^k)
            Uk = np.linalg.matrix_power(M, 2**k)
            Uk_gate.name=f"U^{2**k}"
        else:
            # Compute exp(-i H t0 2^k)
            Uk = sp.linalg.expm(-1j * M * t0 * (2**k))
            Uk_gate.name=f"e^({2**k}iH)"

        # Convert to unitary gate and control it
        Uk_gate.unitary(Uk, qubits=range(n))
        CUk_gate = Uk_gate.to_gate().control(1)
        qc.append(CUk_gate, [control_reg[k]] + list(target_reg))

    # Apply inverse Quantum Fourier Transform
    iQFT = qk.circuit.library.QFT(t,inverse=True,do_swaps=True).to_gate(label="iQFT")
    qc.append(iQFT, control_reg[:]) 

    # Simulate circuit using statevector and sample only control qubits
    state = qk.quantum_info.Statevector.from_instruction(qc)
    counts = state.sample_counts(shots, qargs=range(t))

    # Convert measured bitstrings into phases
    phases = []
    results = [item for item, count in counts.items() for _ in range(count)]
    for bit_string in results:
        t = len(bit_string)
        k = int(bit_string, 2)
        phase = k / (2**t)
        phases.append(phase)

    # Convert phase → eigenvalue for Hermitian case
    if not is_unitary:
        phases = np.array(phases)
        phases = phases * scaling
        # Or equivalent form:
        # phases = (2 * pi * phases) / t0
    
    eigenvalue = np.mean(phases)

    # Optionally add measurement gates to circuit
    if measure: qc.measure(control_reg, meas_reg)

    # Print summary statistics
    if verbosity > 0:
        if verbosity > 1: print(qc.draw())
        print(f"QPE Summary:")
        print(f"\teps = {eps}, t = {t}, T_max = {T_max}, T_total = {T_total}")
        print(f"\tphases mean = {np.mean(phases)}")
        print(f"\tphases mode = {sp.stats.mode(phases).mode}")
        print(f"\tphases median = {np.median(phases)}")
        # print(f"phase list = \n{np.array(phases)}")


    return qc, phases, eigenvalue, counts, T_max, T_total, t

def KQPE(M, Init=None, eps=None, t=None, T_max=None, T_total=None, shots=1000, scaling=0.0, is_unitary=True, verbosity=1):
    """
    Kitaev's Iterative Quantum Phase Estimation (KQPE).

    Estimates phase φ by determining bits one at a time (most significant → least).
    Avoids inverse QFT by using adaptive phase corrections.

    If M is Hermitian:
        U = exp(-i H t0), and final phase is scaled to recover eigenvalue.

    Returns:
        phase : Estimated phase (or eigenvalue if Hermitian)
        bits  : Binary representation of phase
    """

    # Ensure either precision or number of qubits is specified
    if t is None and eps is None and T_max is None and T_total is None:
        raise ValueError("One of these must be specified: eps, t, T_max, T_total")

    # Number of target qubits
    n = int(np.log2(len(M)))

    # Scaling for Hermitian matrices
    t0 = 0.0
    if not is_unitary:
        if scaling == 0: 
            scaling = np.linalg.norm(M, ord=2)
        t0 = 2 * np.pi / scaling

    # Determine number of bits of precision
    if t is None: 
        if eps is not None:
            t = np.ceil(np.log2(1/eps)) + 2
        elif T_max is not None:
            t = np.floor(np.log2(T_max / t0)) + 1
        elif T_total is not None:
            t = np.floor(np.log2(T_total / t0))
    t = int(t)

    # Storage for estimated bits
    bits = [0]*t

    # Increase shots for statistical confidence
    shots = shots*t

    # Prepare initialization gate
    if Init is not None:
        Init = normalize_init_operator(Init, label="Init")

    T_max = t0*2**(t-1)
    T_total = 0

    qcs = np.empty(t, dtype=qk.circuit.quantumcircuit.QuantumCircuit)

    # Iterate from most significant bit to least
    for k in reversed(range(t)):
        T_total += t0 * 2**k

        # Compute phase correction using previously determined bits
        phase_correction = 0.0
        for j in range(k+1, t):
            phase_correction += bits[j] / (2 ** (j - k + 1))

        # Build circuit for this iteration
        qc = KQPE_circuit(M, k, n, t0, phase_correction, Init=Init, is_unitary=is_unitary, measure=False)
        qcs[k] = qc

        if verbosity > 1: print(qc.draw())

        # Simulate and measure control qubit
        state = qk.quantum_info.Statevector.from_instruction(qc)
        counts = state.sample_counts(shots, qargs=[0])

        # Determine bit based on measurement probability
        prob_0 = counts.get("0", 0) / shots
        bits[k] = 0 if prob_0 > 0.5 else 1
    
    # Convert binary fraction to phase
    phase = sum(bits[i] * 2**(-(i+1)) for i in range(t))

    # Convert phase → eigenvalue for Hermitian case
    if not is_unitary:
        phase = phase * scaling

    eigenvalue = abs(phase)

    if verbosity > 0: 
        print(f"KQPE Summary:")
        print(f"\teps = {eps}, t = {t}, T_max = {T_max}, T_total = {T_total}")
        print(f"\tphase = {phase}")

    return qcs, phase, eigenvalue, bits, T_max, T_total, t

def KQPE_circuit(M, k, n, t0 = 0.0, prev_phase=0.0, Init=None, is_unitary=True, measure=False):
    """
    Constructs the circuit for a single iteration of Kitaev QPE.

    This circuit estimates the k-th bit of the phase by:
        1. Applying Hadamard to control qubit
        2. Applying controlled-U^(2^k)
        3. Applying phase correction from previously determined bits
        4. Applying Hadamard and measuring

    Parameters:
        k           : Bit index being estimated
        prev_phase  : Phase correction from higher-order bits
        t0          : Time scaling (for Hermitian case)
    """

    control_reg = qk.QuantumRegister(1, name="ctrl")
    target_reg = qk.QuantumRegister(n, name="tgt")
    meas_reg = qk.ClassicalRegister(1, name="meas")

    regs = []
    regs.append(control_reg)
    regs.append(target_reg)
    
    qc = qk.QuantumCircuit(*regs, meas_reg)
    qc.name = f"KQPE_{k}"

    # Prepare superposition on control qubit
    qc.h(control_reg)

    # Initialize eigenstate if provided
    if Init is not None:
        qc.append(Init, target_reg)        

    qc.barrier()

    # Build U^(2^k)
    Uk_gate = qk.QuantumCircuit(n)

    if is_unitary:
        Uk = np.linalg.matrix_power(M, 2**k)
        Uk_gate.name=f"U^{2**k}"
    else:
        Uk = sp.linalg.expm(-1j * M * t0 * (2**k))
        Uk_gate.name=f"e^({2**k}iH)"

    # Apply controlled unitary
    Uk_gate.unitary(Uk, qubits=range(n))
    CUk_gate = Uk_gate.to_gate().control(1)
    qc.append(CUk_gate, list(control_reg) + list(target_reg))

    # Apply phase correction based on previously determined bits
    qc.p(-2 * np.pi * prev_phase, control_reg)

    # Interference step
    qc.h(control_reg)

    # Optional measurement
    if measure: qc.measure(control_reg, meas_reg)

    return qc

def HadamardTest(M, Init, t, img=False, is_unitary=True, measure=False):
    """
    Constructs a Hadamard Test circuit to estimate expectation values of the form ⟨φ| U |φ⟩ where |φ⟩ is the state prepared by Init

    U is either:
        - M^(2^t) if M is unitary
        - exp(-i M t) if M is Hermitian

    The Hadamard Test allows estimation of either:
        ⟨Z⟩ = Re(⟨φ| U |φ⟩)   if img == False
        ⟨Z⟩ = Im(⟨φ| U |φ⟩)   if img == True

    Measurement statistics:
        P(0) = (1 + ⟨Z⟩)/2
        P(1) = (1 - ⟨Z⟩)/2

    Parameters:
        M           : Input matrix (unitary or Hermitian)
        Init        : State vector (or unitary) preparing |φ⟩
        t           : Time / exponent parameter
        img         : If True, estimates imaginary part; else real part
        is_unitary  : Whether M is unitary or Hermitian
        measure     : Whether to include measurement in circuit

    Returns:
        qc : QuantumCircuit implementing the Hadamard Test
    """

    # Number of target qubits
    n = int(np.log2(len(M)))

    # Define registers
    control_reg = qk.QuantumRegister(1, name="ctrl")
    target_reg = qk.QuantumRegister(n, name="tgt")
    meas_reg = qk.ClassicalRegister(1, name="meas")

    regs = []
    regs.append(control_reg)
    regs.append(target_reg)
    
    qc = qk.QuantumCircuit(*regs, meas_reg)
    qc.name = "HadamardTest"

    # Initialize target register to |φ⟩
    Init = normalize_init_operator(Init, label="Init")
    qc.append(Init, target_reg)

    # Prepare control qubit in superposition
    qc.h(control_reg)

    qc.barrier()

    # Build unitary U
    U_gate = qk.QuantumCircuit(n)

    if is_unitary:
        # Compute U^(2^t)
        # Note: t is cast to integer exponent
        # TODO: Figure out what happens when we cast into a positive integer
        # print(f"[DEBUG] t = {t}, int(t) = {int(t)}, abs(int(t)) = {abs(int(t))}, 2^ = {2**abs(int(t))}")
        U = np.linalg.matrix_power(M, 2**abs(int(t)))
        U_gate.name=f"U^{2**int(t)}"
    else:
        # Compute time evolution operator exp(-i H t)
        U = sp.linalg.expm(-1j * M * t)
        U_gate.name=f"e^(-{t}iH)"

    # Apply controlled-U
    U_gate.unitary(U, qubits=range(n))
    CU_gate = U_gate.to_gate().control(1)
    qc.append(CU_gate, list(control_reg) + list(target_reg))
    
    # If estimating imaginary part, apply S† before final Hadamard
    if img: qc.sdg(control_reg)

    # Interference step
    qc.h(control_reg)

    # Optional measurement (note: meas_reg must exist externally if used)
    if measure: qc.measure(control_reg, meas_reg)

    return qc

def GeneralizedHadamardTest(M, U, V, t, img=False, is_unitary=True, measure=False):
    """
    Constructs a Generalized Hadamard Test circuit to estimate expectation values of the form:

        ⟨φ| U† V |ψ⟩

    where:
        |φ⟩ is the state prepared by U
        |ψ⟩ is the state prepared by V

    -------------------------------------------------------------------------
    Difference from the standard Hadamard Test:

    - Standard Hadamard Test estimates:
          ⟨φ| W |φ⟩
      for a single unitary W.

    - Generalized Hadamard Test estimates:
          ⟨φ| U† V |ψ⟩
      by coherently applying:
          U when control = |0⟩
          V when control = |1⟩

      This allows estimation of overlaps, inner products, and transition amplitudes
      between two different unitary evolutions, rather than a single operator.

    -------------------------------------------------------------------------
    Measurement interpretation:

        P(0) = (1 + ⟨Z⟩)/2
        P(1) = (1 - ⟨Z⟩)/2

    where:
        ⟨Z⟩ = Re(⟨φ| U† V |φ⟩)   if img == False
        ⟨Z⟩ = Im(⟨φ| U† V |φ⟩)   if img == True

    Therefore:
        ⟨Z⟩ = 2 * P(0) - 1

    -------------------------------------------------------------------------
    Additional unitary evolution (optional):

    After applying U and V, the circuit applies an additional controlled unitary:

        - M^(2^t)           if is_unitary == True
        - exp(-i M t)       if is_unitary == False

    This allows combining overlap estimation with phase evolution, which is useful
    in algorithms such as Post-Kitaev QPE and signal-based spectral estimation.

    -------------------------------------------------------------------------
    Parameters:
        M           : Matrix defining additional unitary evolution
        U, V        : Unitary operators used in generalized Hadamard test
        Init        : StatePreparation gate for preparing |φ⟩
        t           : Time / exponent parameter
        img         : If True, estimates imaginary part; else real part
        is_unitary  : Whether M is unitary or Hermitian
        measure     : Whether to include measurement in circuit

    Returns:
        qc : QuantumCircuit implementing the generalized Hadamard test
    """

    # Number of target qubits
    n = int(np.log2(len(M)))

    # Define registers
    control_reg = qk.QuantumRegister(1, name="ctrl")
    target_reg = qk.QuantumRegister(n, name="tgt")
    meas_reg = qk.ClassicalRegister(1, name="meas")

    regs = []
    regs.append(control_reg)
    regs.append(target_reg)
    
    qc = qk.QuantumCircuit(*regs, meas_reg)
    qc.name = "GeneralizedHadamardTest"

    # # Initialize target register to |φ⟩
    # qc_temp = QuantumCircuit(target_reg)
    # qc_temp.append(Init, target_reg)
    # Init_gate = qc_temp.to_gate(label="Init")
    # qc.append(Init_gate, target_reg)

    # Prepare control qubit in superposition
    qc.h(control_reg)

    # Build unitary U and V
    U_gate = normalize_init_operator(U, label="U", control=True, ctrl_state="0")
    qc.append(U_gate, list(control_reg) + list(target_reg))
    V_gate = normalize_init_operator(V, label="V", control=True, ctrl_state="1")
    qc.append(V_gate, list(control_reg) + list(target_reg))
    
    # U_gate = UnitaryGate(U)
    # V_gate = UnitaryGate(V)
    # U_gate.name = "U"
    # V_gate.name = "V"
    # U_gate = U_gate.control(1, ctrl_state="0")
    # V_gate = V_gate.control(1, ctrl_state="1")
    # qc.append(U_gate, list(control_reg) + list(target_reg))
    # qc.append(V_gate, list(control_reg) + list(target_reg))

    qc.barrier()

    # Build unitary U
    Uni_gate = qk.QuantumCircuit(n)
    Uni = np.zeros((n, n))

    if is_unitary:
        # Compute U^(2^t)
        # Note: t is cast to integer exponent
        # TODO: Figure out what happens when we cast into a positive integer
        # print(f"[DEBUG] t = {t}, int(t) = {int(t)}, abs(int(t)) = {abs(int(t))}, 2^ = {2**abs(int(t))}")
        Uni = np.linalg.matrix_power(M, 2**abs(int(t)))
        Uni_gate.name=f"U^{2**int(t)}"
    else:
        # Compute time evolution operator exp(-i H t)
        Uni = sp.linalg.expm(-1j * M * t)
        Uni_gate.name=f"e^(-{t}iH)"

    # Apply controlled-U
    Uni_gate.unitary(Uni, qubits=range(n))
    CUni_gate = Uni_gate.to_gate().control()
    qc.append(CUni_gate, list(control_reg) + list(target_reg))
    
    # If estimating imaginary part, apply S† before final Hadamard
    if img: qc.sdg(control_reg)

    # Interference step
    qc.h(control_reg)

    # Optional measurement (note: meas_reg must exist externally if used)
    if measure: qc.measure(control_reg, meas_reg)

    return qc

def QMEGS(Z, dx, t_list, K, T_max, alpha=5):
    """
    QMEGS new algorithm
    
    Note: This code is slightly different from the algorithm in the paper. 
    
    To avoid long classical running time, we first do a rough search 
    then do a detailed search around the rough maximal point.
    """
    num_x=int(2*np.pi/(dx*10))
    num_x_detail=int(2*alpha/dx/T_max)
    x_rough=np.arange(0,num_x)*dx*10-np.pi
    G=np.abs(Z.dot(np.exp(1j*np.outer(t_list,x_rough)))/len(Z)) #Gaussian filter function
    Dominant_freq=np.zeros(K,dtype='float')
    for k in range(K):
        max_idx_rough = np.argmax(G)
        Dominant_potential=x_rough[max_idx_rough]
        x=np.arange(0,num_x_detail)*dx+Dominant_potential-alpha/T_max
        G_detail=np.abs(Z.dot(np.exp(1j*np.outer(t_list,x)))/len(Z))
        max_idx_detail = np.argmax(G_detail)
        Dominant_freq[k]=x[max_idx_detail]
        interval_max=x[max_idx_detail]+alpha/T_max
        interval_min=x[max_idx_detail]-alpha/T_max
        G=np.multiply(G,x_rough>interval_max)+np.multiply(G,x_rough<interval_min) #eliminate interval
    return Dominant_freq

def QFAMES(Z, dx, t_list, K, T, tau, alpha=5, verbose=False):
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
    for k in range(K):
        max_idx_rough = np.argmax(G) # Rough search
        # print('maxG',max(G))
        if G[max_idx_rough] < 1e-6:
            if verbose: print('No more clusters found, exiting early.')
            break
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
        V = V.conj().T
        # print('k=',k,'Dominant_freq=',Dominant_freq[k])
        # print(G_detail[max_idx_detail])
        # print(k,S)
        Dominant_num[k] = int(np.sum(S > tau))
        if Dominant_num[k]>0: #if there is a cluster, calculate the orthogonal vector space
        #    print('k=',k,'S=',S)
            # Calculate the orthogonal vector space of the cluster
           if verbose:
                if L>R:
                    Dominant_vector.append(U[:,Dominant_num[k]:])
                else:
                    Dominant_vector.append(V[:,Dominant_num[k]:])
        interval_max=x[max_idx_detail]+alpha/T
        interval_min=x[max_idx_detail]-alpha/T
        # print('interval_min=',interval_min,'interval_max=',interval_max)
        G=np.multiply(G,x_rough>interval_max)+np.multiply(G,x_rough<interval_min) #eliminate interval
    if verbose:
        return Dominant_freq, Dominant_num, Dominant_vector
    else:
        return Dominant_freq, Dominant_num

def TSRHSE(Z, t_list, verbosity = 0):
    N = len(t_list)
    Q = np.shape(Z)[0]

    sorted_idx = np.argsort(t_list)
    pairs = [(int(sorted_idx[i]), int(sorted_idx[i+1])) for i in range(N-1)]
    pairs = sorted(pairs, key=lambda p: abs(t_list[p[0]] - t_list[p[1]]))

    # Scale ladder: double dt each step
    dt_min = abs(t_list[pairs[0][0]] - t_list[pairs[0][1]])
    dt_max = abs(t_list[pairs[-1][0]] - t_list[pairs[-1][1]])
    if verbosity > 0: print(f"dt_min = {dt_min}, dt_max = {dt_max}")

    scales = []
    dt = max(dt_min, 0.05)
    while dt <= dt_max * 1.01:
        scales.append(dt)
        dt *= 2.0

    current_est = None
    for target_dt in scales:
        if verbosity > 1: print(f"target_dt = {target_dt}")
        candidates = sorted(pairs, key = lambda p: abs(abs(t_list[p[0]] - t_list[p[1]]) - target_dt))[:15] #TODO: if t_list[p[0]] - t_list[p[1]] is greater than target_dt it is not a candidate

        estimates = []
        for a, b in candidates:
            try:
                lambdas, alphas = Jenrich(Z, t_list, Q, a, b)
            except:
                continue
            formated_alphas = [f"{x:.3f}" for x in alphas]
            formated_alphas = np.array([complex(s) for s in formated_alphas])

            formated_lambdas = [f"{x:.3f}" for x in lambdas]
            formated_lambdas = np.array([float(s) for s in formated_lambdas])
            if verbosity > 1: print(f"\ta = {a} (t_list[a] = {t_list[a]:.3f}), b = {b} (t_list[b] = {t_list[b]:.3f}), dt = {np.abs(t_list[a] - t_list[b]):.3f}, alphas = {formated_alphas}, lambdas = {formated_lambdas}")

            if np.max(np.abs(np.abs(alphas) - 1.0)) < 0.25: # unit-circle check
                estimates.append(lambdas)

        if not estimates:
            continue

        raw = np.median(np.array(estimates), axis=0)

        if current_est is None:
            current_est = raw
        else:
            # Unwap: find the integer k that brings raw closest to current_est
            k = np.round((current_est - raw) * target_dt / (2 * np.pi))
            current_est = raw + 2 * np.pi * k / target_dt

    return current_est

def Jenrich(Z, t_list, D, a, b):
    assert (a >= 0 and b >= 0), "a and b must be nonnegative"

    dt = t_list[a] - t_list[b]

    # Obtain slices
    A = Z[:, :, a]
    B = Z[:, :, b]

    # Perform truncated svd to keep top D singular values
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    U_D = U[:, :D]
    S_D = S[:D]
    Vt_D = Vt[:D, :]

    # Compute B dagger
    B_dag = Vt_D.conj().T @ np.diag(1.0 / S_D) @ U_D.conj().T

    # Compute M
    M = A @ B_dag

    # Obtain D eigenvalues of M
    alphas = np.linalg.eigvals(M)
    alphas = sorted(alphas, key=lambda z: abs(abs(z) - 1.0))[:D]
    alphas = np.array(alphas)
    
    lambdas = -np.angle(alphas) / dt
    lambdas = np.sort(lambdas)

    return lambdas, alphas