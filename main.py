import os
import time
os.system("clear")

import numpy as np
import pandas as pd

import algorithms
import aux_functions

############################
## Initialize Hamiltonian ##
############################
# Name, initialize, and normalize hamiltonian
name = "belldiagonal"
Ham = np.array([
    [-2, 0, 0, -1],
    [0, 3, -1, 0],
    [0, -1, 3, 0],
    [-1, 0, 0, -2]
])
M = (np.pi/(4*np.linalg.norm(Ham))) * Ham

# Obtain information about matrix
is_unitary = aux_functions.is_matrix_unitary(M)
num_eigenvalues = len(M)
eigenvalues, eigenvectors = np.linalg.eig(M)

# Print Information about my matrix
print(f"My Matrix: \n{M}\n")
print(f"is_unitary: {is_unitary}")
print(f"Norm of my matrix: {np.linalg.norm(M)}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors: \n{eigenvectors}\n")

# Choose Init State: Targeted eigenvalue for QPE, KQPE, and QMEGS
lambda_i = 0 # This is the index of the INIT state, # If 1 then QPE and KQPE need a scaling factor greater than ||M||
eigenvalue = abs(np.real(eigenvalues[lambda_i]))

#########################
## Set Test Parameters ##
#########################
# Choose which algorithms to test
# Options are "QPE", "KQPE", "QMEGS", "QFAMES"
# Having QFAMES on also tests TSRHSE
algorithms_array = ["QPE", "KQPE", "QMEGS", "QFAMES"] # "TSRHSE"

# Verbosity parameters
verbosity = 0
verbose = True if verbosity > 0 else False

# Number of perturbations and their strength
perturbation_params = { # length 3
    "None": {"range": None, "scale": None},
    # "Small": {"range": 1, "scale": 0.5},
    # "Big": {"range": 3, "scale": 1},
}
num_perturbations = len(perturbation_params)

# Test configurations: iterate through eps_array and T_max_array separately
# eps_array = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
T_max_array = np.array([300, 400, 500, 750, 1000]) #
test_configs = {
    # "eps": {"array": eps_array, "name": "eps"},
    "T_max": {"array": T_max_array, "name": "T_max"}
}

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
    "t or N": None,
    "T_max": None,
    "T_total": None,
    "eval(s)": str(np.real(eigenvalues))
}
results_df = pd.concat([results_df, pd.DataFrame([true_evals_row])], ignore_index=True)

# Iterate through the two different tests being performed: scaling epsilon and T_max
for test_type, config in test_configs.items():
    param_array = config["array"]
    print(f"test_type = {test_type}, param_array = {param_array}")

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
            Init = eigenvectors[:, lambda_i]
            U_list = eigenvectors

            # Obtain current test's perturbation 
            params = perturbation_params[perturb]
            perturb_range = params["range"] # Ontain current perturbation's range
            perturb_scale = params["scale"] # Ontain current perturbation's scale

            # Apply perturbation if needed
            if perturb != "None":
                # Create the new Init state and U_list unitaries
                PHI = eigenvectors + np.random.uniform(-perturb_range, perturb_range) * perturb_scale
                U_list = PHI
                Init = PHI[:, lambda_i]

            # Print information about current test
            print(f"Test: {test_type} = {param_value}, perturb = {perturb} [range = {perturb_range}, scale = {perturb_scale}]")

            # Perform the above test for each selected algorithm
            for alg in algorithms_array:
                function = getattr(algorithms, alg)
                multiplicities = None


                if alg == "QPE" or alg == "KQPE":
                    start_time = time.perf_counter()
                    _, phases, my_eigenvalue, _, T_max_alg, T_total, torN = function(M, Init=Init, eps=eps, T_max=T_max, is_unitary=is_unitary, verbosity=verbosity)
                    end_time = time.perf_counter()
                elif alg == "QMEGS":
                    aux_function = getattr(aux_functions, alg+"_setup")

                    data_start_time = time.perf_counter()
                    Z, dx, t_list, K, T_max_alg, T_total, torN = aux_function(M, Init, eps=eps, T_max=T_max, is_unitary=is_unitary)
                    data_end_time = time.perf_counter()
                    print(f"\tfinished Z array creation in {data_end_time - data_start_time:.6f} seconds")


                    start_time = time.perf_counter()
                    output_energy = function(Z, dx, t_list, K, T_max_alg)
                    end_time = time.perf_counter()

                    my_eigenvalue = str(np.real(output_energy))
                else:
                    aux_function = getattr(aux_functions, alg+"_setup")

                    data_start_time = time.perf_counter()
                    Z_qfames, Z_tsrhse, dx, tau, t_list, K, T_max_alg, T_total, torN = aux_function(M, U_list, U_list, eps=eps, T_max=T_max, is_unitary=is_unitary, verbose=verbose)
                    data_end_time = time.perf_counter()
                    print(f"\tfinished Z tensor creation in {data_end_time - data_start_time:.6f} seconds")

                    start_time = time.perf_counter()
                    output_energy = algorithms.TSRHSE(Z_tsrhse, t_list)
                    end_time = time.perf_counter()
                    my_eigenvalue = str(np.real(output_energy))
                    print(f"\tfinished TSRHSE in {end_time - start_time:.6f} seconds")

                    results_row = {
                        "test_type": test_type,
                        "param_value": param_value,
                        "perturb": perturb,
                        "algorithm": "TSRHSE",
                        "t or N": torN,
                        "T_max": T_max_alg,
                        "T_total": T_total,
                        "eval(s)": my_eigenvalue,
                        "multiplicities": multiplicities
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
                    results_df.to_csv(f"data/dataframes/{name}.csv", index=False)

                    start_time = time.perf_counter()
                    try:
                        if verbose:
                            output_energy, output_num, output_vector = algorithms.QFAMES(Z_qfames, dx, t_list, K, T_max_alg, tau, verbose=verbose)
                        else:
                            output_energy, output_num = algorithms.QFAMES(Z_qfames, dx, t_list, K, T_max_alg, tau, verbose=verbose)

                        multiplicities = str(output_num)
                    except Exception as e:
                        print(f"\tQFAMES failed: {e}")
                    end_time = time.perf_counter()

                    my_eigenvalue = str(np.real(output_energy))

                print(f"\tfinished {alg} in {end_time - start_time:.6f} seconds")
                results_row = {
                    "test_type": test_type,
                    "param_value": param_value,
                    "perturb": perturb,
                    "algorithm": alg,
                    "t or N": torN,
                    "T_max": T_max_alg,
                    "T_total": T_total,
                    "eval(s)": my_eigenvalue,
                    "multiplicities": multiplicities
                }
                results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
                results_df.to_csv(f"data/dataframes/{name}.csv", index=False)