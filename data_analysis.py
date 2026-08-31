import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("data/plots", exist_ok=True)

use_y_axis_log_scale = True  # Toggle this to switch between log and linear scale
use_x_axis_log_scale = True  # Toggle this to switch between log and linear scale

def parse_eigenvalues(value):
    if pd.isna(value) or str(value).strip() in {"", "None"}:
        return np.array([], dtype=float)

    value_text = str(value).strip()

    try:
        parsed_value = ast.literal_eval(value_text)
        return np.atleast_1d(np.asarray(parsed_value, dtype=float))
    except (SyntaxError, ValueError):
        if value_text.startswith("[") and value_text.endswith("]"):
            value_text = value_text[1:-1]
        return np.fromstring(value_text, sep=" ", dtype=float)

def calculate_error_to_target(row):
    found_eigenvalues = parse_eigenvalues(row["eval(s)"])
    if found_eigenvalues.size == 0:
        return np.nan

    if row["algorithm"] in {"QPE", "KQPE"}:
        found_eigenvalue = found_eigenvalues[0]
        if target_eigenvalue < 0:
            found_eigenvalue = -found_eigenvalue
        return abs(target_eigenvalue - found_eigenvalue)

    return np.min(np.abs(found_eigenvalues - target_eigenvalue))
    
perturbation_params = { # length 3
    "None": None,
    "Small": {"range": 1, "scale": 0.5},
    "Big": {"range": 3, "scale": 1},
}
Hamiltonian_name = "belldiagonal"
df = pd.read_csv(f"data/dataframes/{Hamiltonian_name}.csv")

target_eigenvalue_index = 0
true_eigenvalues = parse_eigenvalues(df.loc[df["test_type"] == "true eigenvalues", "eval(s)"].iloc[0])
target_eigenvalue = true_eigenvalues[target_eigenvalue_index]

df["error_to_target"] = df.apply(calculate_error_to_target, axis=1)
df.to_csv(f"data/dataframes/{Hamiltonian_name}.csv", index=False)
algo_order = df["algorithm"].unique().dropna()

for test_type in df["test_type"].unique():
    # if test_type == "true eigenvalues": continue
    if test_type == "eps":
        # 3x3 grid for eps tests
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        x_labels = ["eps", "T_max", "T_total"]
        x_cols = ["param_value", "T_max", "T_total"]
        
        for row, perturb in enumerate(perturbation_params.keys()):
            for col, (x_label, x_col) in enumerate(zip(x_labels, x_cols)):
                ax = axes[row, col]
                
                # Filter data for this perturbation (including true eigenvalues)
                # Handle both string "None" and NaN/None values for perturb column
                if perturb == "None":
                    df_filtered_with_true = df[(df["perturb"] == "None") | (pd.isna(df["perturb"]))]
                else:
                    df_filtered_with_true = df[df["perturb"] == perturb]
                
                # Filter out true eigenvalues for plotting
                df_filtered = df_filtered_with_true[df_filtered_with_true["test_type"] != "true eigenvalues"]
                
                # Plot each algorithm in specified order
                for algo in algo_order:
                    df_algo = df_filtered[df_filtered["algorithm"] == algo]
                    if df_algo.empty:
                        continue
                    df_algo = df_algo.sort_values(by=x_col)
                    ax.plot(df_algo[x_col], df_algo["error_to_target"], marker='o', label=algo, linewidth=2)
                
                ax.set_xlabel(x_label)
                ax.set_ylabel("error_to_target")
                ax.set_title(f"perturb={perturb}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                if use_y_axis_log_scale:
                    ax.set_yscale('log')
                if use_x_axis_log_scale:
                    ax.set_xscale('log')

                # Invert x-axis for first column (eps)
                if col == 0:
                    ax.invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(f"data/plots/{Hamiltonian_name}_eps_plots.png", dpi=150)
        plt.close()
        print(f"Saved eps plots to data/plots/{Hamiltonian_name}_eps_plots.png")

    elif test_type == "T_max":
        # 3x2 grid for T_max tests
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        x_labels = ["T_max", "T_total"]
        x_cols = ["T_max", "T_total"]
        
        for row, perturb in enumerate(perturbation_params.keys()):
            for col, (x_label, x_col) in enumerate(zip(x_labels, x_cols)):
                ax = axes[row, col]
                
                # Filter data for this perturbation (including true eigenvalues)
                # Handle both string "None" and NaN/None values for perturb column
                if perturb == "None":
                    df_filtered_with_true = df[(df["perturb"] == "None") | (pd.isna(df["perturb"]))]
                else:
                    df_filtered_with_true = df[df["perturb"] == perturb]
                
                # Filter out true eigenvalues for plotting
                df_filtered = df_filtered_with_true[df_filtered_with_true["test_type"] != "true eigenvalues"]
                
                # Plot each algorithm in specified order
                for algo in algo_order:
                    df_algo = df_filtered[df_filtered["algorithm"] == algo]
                    if df_algo.empty:
                        continue
                    df_algo = df_algo.sort_values(by=x_col)
                    ax.plot(df_algo[x_col], df_algo["error_to_target"], marker='o', label=algo, linewidth=2)
                
                ax.set_xlabel(x_label)
                ax.set_ylabel("error_to_target")
                ax.set_title(f"perturb={perturb}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                if use_y_axis_log_scale:
                    ax.set_yscale('log')
                if use_x_axis_log_scale:
                    ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f"data/plots/{Hamiltonian_name}_T_max_plots.png", dpi=150)
        plt.close()
        print(f"Saved T_max plots to data/plots/{Hamiltonian_name}_T_max_plots.png")
