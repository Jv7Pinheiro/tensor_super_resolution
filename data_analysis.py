
import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_numeric_value(value):
    if value is None or pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=object)
        parsed = []
        for item in arr.tolist():
            parsed.append(parse_numeric_value(item))
        return np.asarray(parsed, dtype=float)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return np.nan
        s_lower = s.lower()
        if s_lower in {"nan", "none", "null"}:
            return np.nan
        if s_lower in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if s_lower in {"-inf", "-infinity"}:
            return float("-inf")

        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return np.array([], dtype=float)
            items = [part.strip() for part in inner.split(",")]
            parsed = [parse_numeric_value(item) for item in items]
            return np.asarray(parsed, dtype=float)

        # Fall back to literal_eval for normal scalar strings and simple Python lists.
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return float(s)
        return parse_numeric_value(parsed)

    return float(value)


def get_target_error(value):
    value = parse_numeric_value(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        if target_eigenvalue_index >= len(value):
            return np.nan
        value = value[target_eigenvalue_index]
    return float(value)


def get_average_error(value):
    value = parse_numeric_value(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        return float(np.nanmean(value))
    return float(value)


os.makedirs("data/plots", exist_ok=True)

use_y_axis_log_scale = True  # Toggle this to switch between log and linear scale
use_x_axis_log_scale = True  # Toggle this to switch between log and linear scale
    
perturbation_params = { # length 3
    "None": None,
    "Small": {"range": 1, "scale": 0.5},
    "Big": {"range": 3, "scale": 1},
}
Hamiltonian_name = "XXZ_8x8"
df = pd.read_csv(f"data/dataframes/{Hamiltonian_name}.csv")
algo_order = df["algorithm"].unique().dropna()

target_eigenvalue_index = 2
df["error_to_target"] = df["errors"].map(get_target_error)
df["average_error"] = df["errors"].map(get_average_error)

for test_type in df["test_type"].unique():
    # if test_type == "true eigenvalues": continue
    if test_type == "eps":
        # 3x3 grid for eps tests
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"Eps scaling tests for {Hamiltonian_name} eigenvalue {target_eigenvalue_index}", fontsize=32)

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
                
                # Keep this figure focused on the current experiment type.
                df_filtered = df_filtered_with_true[df_filtered_with_true["test_type"] == "eps"]
                
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

        # Average error across all eigenvalues
        fig_avg, axes_avg = plt.subplots(3, 3, figsize=(15, 12))
        fig_avg.suptitle(f"Average error across all eigenvalues for {Hamiltonian_name} (eps)", fontsize=32)

        for row, perturb in enumerate(perturbation_params.keys()):
            for col, (x_label, x_col) in enumerate(zip(x_labels, x_cols)):
                ax = axes_avg[row, col]

                if perturb == "None":
                    df_filtered_with_true = df[(df["perturb"] == "None") | (pd.isna(df["perturb"]))]
                else:
                    df_filtered_with_true = df[df["perturb"] == perturb]

                df_filtered = df_filtered_with_true[df_filtered_with_true["test_type"] == "eps"]

                for algo in algo_order:
                    df_algo = df_filtered[df_filtered["algorithm"] == algo]
                    if df_algo.empty:
                        continue
                    df_algo = df_algo.sort_values(by=x_col)
                    ax.plot(df_algo[x_col], df_algo["average_error"], marker='o', label=algo, linewidth=2)

                ax.set_xlabel(x_label)
                ax.set_ylabel("average_error")
                ax.set_title(f"perturb={perturb}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                if use_y_axis_log_scale:
                    ax.set_yscale('log')
                if use_x_axis_log_scale:
                    ax.set_xscale('log')

                if col == 0:
                    ax.invert_xaxis()

        plt.tight_layout()
        plt.savefig(f"data/plots/{Hamiltonian_name}_eps_plots_avg.png", dpi=150)
        plt.close()
        print(f"Saved avg eps plots to data/plots/{Hamiltonian_name}_eps_plots_avg.png")

    elif test_type == "T_max":
        # 3x2 grid for T_max tests
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(f"T_max scaling tests for {Hamiltonian_name} eigenvalue {target_eigenvalue_index}", fontsize=32)

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
                
                # Keep this figure focused on the current experiment type.
                df_filtered = df_filtered_with_true[df_filtered_with_true["test_type"] == "T_max"]
                
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

        # Average error across all eigenvalues
        fig_avg, axes_avg = plt.subplots(3, 2, figsize=(12, 12))
        fig_avg.suptitle(f"Average error across all eigenvalues for {Hamiltonian_name} (T_max)", fontsize=32)

        for row, perturb in enumerate(perturbation_params.keys()):
            for col, (x_label, x_col) in enumerate(zip(x_labels, x_cols)):
                ax = axes_avg[row, col]

                if perturb == "None":
                    df_filtered_with_true = df[(df["perturb"] == "None") | (pd.isna(df["perturb"]))]
                else:
                    df_filtered_with_true = df[df["perturb"] == perturb]

                df_filtered = df_filtered_with_true[df_filtered_with_true["test_type"] == "T_max"]

                for algo in algo_order:
                    df_algo = df_filtered[df_filtered["algorithm"] == algo]
                    if df_algo.empty:
                        continue
                    df_algo = df_algo.sort_values(by=x_col)
                    ax.plot(df_algo[x_col], df_algo["average_error"], marker='o', label=algo, linewidth=2)

                ax.set_xlabel(x_label)
                ax.set_ylabel("average_error")
                ax.set_title(f"perturb={perturb}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                if use_y_axis_log_scale:
                    ax.set_yscale('log')
                if use_x_axis_log_scale:
                    ax.set_xscale('log')

        plt.tight_layout()
        plt.savefig(f"data/plots/{Hamiltonian_name}_T_max_plots_avg.png", dpi=150)
        plt.close()
        print(f"Saved avg T_max plots to data/plots/{Hamiltonian_name}_T_max_plots_avg.png")
