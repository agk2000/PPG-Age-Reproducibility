import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import subprocess

seed = 0
random.seed(seed)
np.random.seed(seed)

def run_dx_upload_minimal(filename):
    command = ["dx", "upload", filename]
    try:
        subprocess.run(command, capture_output=True)
    except FileNotFoundError:
        print("Error: 'dx' command not found. Make sure dx-toolkit is installed and in your PATH.")

df = pd.read_csv("data.csv")
df = df.dropna(subset=['p21021_i0', 'p21003_i0'])  # ASI, Age

X = df['p21021_i0'].values.astype(np.float64)
Y = df['p21003_i0'].values.astype(np.float64)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
all_results = []
indices = np.arange(len(Y))

print("\nStarting 5-Fold Cross-Validation for ASI-only baseline...")

for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(indices), total=kf.get_n_splits(), desc="CV Folds")):
    fold = fold_idx + 1

    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]

    # simple fit: age = a*ASI + b
    x_mean = np.mean(X_tr)
    y_mean = np.mean(Y_tr)
    x_var = np.mean((X_tr - x_mean) ** 2)

    if x_var == 0:
        a = 0.0
        b = y_mean
    else:
        a = np.mean((X_tr - x_mean) * (Y_tr - y_mean)) / x_var
        b = y_mean - a * x_mean

    preds = a * X_te + b

    preds_flat = preds.flatten()
    Y_te_flat = Y_te.flatten()

    valid_indices = ~np.isnan(preds_flat) & ~np.isnan(Y_te_flat) & ~np.isinf(preds_flat) & ~np.isinf(Y_te_flat)
    n_test_valid = int(np.sum(valid_indices))

    mse = mean_squared_error(Y_te_flat[valid_indices], preds_flat[valid_indices])
    mae = mean_absolute_error(Y_te_flat[valid_indices], preds_flat[valid_indices])
    rmse = float(np.sqrt(mse))
    r2 = r2_score(Y_te_flat[valid_indices], preds_flat[valid_indices])

    spearman_corr, spearman_p = spearmanr(preds_flat[valid_indices], Y_te_flat[valid_indices])

    if n_test_valid > 1:
        pearson_corr, pearson_p = pearsonr(preds_flat[valid_indices], Y_te_flat[valid_indices])
    else:
        pearson_corr, pearson_p = np.nan, np.nan

    fold_results_dict = {
        "fold": fold,
        "a_slope": a,
        "b_intercept": b,
        "mean_train_age": float(np.mean(Y_tr)),
        "mean_train_ASI": float(np.mean(X_tr)),
        "test_mse": mse,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "test_spearman_rho": spearman_corr,
        "test_spearman_p": spearman_p,
        "test_pearson_r": pearson_corr,
        "test_pearson_p": pearson_p,
        "n_train": len(Y_tr),
        "n_test": len(Y_te),
        "n_test_valid": n_test_valid
    }
    all_results.append(fold_results_dict)
    print(f"Fold {fold} - MAE: {mae:.5f}, RMSE: {rmse:.5f}, R2: {r2:.5f}, Rho: {spearman_corr:.5f}")

overall_results_df = pd.DataFrame(all_results)
overall_results_filename = "all_folds_results_ASI_only_baseline.csv"
overall_results_df.to_csv(overall_results_filename, index=False, float_format='%.8g')
print(f"\nSaved overall results to {overall_results_filename}")
run_dx_upload_minimal(overall_results_filename)

avg_mse = overall_results_df['test_mse'].mean()
avg_rmse = overall_results_df['test_rmse'].mean()
avg_mae = overall_results_df['test_mae'].mean()
avg_r2 = overall_results_df['test_r2'].mean()
avg_rho = overall_results_df['test_spearman_rho'].mean()

print("Average Metrics Across Folds")
print(f"Average MSE: {avg_mse:.5f}")
print(f"Average RMSE: {avg_rmse:.5f}")
print(f"Average MAE: {avg_mae:.5f}")
print(f"Average R2: {avg_r2:.5f}")
print(f"Average Spearman ρ: {avg_rho:.5f}")

print("Cross-validation for ASI-only baseline finished.")
