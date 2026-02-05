import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import subprocess
from pygam import LinearGAM, s

seed = 0
random.seed(seed)
np.random.seed(seed)

def run_dx_upload_minimal(filename):
    command = ["dx", "upload", filename]
    try:
        subprocess.run(command, capture_output=True)
    except FileNotFoundError:
        print("Error: 'dx' command not found. Make sure dx-toolkit is installed and in your PATH.")

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    ok = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    n_ok = int(np.sum(ok))

    if n_ok == 0:
        return dict(
            n_ok=0,
            mse=np.nan,
            rmse=np.nan,
            mae=np.nan,
            r2=np.nan,
            spearman_rho=np.nan,
            spearman_p=np.nan,
            pearson_r=np.nan,
            pearson_p=np.nan,
        )

    mse = mean_squared_error(y_true[ok], y_pred[ok])
    mae = mean_absolute_error(y_true[ok], y_pred[ok])
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true[ok], y_pred[ok])

    spearman_rho, spearman_p = spearmanr(y_pred[ok], y_true[ok])

    if n_ok > 1:
        pearson_r, pearson_p = pearsonr(y_pred[ok], y_true[ok])
    else:
        pearson_r, pearson_p = np.nan, np.nan

    return dict(
        n_ok=n_ok,
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
    )

df = pd.read_csv("data.csv")
df = df.dropna(subset=["p21021_i0", "p21003_i0"])

X = df["p21021_i0"].values.astype(np.float64)
Y = df["p21003_i0"].values.astype(np.float64)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
indices = np.arange(len(Y))
all_results = []

print("\nStarting 5-Fold Cross-Validation for ASI-only baselines (linear + GAM)...")

for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(indices), total=kf.get_n_splits(), desc="CV Folds")):
    fold = fold_idx + 1

    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]

    x_mean = np.mean(X_tr)
    y_mean = np.mean(Y_tr)
    x_var = np.mean((X_tr - x_mean) ** 2)

    if x_var == 0:
        a = 0.0
        b = y_mean
    else:
        a = np.mean((X_tr - x_mean) * (Y_tr - y_mean)) / x_var
        b = y_mean - a * x_mean

    preds_lin = a * X_te + b

    gam = LinearGAM(s(0, n_splines=20)).fit(X_tr.reshape(-1, 1), Y_tr)
    preds_gam = gam.predict(X_te.reshape(-1, 1))

    m_lin = compute_metrics(Y_te, preds_lin)
    m_gam = compute_metrics(Y_te, preds_gam)

    fold_results_dict = {
        "fold": fold,
        "lin_a_slope": a,
        "lin_b_intercept": b,
        "mean_train_age": float(np.mean(Y_tr)),
        "mean_train_ASI": float(np.mean(X_tr)),
        "lin_test_mse": m_lin["mse"],
        "lin_test_rmse": m_lin["rmse"],
        "lin_test_mae": m_lin["mae"],
        "lin_test_r2": m_lin["r2"],
        "lin_test_spearman_rho": m_lin["spearman_rho"],
        "lin_test_spearman_p": m_lin["spearman_p"],
        "lin_test_pearson_r": m_lin["pearson_r"],
        "lin_test_pearson_p": m_lin["pearson_p"],
        "gam_test_mse": m_gam["mse"],
        "gam_test_rmse": m_gam["rmse"],
        "gam_test_mae": m_gam["mae"],
        "gam_test_r2": m_gam["r2"],
        "gam_test_spearman_rho": m_gam["spearman_rho"],
        "gam_test_spearman_p": m_gam["spearman_p"],
        "gam_test_pearson_r": m_gam["pearson_r"],
        "gam_test_pearson_p": m_gam["pearson_p"],
        "n_train": len(Y_tr),
        "n_test": len(Y_te),
        "n_test_valid": m_lin["n_ok"],
    }
    all_results.append(fold_results_dict)

    print(
        f"Fold {fold} | "
        f"Linear MAE: {m_lin['mae']:.5f}, RMSE: {m_lin['rmse']:.5f}, R2: {m_lin['r2']:.5f}, Rho: {m_lin['spearman_rho']:.5f} | "
        f"GAM MAE: {m_gam['mae']:.5f}, RMSE: {m_gam['rmse']:.5f}, R2: {m_gam['r2']:.5f}, Rho: {m_gam['spearman_rho']:.5f}"
    )

overall_results_df = pd.DataFrame(all_results)
overall_results_filename = "all_folds_results_ASI_only_linear_and_GAM.csv"
overall_results_df.to_csv(overall_results_filename, index=False, float_format="%.8g")
print(f"\nSaved overall results to {overall_results_filename}")
run_dx_upload_minimal(overall_results_filename)

for prefix in ["lin", "gam"]:
    avg_mse = overall_results_df[f"{prefix}_test_mse"].mean()
    avg_rmse = overall_results_df[f"{prefix}_test_rmse"].mean()
    avg_mae = overall_results_df[f"{prefix}_test_mae"].mean()
    avg_r2 = overall_results_df[f"{prefix}_test_r2"].mean()
    avg_rho = overall_results_df[f"{prefix}_test_spearman_rho"].mean()
    print(f"Average {prefix.upper()} Metrics Across Folds")
    print(f"Average MSE: {avg_mse:.5f}")
    print(f"Average RMSE: {avg_rmse:.5f}")
    print(f"Average MAE: {avg_mae:.5f}")
    print(f"Average R2: {avg_r2:.5f}")
    print(f"Average Spearman ρ: {avg_rho:.5f}")

print("Cross-validation for ASI-only baselines finished.")
