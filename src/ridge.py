import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import joblib

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (16, 8),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "SEED": 42,
    "TEST_RATIO": 0.20,
    "LAG_DAYS_OPTIONS": [10, 20, 30],
    "WINDOW_TYPES": ["expanding", "rolling"],
    "OPTUNA_N_TRIALS": 60,
    "OPTUNA_TIMEOUT": 1800,
    "ROLLING_WINDOW_SIZE": 800,
}

BASE_DIR = Path("nepse_ridge_final_fixed")
DIRS = {
    "raw_data": BASE_DIR / "raw_data",
    "processed": BASE_DIR / "processed",
    "predictions": BASE_DIR / "predictions",
    "plots": BASE_DIR / "plots",
    "tables": BASE_DIR / "tables",
    "models": BASE_DIR / "models",
    "optuna_studies": BASE_DIR / "optuna_studies",
    "archives": BASE_DIR / "archives",
}
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

np.random.seed(CONFIG["SEED"])

# =============================================================================
# DATA PREPARATION – ONLY PURE LAGS (NO LOOK-AHEAD FEATURES)
# =============================================================================
def prepare_data(filepath: str, lag_days: int) -> pd.DataFrame:
    print(f"Preparing data with {lag_days} pure lag days (no rolling indicators to prevent leakage)...")
    with open(filepath, "r") as f:
        raw = json.load(f)
    data = raw["data"] if "data" in raw else raw
    df = pd.DataFrame(data)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["f_date"] = pd.to_datetime(df["f_date"])
    df = df[["f_date", "close"]].sort_values("f_date").reset_index(drop=True)

    df["Log_Return"] = np.log(df["close"] / df["close"].shift(1))

    # Only pure lagged log-returns (no future information)
    for lag in range(1, lag_days + 1):
        df[f"lag_{lag}"] = df["Log_Return"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    print(f"Dataset ready: {df.shape[0]} observations, {df.shape[1]} features (pure lags only)")

    df[["f_date", "close"]].to_csv(DIRS["raw_data"] / "nepse_raw.csv", index=False)
    df.to_csv(DIRS["processed"] / f"nepse_processed_lag_{lag_days}.csv", index=False)

    return df

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    directional = np.mean(np.sign(actual[1:]) == np.sign(np.diff(predicted))) * 100 if len(actual) > 1 else 0.0

    return {
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "R²": round(r2, 4),
        "Directional Accuracy (%)": round(directional, 2),
    }

# =============================================================================
# RIDGE HYPERPARAMETER OPTIMIZATION
# =============================================================================
def tune_ridge(X: pd.DataFrame, y: pd.Series, lag_days: int):
    print("Initiating Optuna hyperparameter optimization for Ridge regression...")
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-5, 100.0, log=True)
        model = Ridge(alpha=alpha, random_state=CONFIG["SEED"])

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(mean_squared_error(y.iloc[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=CONFIG["SEED"]))
    study.optimize(objective, n_trials=CONFIG["OPTUNA_N_TRIALS"], timeout=CONFIG["OPTUNA_TIMEOUT"],
                   show_progress_bar=True)

    print(f"Best Ridge alpha found: {study.best_params['alpha']:.6f}")
    joblib.dump(study, DIRS["optuna_studies"] / f"ridge_study_lag_{lag_days}.pkl")

    return study.best_params["alpha"]

# =============================================================================
# WALK-FORWARD VALIDATION WITH TUNED RIDGE (NO DATA LEAKAGE)
# =============================================================================
def run_ridge_forecast(df: pd.DataFrame, window_type: str, lag_days: int):
    feature_columns = [col for col in df.columns if col.startswith("lag_")]
    target = "Log_Return"

    split_idx = int(len(df) * (1 - CONFIG["TEST_RATIO"]))
    initial_train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    last_training_price = initial_train["close"].iloc[-1]

    print(f"\n=== {window_type.upper()} WINDOW | LAG DAYS = {lag_days} | Test observations = {len(test)} ===")

    # Hyperparameter tuning on initial training period (leak-free)
    best_alpha = tune_ridge(initial_train[feature_columns], initial_train[target], lag_days)

    # Storage for predictions
    log_return_predictions = []

    print("Commencing walk-forward forecasting with tuned Ridge...")
    for i in tqdm(range(len(test)), desc="Walk-Forward Steps", unit="day"):
        if window_type == "expanding":
            train = df.iloc[:split_idx + i]
        else:  # rolling
            start = max(0, split_idx + i - CONFIG["ROLLING_WINDOW_SIZE"])
            train = df.iloc[start:split_idx + i]

        X_train = train[feature_columns]
        y_train = train[target]
        X_test = test.iloc[[i]][feature_columns]

        model = Ridge(alpha=best_alpha, random_state=CONFIG["SEED"])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        log_return_predictions.append(pred)

        if i == len(test) - 1:
            final_model = model

    # Convert predictions
    log_return_predictions = np.array(log_return_predictions)
    actual_log_returns = test["Log_Return"].values

    # Price reconstruction
    previous_prices = np.concatenate(([last_training_price], test["close"].values[:-1]))
    predicted_prices = previous_prices * np.exp(log_return_predictions)
    actual_prices = test["close"].values

    # Metrics
    log_metrics = compute_metrics(actual_log_returns, log_return_predictions)
    log_metrics = {f"Log_Return_{k}": v for k, v in log_metrics.items()}

    price_metrics = compute_metrics(actual_prices, predicted_prices)
    price_metrics = {f"Price_{k}": v for k, v in price_metrics.items()}

    metrics = {**log_metrics, **price_metrics}

    # Save results
    suffix = f"{window_type}_lag{lag_days}"
    results_df = pd.DataFrame({
        "Date": test["f_date"].values,
        "Actual_Close": actual_prices,
        "Predicted_Close": predicted_prices,
        "Actual_Log_Return": actual_log_returns,
        "Predicted_Log_Return": log_return_predictions,
    })
    results_df.to_csv(DIRS["predictions"] / f"forecast_{suffix}.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(DIRS["tables"] / f"performance_metrics_{suffix}.csv", index=False)

    # Plot
    plot_n = min(400, len(actual_prices))
    plt.figure()
    plt.plot(actual_prices[-plot_n:], label="Actual Close", color="black", linewidth=2)
    plt.plot(predicted_prices[-plot_n:], label="Ridge Predicted Close", color="darkorange", linewidth=2)
    plt.title(f"NEPSE Index Forecast – Last {plot_n} Days ({window_type.capitalize()}, Lag={lag_days})")
    plt.xlabel("Time Steps")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / f"forecast_comparison_{suffix}.pdf", dpi=300)
    plt.savefig(DIRS["plots"] / f"forecast_comparison_{suffix}.png", dpi=300)
    plt.close()

    # Save model and parameters
    joblib.dump(final_model, DIRS["models"] / f"ridge_final_{suffix}.pkl")
    with open(DIRS["models"] / f"best_alpha_{suffix}.json", "w") as f:
        json.dump({"alpha": best_alpha}, f, indent=4)

    return metrics

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    data_filepath = "/home/raja/data1/sahaj/nepse-quantum-computing/data/nepse.json"

    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found at {data_filepath}")
        sys.exit(1)

    summary_results = {}

    for lag_days in CONFIG["LAG_DAYS_OPTIONS"]:
        print(f"\n{'=' * 90}")
        print(f"PROCESSING LAG CONFIGURATION: {lag_days} DAYS")
        print(f"{'=' * 90}")
        df = prepare_data(data_filepath, lag_days)

        for window_type in CONFIG["WINDOW_TYPES"]:
            config_key = f"{window_type}_lag{lag_days}"
            print(f"\nRunning: {config_key.upper()}")
            metrics = run_ridge_forecast(df, window_type, lag_days)
            summary_results[config_key] = metrics

    # Summary
    print("\n" + "=" * 100)
    print("RIDGE REGRESSION OUT-OF-SAMPLE PERFORMANCE SUMMARY")
    print("=" * 100)
    summary_df = pd.DataFrame(summary_results).T
    print(summary_df[[col for col in summary_df.columns if "Log_Return" in col or "Price" in col]])
    summary_df.to_csv(DIRS["tables"] / "performance_summary_all_configs.csv")

    # Archive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = DIRS["archives"] / f"ridge_complete_results_{timestamp}"
    shutil.make_archive(str(archive_path), "zip", BASE_DIR)
    print(f"\nAll outputs saved. Complete archive: {archive_path}.zip")