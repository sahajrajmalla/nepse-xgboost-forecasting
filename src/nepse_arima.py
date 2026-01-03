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
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    "WINDOW_TYPES": ["expanding", "rolling"],
    "OPTUNA_N_TRIALS": 60,
    "OPTUNA_TIMEOUT": 2400,  # 40 minutes per tuning session
    "ROLLING_WINDOW_SIZE": 800,
    "ARIMA_SEARCH_SPACE": {
        "p": (0, 8),
        "d": (0, 1),
        "q": (0, 4),
    },
}

BASE_DIR = Path("nepse_arima_final")
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
# DATA PREPARATION
# =============================================================================
def prepare_data(filepath: str) -> pd.DataFrame:
    print("Preparing data for ARIMA forecasting...")
    with open(filepath, "r") as f:
        raw = json.load(f)
    data = raw["data"] if "data" in raw else raw
    df = pd.DataFrame(data)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["f_date"] = pd.to_datetime(df["f_date"])
    df = df[["f_date", "close"]].sort_values("f_date").reset_index(drop=True)

    df["Log_Return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna().reset_index(drop=True)

    print(f"Dataset ready: {df.shape[0]} observations")

    df[["f_date", "close"]].to_csv(DIRS["raw_data"] / "nepse_raw.csv", index=False)
    df.to_csv(DIRS["processed"] / "nepse_log_returns.csv", index=False)

    return df

# =============================================================================
# PERFORMANCE METRICS (Identical to XGBoost)
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
# ARIMA HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Parallel to XGBoost Tuning)
# =============================================================================
def tune_arima(train_series: pd.Series, window_type: str):
    print(f"Initiating Optuna hyperparameter optimization for ARIMA ({window_type} window)...")
    def objective(trial):
        p = trial.suggest_int("p", CONFIG["ARIMA_SEARCH_SPACE"]["p"][0], CONFIG["ARIMA_SEARCH_SPACE"]["p"][1])
        d = trial.suggest_int("d", CONFIG["ARIMA_SEARCH_SPACE"]["d"][0], CONFIG["ARIMA_SEARCH_SPACE"]["d"][1])
        q = trial.suggest_int("q", CONFIG["ARIMA_SEARCH_SPACE"]["q"][0], CONFIG["ARIMA_SEARCH_SPACE"]["q"][1])
        try:
            model = ARIMA(train_series, order=(p, d, q))
            fit = model.fit()
            return fit.aic
        except Exception:
            return np.inf

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=CONFIG["SEED"]))
    study.optimize(objective, n_trials=CONFIG["OPTUNA_N_TRIALS"], timeout=CONFIG["OPTUNA_TIMEOUT"],
                   show_progress_bar=True)

    print(f"Best ARIMA order found: {study.best_params}")
    joblib.dump(study, DIRS["optuna_studies"] / f"arima_study_{window_type}.pkl")

    return (study.best_params["p"], study.best_params["d"], study.best_params["q"])

# =============================================================================
# WALK-FORWARD VALIDATION WITH TUNED ARIMA (Parallel Structure to XGBoost)
# =============================================================================
def run_arima_forecast(df: pd.DataFrame, window_type: str):
    target = "Log_Return"

    split_idx = int(len(df) * (1 - CONFIG["TEST_RATIO"]))
    initial_train_series = df[target].iloc[:split_idx]
    test = df.iloc[split_idx:]

    last_training_price = df["close"].iloc[split_idx - 1]

    print(f"\n=== {window_type.upper()} WINDOW | Test observations = {len(test)} ===")

    # Hyperparameter tuning on initial training period
    best_order = tune_arima(initial_train_series, window_type)

    # Storage for predictions
    log_return_predictions = []

    print("Commencing walk-forward forecasting with tuned ARIMA...")
    for i in tqdm(range(len(test)), desc="Walk-Forward Steps", unit="day"):
        if window_type == "expanding":
            train_series = df[target].iloc[:split_idx + i]
        else:  # rolling
            start = max(0, split_idx + i - CONFIG["ROLLING_WINDOW_SIZE"])
            train_series = df[target].iloc[start:split_idx + i]

        try:
            model = ARIMA(train_series, order=best_order)
            fit = model.fit()
            pred = fit.forecast(steps=1).iloc[0]
        except Exception as e:
            print(f"ARIMA convergence failed at step {i}: {e}. Using zero forecast.")
            pred = 0.0

        log_return_predictions.append(pred)

    # Convert predictions
    log_return_predictions = np.array(log_return_predictions)
    actual_log_returns = test["Log_Return"].values

    # Price reconstruction
    previous_prices = np.concatenate(([last_training_price], test["close"].values[:-1]))
    predicted_prices = previous_prices * np.exp(log_return_predictions)
    actual_prices = test["close"].values

    # Metrics (identical format)
    log_metrics = compute_metrics(actual_log_returns, log_return_predictions)
    log_metrics = {f"Log_Return_{k}": v for k, v in log_metrics.items()}

    price_metrics = compute_metrics(actual_prices, predicted_prices)
    price_metrics = {f"Price_{k}": v for k, v in price_metrics.items()}

    metrics = {**log_metrics, **price_metrics}

    # Save detailed results (comparable naming)
    suffix = f"{window_type}_arima_tuned"
    results_df = pd.DataFrame({
        "Date": test["f_date"].values,
        "Actual_Close": actual_prices,
        "Predicted_Close": predicted_prices,
        "Actual_Log_Return": actual_log_returns,
        "Predicted_Log_Return": log_return_predictions,
    })
    results_df.to_csv(DIRS["predictions"] / f"forecast_{suffix}.csv", index=False)

    # Metrics table (clean, readable)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(DIRS["tables"] / f"performance_metrics_{suffix}.csv", index=False)

    # Forecast plot
    plot_n = min(400, len(actual_prices))
    plt.figure()
    plt.plot(actual_prices[-plot_n:], label="Actual Close", color="black", linewidth=2)
    plt.plot(predicted_prices[-plot_n:], label=f"ARIMA{best_order} Predicted Close", color="darkblue", linewidth=2)
    plt.title(f"NEPSE Index Forecast – Last {plot_n} Days ({window_type.capitalize()}, Tuned ARIMA)")
    plt.xlabel("Time Steps")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / f"forecast_comparison_{suffix}.pdf", dpi=300)
    plt.savefig(DIRS["plots"] / f"forecast_comparison_{suffix}.png", dpi=300)
    plt.close()

    # Save best order
    with open(DIRS["models"] / f"best_order_{suffix}.json", "w") as f:
        json.dump({"order": best_order}, f, indent=4)

    return metrics

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    data_filepath = "/home/raja/data1/sahaj/nepse-quantum-computing/data/nepse.json"

    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found at {data_filepath}")
        sys.exit(1)

    df = prepare_data(data_filepath)

    summary_results = {}

    for window_type in CONFIG["WINDOW_TYPES"]:
        config_key = f"{window_type}_arima_tuned"
        print(f"\n{'=' * 90}")
        print(f"RUNNING CONFIGURATION: {config_key.upper()}")
        print(f"{'=' * 90}")
        metrics = run_arima_forecast(df, window_type)
        summary_results[config_key] = metrics

    # Final summary table
    print("\n" + "=" * 100)
    print("ARIMA OUT-OF-SAMPLE PERFORMANCE SUMMARY")
    print("=" * 100)
    summary_df = pd.DataFrame(summary_results).T
    print(summary_df[[col for col in summary_df.columns if "Log_Return" in col or "Price" in col]])
    summary_df.to_csv(DIRS["tables"] / "performance_summary_all_configs.csv")

    # Archive all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = DIRS["archives"] / f"arima_complete_results_{timestamp}"
    shutil.make_archive(str(archive_path), "zip", BASE_DIR)
    print(f"\nAll outputs saved. Complete archive: {archive_path}.zip")