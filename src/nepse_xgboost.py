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
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
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
    "OPTUNA_TIMEOUT": 2400,  # 40 minutes per tuning session
    "ROLLING_WINDOW_SIZE": 800,
}

BASE_DIR = Path("nepse_xgboost_final")
DIRS = {
    "raw_data": BASE_DIR / "raw_data",
    "processed": BASE_DIR / "processed",
    "predictions": BASE_DIR / "predictions",
    "feature_importance": BASE_DIR / "feature_importance",
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
# DATA PREPARATION AND FEATURE ENGINEERING
# =============================================================================
def prepare_data(filepath: str, lag_days: int) -> pd.DataFrame:
    print(f"Preparing data with {lag_days} lag days...")
    with open(filepath, "r") as f:
        raw = json.load(f)
    data = raw["data"] if "data" in raw else raw
    df = pd.DataFrame(data)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["f_date"] = pd.to_datetime(df["f_date"])
    df = df[["f_date", "close"]].sort_values("f_date").reset_index(drop=True)

    df["Log_Return"] = np.log(df["close"] / df["close"].shift(1))

    # Lagged log returns
    for lag in range(1, lag_days + 1):
        df[f"lag_{lag}"] = df["Log_Return"].shift(lag)

    # Technical indicators
    df["vol_5"] = df["Log_Return"].rolling(5).std()
    df["vol_20"] = df["Log_Return"].rolling(20).std()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["mean_10"] = df["Log_Return"].rolling(10).mean()
    df["std_20"] = df["Log_Return"].rolling(20).std()

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"Dataset ready: {df.shape[0]} observations, {df.shape[1]} features")

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
# XGBOOST HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# =============================================================================
def tune_xgboost(X: pd.DataFrame, y: pd.Series, lag_days: int):
    print("Initiating Optuna hyperparameter optimization for XGBoost...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "objective": "reg:squarederror",
            "random_state": CONFIG["SEED"],
            "tree_method": "hist",
            "n_jobs": -1,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            model = XGBRegressor(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(mean_squared_error(y.iloc[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=CONFIG["SEED"]))
    study.optimize(objective, n_trials=CONFIG["OPTUNA_N_TRIALS"], timeout=CONFIG["OPTUNA_TIMEOUT"],
                   show_progress_bar=True)

    print(f"Best hyperparameters found: {study.best_params}")
    joblib.dump(study, DIRS["optuna_studies"] / f"xgboost_study_lag_{lag_days}.pkl")

    best_params = study.best_params
    best_params.update({
        "objective": "reg:squarederror",
        "random_state": CONFIG["SEED"],
        "tree_method": "hist",
        "n_jobs": -1,
    })
    return best_params

# =============================================================================
# WALK-FORWARD VALIDATION WITH XGBOOST
# =============================================================================
def run_xgboost_forecast(df: pd.DataFrame, window_type: str, lag_days: int):
    feature_columns = [col for col in df.columns if col not in ["f_date", "close", "Log_Return"]]
    target = "Log_Return"

    split_idx = int(len(df) * (1 - CONFIG["TEST_RATIO"]))
    initial_train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    last_training_price = initial_train["close"].iloc[-1]

    print(f"\n=== {window_type.upper()} WINDOW | LAG DAYS = {lag_days} | Test observations = {len(test)} ===")

    # Hyperparameter tuning on initial training period
    best_params = tune_xgboost(initial_train[feature_columns], initial_train[target], lag_days)

    # Storage for predictions
    log_return_predictions = []

    print("Commencing walk-forward forecasting...")
    for i in tqdm(range(len(test)), desc="Walk-Forward Steps", unit="day"):
        if window_type == "expanding":
            train_end = split_idx + i
            train = df.iloc[:train_end]
        else:  # rolling
            start = max(0, split_idx + i - CONFIG["ROLLING_WINDOW_SIZE"])
            train = df.iloc[start:split_idx + i]

        X_train = train[feature_columns]
        y_train = train[target]
        X_test = test.iloc[[i]][feature_columns]

        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        log_return_predictions.append(pred)

        # Save final model and importance
        if i == len(test) - 1:
            final_model = model
            importance = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

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

    # Save detailed results
    suffix = f"{window_type}_lag{lag_days}"
    results_df = pd.DataFrame({
        "Date": test["f_date"].values,
        "Actual_Close": actual_prices,
        "Predicted_Close": predicted_prices,
        "Actual_Log_Return": actual_log_returns,
        "Predicted_Log_Return": log_return_predictions,
    })
    results_df.to_csv(DIRS["predictions"] / f"forecast_{suffix}.csv", index=False)

    # Metrics table (clean, readable format)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(DIRS["tables"] / f"performance_metrics_{suffix}.csv", index=False)

    # Feature importance
    importance.to_csv(DIRS["feature_importance"] / f"importance_{suffix}.csv", index=False)
    plt.figure()
    sns.barplot(data=importance.head(15), x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top 15 Feature Importance ({window_type.capitalize()}, Lag={lag_days})")
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / f"feature_importance_{suffix}.pdf", dpi=300)
    plt.savefig(DIRS["plots"] / f"feature_importance_{suffix}.png", dpi=300)
    plt.close()

    # Forecast plot
    plot_n = min(400, len(actual_prices))
    plt.figure()
    plt.plot(actual_prices[-plot_n:], label="Actual Close", color="black", linewidth=2)
    plt.plot(predicted_prices[-plot_n:], label="XGBoost Predicted Close", color="darkgreen", linewidth=2)
    plt.title(f"NEPSE Index Forecast – Last {plot_n} Days ({window_type.capitalize()}, Lag={lag_days})")
    plt.xlabel("Time Steps")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIRS["plots"] / f"forecast_comparison_{suffix}.pdf", dpi=300)
    plt.savefig(DIRS["plots"] / f"forecast_comparison_{suffix}.png", dpi=300)
    plt.close()

    # Save final model and parameters
    joblib.dump(final_model, DIRS["models"] / f"xgboost_final_{suffix}.pkl")
    with open(DIRS["models"] / f"best_params_{suffix}.json", "w") as f:
        json.dump(best_params, f, indent=4)

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
            metrics = run_xgboost_forecast(df, window_type, lag_days)
            summary_results[config_key] = metrics

    # Final summary table
    print("\n" + "=" * 100)
    print("XGBOOST OUT-OF-SAMPLE PERFORMANCE SUMMARY")
    print("=" * 100)
    summary_df = pd.DataFrame(summary_results).T
    print(summary_df[[col for col in summary_df.columns if "Log_Return" in col or "Price" in col]])
    summary_df.to_csv(DIRS["tables"] / "performance_summary_all_configs.csv")

    # Archive all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = DIRS["archives"] / f"xgboost_complete_results_{timestamp}"
    shutil.make_archive(str(archive_path), "zip", BASE_DIR)
    print(f"\nAll outputs and models saved. Complete archive: {archive_path}.zip")