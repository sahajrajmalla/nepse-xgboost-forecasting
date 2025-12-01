# =============================================================================
# ⚛️ FINAL SOTA QUANTUM KERNEL SVR (QSVR)
# Architecture: Technicals -> PCA -> Angle Embedding (VQC Kernel) -> SVR
# Target: Direct Price Forecasting (Scaled)
# Optimized for: High R-Squared and Low RMSE
# =============================================================================

# 1. INSTALL
import sys, subprocess, os
def install():
    try: import pennylane
    except ImportError:
        print("⬇️ Installing Quantum Stack...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                               "pennylane", "torch", "pandas", "numpy", "scikit-learn", 
                               "matplotlib", "seaborn", "optuna", "loguru", "tqdm"])
install()

# 2. IMPORTS
import json
import time
import warnings
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import optuna
from loguru import logger
from pathlib import Path
from datetime import datetime
import shutil

warnings.filterwarnings("ignore")

# ==========================================
# 🎛️ 3. CONFIGURATION
# ==========================================
MODE = "PROD" 

CONFIG = {
    "N_QUBITS": 6,          # High dimension for better feature mapping
    "TRAIN_SIZE": 300,      # Last ~1.2 years (Optimal for Regime tracking)
    "TEST_SIZE": 30,        # Forecast next 1 month
    "N_TRIALS": 20,         # Optuna Trials
    "SEED": 42
}

# Setup Paths
BASE_DIR = Path("/content/qsvr_final_v2")
DIRS = {k: BASE_DIR / k for k in ["models", "results", "predictions", "logs", "plots", "params"]}
for p in DIRS.values(): p.mkdir(parents=True, exist_ok=True)

# Logging
logger.remove()
logger.add(DIRS["logs"] / f"run_{datetime.now():%Y%m%d_%H%M}.log", rotation="10 MB")
logger.add(lambda msg: print(msg), format="{message}", level="INFO")

# Seeding
def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    import torch; torch.manual_seed(seed)

seed_everything(CONFIG["SEED"])

# =============================================================================
# 📊 4. DATA PROCESSING (Direct Price Focus)
# =============================================================================
def process_data(filepath):
    with open(filepath, 'r') as f: raw = json.load(f)
    data = raw['data'] if 'data' in raw else raw
    df = pd.DataFrame(data)
    
    # Clean
    for c in ['close', 'open', 'high', 'low', 'volume']: 
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['f_date'] = pd.to_datetime(df['f_date'])
    df = df.sort_values('f_date').reset_index(drop=True)
    
    # --- Technical Indicators (Normalized) ---
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-8))))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    
    # Bollinger Position
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['B_Pos'] = (df['close'] - sma20) / (2 * std20 + 1e-8)
    
    # Momentum
    df['ROC'] = df['close'].pct_change(10)
    
    # CLEANUP
    df = df.dropna().reset_index(drop=True)
    
    # TRIM to Regime (Train + Test)
    # We only take the most recent data to ensure SVR doesn't get confused by 2010 prices
    total_needed = CONFIG["TRAIN_SIZE"] + CONFIG["TEST_SIZE"]
    if len(df) > total_needed:
        df = df.tail(total_needed).reset_index(drop=True)
        
    return df

def get_dataset(df):
    # Feature Selection
    features = ['close', 'RSI', 'MACD', 'B_Pos', 'ROC', 'volume']
    
    X = df[features].values[:-1] # Inputs
    y = df['close'].shift(-1).dropna().values # Target: Next Day Close Price
    
    min_len = min(len(X), len(y))
    X, y = X[-min_len:], y[-min_len:]
    dates = df['f_date'].iloc[-min_len:].values
    
    # Split
    split = len(X) - CONFIG["TEST_SIZE"]
    
    # --- SCALING (CRITICAL FOR SVR) ---
    # We scale X to [0, pi] for Quantum Embedding
    # We scale y to [0, 1] for SVR stability
    
    scaler_X = MinMaxScaler(feature_range=(0, np.pi)) 
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_train_raw = X[:split]
    X_test_raw = X[split:]
    y_train_raw = y[:split].reshape(-1, 1)
    y_test_raw = y[split:].reshape(-1, 1)
    
    # Fit on Train, Apply to Test
    X_train_sc = scaler_X.fit_transform(X_train_raw)
    X_test_sc = scaler_X.transform(X_test_raw)
    
    y_train_sc = scaler_y.fit_transform(y_train_raw).ravel() # SVR expects 1D array
    # Keep y_test raw for final metric comparison to be honest
    
    # PCA compression to fit Qubits
    pca = PCA(n_components=CONFIG["N_QUBITS"])
    X_train_q = pca.fit_transform(X_train_sc)
    X_test_q = pca.transform(X_test_sc)
    
    # Rescale PCA output to [0, pi] again to be safe for Quantum Circuit
    scaler_pca = MinMaxScaler(feature_range=(0, np.pi))
    X_train_q = scaler_pca.fit_transform(X_train_q)
    X_test_q = scaler_pca.transform(X_test_q)

    return {
        "X_train": X_train_q, "X_test": X_test_q,
        "y_train": y_train_sc, "y_test_raw": y_test_raw.ravel(),
        "dates_test": dates[split:], "scaler_y": scaler_y
    }

# =============================================================================
# ⚛️ 5. QUANTUM KERNEL (ANGLE EMBEDDING)
# =============================================================================
dev = qml.device("lightning.qubit", wires=CONFIG["N_QUBITS"])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    # 1. Embedding Data 1
    qml.AngleEmbedding(x1, wires=range(CONFIG["N_QUBITS"]), rotation='Y')
    qml.StronglyEntanglingLayers(weights=[[[0.1]*3]*CONFIG["N_QUBITS"]], wires=range(CONFIG["N_QUBITS"]))
    
    # 2. Inverse Embedding Data 2 (Adjoint)
    qml.adjoint(qml.StronglyEntanglingLayers)(weights=[[[0.1]*3]*CONFIG["N_QUBITS"]], wires=range(CONFIG["N_QUBITS"]))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(CONFIG["N_QUBITS"]), rotation='Y')
    
    return qml.probs(wires=range(CONFIG["N_QUBITS"]))

def compute_kernel_matrix(A, B):
    """Compute Gram Matrix |<x|z>|^2"""
    # Optimized: Returns overlap probability of state |00..0>
    # We only need the 0-th probability (all zeros state) which corresponds to overlap squared
    mat = np.zeros((len(A), len(B)))
    print(f"⚛️ Computing {len(A)}x{len(B)} Kernel...")
    
    for i in range(len(A)):
        for j in range(len(B)):
            probs = kernel_circuit(A[i], B[j])
            mat[i, j] = probs[0] # Prob(00..0) = |<psi|phi>|^2
            
    return mat

# =============================================================================
# ⚡ 6. OPTIMIZATION & PIPELINE
# =============================================================================
def objective(trial, K, y):
    C = trial.suggest_float("C", 0.1, 1000, log=True)
    epsilon = trial.suggest_float("epsilon", 0.0001, 0.1, log=True)
    
    # 3-Fold CV
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for t_idx, v_idx in tscv.split(K):
        K_train = K[np.ix_(t_idx, t_idx)]
        K_val = K[np.ix_(v_idx, t_idx)]
        y_t, y_v = y[t_idx], y[v_idx]
        
        model = SVR(kernel='precomputed', C=C, epsilon=epsilon)
        model.fit(K_train, y_t)
        preds = model.predict(K_val)
        scores.append(mean_squared_error(y_v, preds))
        
    return np.mean(scores)

def main():
    if not os.path.exists("/content/nepse.json"):
        print("❌ Please upload nepse.json")
        return

    # 1. Prepare
    df = process_data("/content/nepse.json")
    data = get_dataset(df)
    logger.info(f"Data Loaded. Train Size: {len(data['X_train'])}")
    
    # 2. Compute Kernels
    # This takes time, but it's the "Quantum" part
    start = time.time()
    K_train = compute_kernel_matrix(data["X_train"], data["X_train"])
    K_test = compute_kernel_matrix(data["X_test"], data["X_train"])
    logger.info(f"✅ Kernels computed in {time.time()-start:.1f}s")
    
    # 3. Tune
    logger.info("🔍 Tuning SVR...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, K_train, data["y_train"]), n_trials=CONFIG["N_TRIALS"])
    
    # 4. Final Model
    best = study.best_params
    svr = SVR(kernel='precomputed', C=best['C'], epsilon=best['epsilon'])
    svr.fit(K_train, data["y_train"])
    
    # 5. Predict & Inverse Scale
    pred_scaled = svr.predict(K_test)
    pred_prices = data["scaler_y"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual_prices = data["y_test_raw"]
    
    # 6. Metrics (ON PRICE)
    mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
    r2 = r2_score(actual_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    
    print("\n" + "═"*50)
    print(f"✨ QSVR FINAL PERFORMANCE (PRICE PREDICTION) ✨")
    print(f"MAPE: {mape:.2f}%  (Excellent if < 2%)")
    print(f"R²:   {r2:.4f}   (Should be positive now)")
    print(f"RMSE: {rmse:.2f}")
    print("═"*50)
    
    # 7. Plot
    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(data["dates_test"], actual_prices, label="Actual Price", color="black", alpha=0.6)
    plt.plot(data["dates_test"], pred_prices, label="Quantum Kernel Prediction", color="#1f77b4", linewidth=2, marker='.')
    plt.title(f"NEPSE Index Forecast: Quantum Kernel SVR (R²={r2:.3f})")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(DIRS["plots"] / "final_result.png")
    plt.show()
    
    # Save
    pd.DataFrame({
        "Date": data["dates_test"],
        "Actual": actual_prices,
        "Predicted": pred_prices
    }).to_csv(DIRS["predictions"] / "final_qsvr.csv", index=False)
    
    shutil.make_archive("/content/qsvr_results", 'zip', BASE_DIR)
    print(f"📦 Results Saved: /content/qsvr_results.zip")

if __name__ == "__main__":
    main()


# Data Loaded. Train Size: 299

# ⚛️ Computing 299x299 Kernel...
# ⚛️ Computing 30x299 Kernel...
# ✅ Kernels computed in 268.6s

# 🔍 Tuning SVR...


# ══════════════════════════════════════════════════
# ✨ QSVR FINAL PERFORMANCE (PRICE PREDICTION) ✨
# MAPE: 1.42%  (Excellent if < 2%)
# R²:   0.4390   (Should be positive now)
# RMSE: 50.44
# ══════════════════════════════════════════════════    