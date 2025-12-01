# =============================================================================
# 🌊 QUANTUM RESERVOIR COMPUTING (QRC) - NEURIPS 2025 RESEARCH EDITION
# Target: NEPSE Index (Log-Return Forecasting)
# Methodology: Windowed Input -> PCA -> Quantum Reservoir (Fixed) -> Ridge Regression
# Advantage: Extremely fast training, captures chaotic dynamics, no vanishing gradients.
# =============================================================================

# 1. INSTALL DEPENDENCIES
import sys, subprocess, os
def install_packages():
    print("⬇️ Installing Quantum Research Stack...")
    try: import pennylane
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                               "pennylane", "torch", "pandas", "numpy", "scikit-learn", 
                               "matplotlib", "seaborn", "optuna", "loguru", "tqdm"])
install_packages()

# 2. IMPORTS
import json
import warnings
import time
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from loguru import logger
from pathlib import Path
from datetime import datetime
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 🎛️ 3. CONFIGURATION
# ==========================================
# "DEV": Fast check (fewer qubits, small data)
# "PROD": Full power (Max qubits, full history, rigorous tuning)
MODE = "DEV"

CONFIG = {
    "N_QUBITS": 8 if MODE == "PROD" else 4,  # Reservoir Size (Width)
    "TRAIN_SIZE": 500 if MODE == "PROD" else 150, # Sliding Window Size
    "TEST_SIZE": 30,
    "WINDOW_SIZE": 10,       # Lookback history fed into reservoir
    "N_TRIALS": 30 if MODE == "PROD" else 5,
    "SEED": 42
}

# Paths
BASE_DIR = Path("/content/qrc_output")
DIRS = {k: BASE_DIR / k for k in ["models", "results", "predictions", "logs", "plots", "params"]}
for p in DIRS.values(): p.mkdir(parents=True, exist_ok=True)

# Logging
logger.remove()
logger.add(DIRS["logs"] / f"qrc_{datetime.now():%Y%m%d_%H%M}.log", rotation="10 MB")
logger.add(lambda msg: print(msg), format="{message}", level="INFO")

# Seeding
def seed_everything(seed=42):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

seed_everything(CONFIG["SEED"])
logger.info(f"🚀 QRC Initialized. Mode: {MODE} | Reservoir Qubits: {CONFIG['N_QUBITS']}")

# =============================================================================
# 📊 4. DATA PIPELINE (LAGGED & STATIONARY)
# =============================================================================
def process_data(filepath):
    with open(filepath, 'r') as f: raw = json.load(f)
    data = raw['data'] if 'data' in raw else raw
    df = pd.DataFrame(data)
    
    # Cleaning
    for c in ['close', 'open', 'high', 'low', 'volume']: 
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['f_date'] = pd.to_datetime(df['f_date'])
    df = df.sort_values('f_date').reset_index(drop=True)
    
    # --- Features ---
    # 1. Log Returns (The Target)
    df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Volatility
    df['Vol_5'] = df['Log_Ret'].rolling(5).std()
    
    # 3. Momentum
    df['Mom_3'] = df['Log_Ret'].rolling(3).mean()
    
    # 4. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-8))))

    # Cleanup
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Regime Selection
    total_needed = CONFIG["TRAIN_SIZE"] + CONFIG["TEST_SIZE"] + CONFIG["WINDOW_SIZE"]
    if len(df) > total_needed:
        df = df.tail(total_needed).reset_index(drop=True)
        
    return df

def create_windowed_dataset(df):
    """
    Creates sliding windows of history to feed into the Reservoir.
    """
    features = ['Log_Ret', 'Vol_5', 'Mom_3', 'RSI']
    data_matrix = df[features].values
    target = df['Log_Ret'].shift(-1).fillna(0).values
    
    X, y = [], []
    w = CONFIG["WINDOW_SIZE"]
    
    # Create windows: [t-w, ..., t] -> Predict t+1
    for i in range(len(data_matrix) - w - 1):
        window = data_matrix[i : i+w].flatten() # Flatten (10 days * 4 features = 40 inputs)
        X.append(window)
        y.append(target[i+w])
        
    X = np.array(X)
    y = np.array(y)
    
    # Dates & Prices for reconstruction
    # We need the dates corresponding to the PREDICTION targets
    valid_indices = range(w, len(data_matrix) - 1)
    dates = df['f_date'].iloc[valid_indices].values
    prices = df['close'].iloc[valid_indices].values # Price at time t (to calc t+1)
    
    # Split
    split = len(X) - CONFIG["TEST_SIZE"]
    
    return {
        "X_train": X[:split], "X_test": X[split:],
        "y_train": y[:split], "y_test": y[split:],
        "dates_test": dates[split:], 
        "last_price_train": prices[split-1],
        "prices_test": prices[split:] # Actual prices for comparison
    }

# =============================================================================
# ⚛️ 5. QUANTUM RESERVOIR ENGINE
# =============================================================================
class QuantumReservoir:
    def __init__(self, n_qubits, n_layers, scaling_factor=1.0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.scaling = scaling_factor
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        # Fixed random weights (The "Echo State")
        # We generate them ONCE and keep them fixed.
        np.random.seed(CONFIG["SEED"])
        self.weights = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3))

        # Define QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, inputs):
        """
        The Reservoir Circuit.
        1. Encoding: Maps classical data to Quantum State.
        2. Evolution: Fixed random unitary evolves the state (Chaos).
        3. Measurement: Extracts state info.
        """
        # Input Embedding (Angle Embedding)
        # We map PCA-reduced inputs to rotation angles
        qml.templates.AngleEmbedding(inputs * self.scaling, wires=range(self.n_qubits))
        
        # Reservoir Dynamics (Random Fixed Circuit)
        qml.templates.StronglyEntanglingLayers(self.weights, wires=range(self.n_qubits))
        
        # Readout (Reservoir States)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def transform(self, X):
        """
        Passes classical data through the Quantum Reservoir.
        X shape: (Samples, Features)
        Output: (Samples, n_qubits)
        """
        # 1. PCA Compression
        # If input dim > n_qubits, we must compress it first to fit into AngleEmbedding
        # Or we can use AmplitudeEncoding, but Angle is more robust for noise.
        if X.shape[1] != self.n_qubits:
            pca = PCA(n_components=self.n_qubits)
            # Fit on X (assuming X includes train/test or fit on train only in proper pipeline)
            # For simplicity in this class, we assume X is the full batch or handle PCA outside.
            # Here we assume inputs are already prepared or we do on-the-fly PCA.
            # Let's do robust on-the-fly PCA fit if train, transform if test? 
            # No, easier to expect X to be matched.
            # We will handle PCA in the main pipeline.
            pass

        # 2. Quantum Processing
        reservoir_states = []
        # Loop is efficient enough for lightning.qubit in batches
        for x_in in X:
            state = self.qnode(x_in)
            reservoir_states.append(state)
            
        return np.array(reservoir_states)

# =============================================================================
# ⚡ 6. OPTIMIZATION & TRAINING
# =============================================================================
def get_reservoir_states(X_train, X_test, n_qubits, n_layers, scaling):
    """
    Helper to prep data and run reservoir.
    """
    # 1. Scale & PCA
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    pca = PCA(n_components=n_qubits)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    
    # 2. MinMax to [0, 2pi]
    mm = MinMaxScaler(feature_range=(0, 2*np.pi))
    X_train_q = mm.fit_transform(X_train_pca)
    X_test_q = mm.transform(X_test_pca)
    
    # 3. Run Reservoir
    qrc = QuantumReservoir(n_qubits, n_layers, scaling)
    # This is the "Expensive" part, done once per trial
    res_train = qrc.transform(X_train_q)
    res_test = qrc.transform(X_test_q)
    
    return res_train, res_test

def objective(trial, data):
    # Hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 5)
    scaling = trial.suggest_float("scaling", 0.1, 2.0) # Controls non-linearity
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True) # Ridge Regularization
    
    # Generate Reservoir States (Dynamically for this trial configuration)
    # Note: In strict QRC, we might cache the reservoir, but here we tune reservoir DEPTH too.
    X_res_train, _ = get_reservoir_states(data["X_train"], data["X_test"], 
                                          CONFIG["N_QUBITS"], n_layers, scaling)
    
    # Cross-Validation (Classical) on the Reservoir States
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for t_idx, v_idx in tscv.split(X_res_train):
        xt, xv = X_res_train[t_idx], X_res_train[v_idx]
        yt, yv = data["y_train"][t_idx], data["y_train"][v_idx]
        
        # Readout Layer (Linear)
        readout = Ridge(alpha=alpha)
        readout.fit(xt, yt)
        preds = readout.predict(xv)
        scores.append(mean_squared_error(yv, preds))
        
    return np.mean(scores)

def main():
    input_file = "/content/nepse.json"
    if not os.path.exists(input_file):
        print("❌ Upload nepse.json")
        return
    
    # 1. Data
    df = process_data(input_file)
    data = create_windowed_dataset(df)
    logger.info(f"Data Ready. Train Window: {len(data['X_train'])} samples.")
    
    # 2. Optimize
    logger.info("🔍 Tuning Reservoir Hyperparameters...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, data), n_trials=CONFIG["N_TRIALS"], show_progress_bar=True)
    
    best = study.best_params
    logger.success(f"🏆 Best: {best}")
    
    # 3. Final Run
    logger.info("⚛️ Generating Final Reservoir States...")
    # We regenerate the states using the best structure found
    X_res_train, X_res_test = get_reservoir_states(
        data["X_train"], data["X_test"], 
        CONFIG["N_QUBITS"], best["n_layers"], best["scaling"]
    )
    
    # 4. Train Readout
    readout = Ridge(alpha=best["alpha"])
    readout.fit(X_res_train, data["y_train"])
    
    # 5. Predict
    pred_returns = readout.predict(X_res_test)
    
    # 6. Reconstruct Prices
    # P_t = P_{t-1} * exp(r_t)
    # data["last_price_train"] is the price right before the test set starts
    # data["prices_test"] contains the actual prices P_t. We need P_{t-1} for the test set.
    # The previous prices for the test set are: [last_train, test[0], test[1]...]
    
    prev_prices = np.concatenate(([data["last_price_train"]], data["prices_test"][:-1]))
    pred_prices = prev_prices * np.exp(pred_returns)
    
    # 7. Metrics
    mape = mean_absolute_percentage_error(data["prices_test"], pred_prices) * 100
    r2 = r2_score(data["prices_test"], pred_prices)
    rmse = np.sqrt(mean_squared_error(data["prices_test"], pred_prices))
    
    # Directional Accuracy
    actual_diff = np.diff(data["prices_test"])
    pred_diff = np.diff(pred_prices)
    da = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
    
    print("\n" + "═"*50)
    print("✨ QUANTUM RESERVOIR COMPUTING RESULTS ✨")
    print(f"MAPE: {mape:.3f}%")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Directional Accuracy: {da:.1f}%")
    print("═"*50)
    
    # 8. Plot
    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(data["dates_test"], data["prices_test"], label="Actual NEPSE", color="black", alpha=0.6)
    plt.plot(data["dates_test"], pred_prices, label="Quantum Reservoir Forecast", color="#884EA0", linewidth=2)
    plt.title("NEPSE Index: Quantum Reservoir Computing (QRC)")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(DIRS["plots"] / "qrc_result.png")
    plt.show()
    
    # Save
    pd.DataFrame({
        "Date": data["dates_test"],
        "Actual": data["prices_test"],
        "Predicted": pred_prices
    }).to_csv(DIRS["predictions"] / "qrc_preds.csv", index=False)
    
    shutil.make_archive("/content/qrc_results", 'zip', BASE_DIR)
    print("📦 Results Saved: /content/qrc_results.zip")

if __name__ == "__main__":
    main()



#     ⬇️ Installing Quantum Research Stack...
# 🚀 QRC Initialized. Mode: DEV | Reservoir Qubits: 4

# Data Ready. Train Window: 149 samples.

# 🔍 Tuning Reservoir Hyperparameters...

# Best trial: 4. Best value: 8.64312e-05: 100%
#  5/5 [00:04<00:00,  1.28it/s]
# 🏆 Best: {'n_layers': 3, 'scaling': 0.8428677372243605, 'alpha': 7.7278296301525105}

# ⚛️ Generating Final Reservoir States...


# ══════════════════════════════════════════════════
# ✨ QUANTUM RESERVOIR COMPUTING RESULTS ✨
# MAPE: 1.314%
# R²:   0.5529
# RMSE: 48.09
# Directional Accuracy: 51.7%
# ══════════════════════════════════════════════════