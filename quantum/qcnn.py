# =============================================================================
# 🧠 QUANTUM CONVOLUTIONAL NEURAL NETWORK (QCNN) - NEURIPS 2025 EDITION
# Target: NEPSE Index (Log-Return Forecasting)
# Methodology: 1D-QCNN Sliding Window -> Quantum Pooling -> Dense Head
# Key Advantage: Extracts local temporal patterns and quantum correlations.
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
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from loguru import logger
from pathlib import Path
from datetime import datetime
import shutil
from tqdm.notebook import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 🎛️ 3. CONFIGURATION
# ==========================================
MODE = "DEV"  # "DEV" for fast check, "PROD" for full training

CONFIG = {
    "SEQ_LEN": 8,           # Must be compatible with QCNN structure (e.g., powers of 2 or log reduction)
    "N_QUBITS": 8,          # Matches SEQ_LEN for direct embedding
    "BATCH_SIZE": 32,
    "EPOCHS": 100 if MODE == "PROD" else 5,
    "PATIENCE": 15 if MODE == "PROD" else 3,
    "N_TRIALS": 25 if MODE == "PROD" else 3,
    "LEARNING_RATE": 0.01,
    "SEED": 42
}

# Paths
BASE_DIR = Path("/content/qcnn_output")
DIRS = {k: BASE_DIR / k for k in ["models", "results", "predictions", "logs", "plots", "params"]}
for p in DIRS.values(): p.mkdir(parents=True, exist_ok=True)

# Logging
logger.remove()
logger.add(DIRS["logs"] / f"qcnn_{datetime.now():%Y%m%d_%H%M}.log", rotation="10 MB")
logger.add(lambda msg: print(msg), format="{message}", level="INFO")

# Seeding
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 QCNN Initialized. Mode: {MODE} | Device: {DEVICE}")

# =============================================================================
# 📊 4. DATA PIPELINE
# =============================================================================
def process_data(filepath):
    with open(filepath, 'r') as f: raw = json.load(f)
    data = raw['data'] if 'data' in raw else raw
    df = pd.DataFrame(data)
    
    for c in ['close', 'open', 'high', 'low', 'volume']: 
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['f_date'] = pd.to_datetime(df['f_date'])
    df = df.sort_values('f_date').reset_index(drop=True)
    
    # 1. Log Returns (Target)
    df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Volatility (Feature)
    df['Vol_5'] = df['Log_Ret'].rolling(5).std()
    
    # 3. RSI (Feature)
    delta = df['close'].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-8))))
    
    # Clean
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Regime Selection (Last ~3 years for PROD)
    if len(df) > 800:
        df = df.tail(800 if MODE == "PROD" else 200).reset_index(drop=True)
        
    return df

class QCNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def get_loaders(df, batch_size):
    # We use a sliding window of size SEQ_LEN
    # For QCNN, we often feed raw time-series chunks or PCA-reduced chunks
    # Here, we feed a window of 'Log_Ret' directly to look for patterns
    
    raw_series = df['Log_Ret'].values
    dates = df['f_date'].values
    prices = df['close'].values
    
    X, y = [], []
    for i in range(len(raw_series) - CONFIG["SEQ_LEN"]):
        X.append(raw_series[i : i + CONFIG["SEQ_LEN"]])
        y.append(raw_series[i + CONFIG["SEQ_LEN"]])
        
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split
    split = int(len(X) * 0.8)
    
    # Scaling: QCNN needs inputs in [0, pi] or [-pi, pi] usually
    # Robust Scaler first to handle outliers
    scaler_in = RobustScaler()
    X_train = scaler_in.fit_transform(X[:split])
    X_test = scaler_in.transform(X[split:])
    
    # MinMax to Quantum Range [0, PI]
    mm_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = mm_scaler.fit_transform(X_train)
    X_test = mm_scaler.transform(X_test)
    
    # Target Scaling
    scaler_out = MinMaxScaler(feature_range=(-1, 1))
    y_train = scaler_out.fit_transform(y[:split])
    y_test_raw = y[split:] # Keep raw for metric calc later
    y_test = scaler_out.transform(y[split:])
    
    train_loader = DataLoader(QCNNDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(QCNNDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Metadata for reconstruction
    meta = {
        "scaler_out": scaler_out,
        "dates_test": dates[CONFIG["SEQ_LEN"] + split:],
        "price_train_last": prices[CONFIG["SEQ_LEN"] + split - 1],
        "prices_test": prices[CONFIG["SEQ_LEN"] + split:]
    }
    
    return train_loader, test_loader, meta

# =============================================================================
# ⚛️ 5. QCNN CIRCUIT (CONV + POOL)
# =============================================================================
dev = qml.device("lightning.qubit", wires=CONFIG["N_QUBITS"])

def one_qubit_unitary(params, wire):
    qml.RX(params[0], wires=wire)
    qml.RY(params[1], wires=wire)
    qml.RZ(params[2], wires=wire)

def two_qubit_unitary(params, wires):
    qml.CRX(params[0], wires=wires)
    qml.CRY(params[1], wires=wires)
    qml.CRZ(params[2], wires=wires)

def conv_layer(params, wires):
    """Quantum Convolution: Applies 2-qubit gates to adjacent pairs."""
    # Even pairs
    for i in range(0, len(wires), 2):
        if i + 1 < len(wires):
            two_qubit_unitary(params[0:3], wires=[wires[i], wires[i+1]])
    # Odd pairs (Strided)
    for i in range(1, len(wires), 2):
        if i + 1 < len(wires):
            two_qubit_unitary(params[3:6], wires=[wires[i], wires[i+1]])
        else:
            # Wrap around boundary condition
            two_qubit_unitary(params[3:6], wires=[wires[i], wires[0]])

def pooling_layer(params, wires):
    """Quantum Pooling: Measures specific qubits to reduce dimensionality."""
    # We trace out half the qubits by controlling operations based on measurement (simulated via gates here)
    # Commonly implemented as parameterized gates condensing info onto half the wires
    for i in range(0, len(wires), 2):
        if i + 1 < len(wires):
            two_qubit_unitary(params[0:3], wires=[wires[i+1], wires[i]])
            qml.RX(params[3], wires=wires[i+1]) # Rotate the remaining qubit

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qcnn_circuit(inputs, weights):
    """
    Hierarchical QCNN Structure.
    Input (8 qubits) -> Conv -> Pool -> (4 qubits) -> Conv -> Pool -> (2 qubits) -> Dense
    """
    # Embedding
    qml.templates.AngleEmbedding(inputs, wires=range(CONFIG["N_QUBITS"]))
    
    # --- Block 1 ---
    # 8 Qubits -> 4 Qubits
    conv_layer(weights[0], wires=range(CONFIG["N_QUBITS"]))
    pooling_layer(weights[1], wires=range(CONFIG["N_QUBITS"]))
    
    # --- Block 2 ---
    # 4 Qubits (active on 1, 3, 5, 7) -> 2 Qubits
    active_wires_1 = list(range(1, CONFIG["N_QUBITS"], 2)) # [1, 3, 5, 7]
    conv_layer(weights[2], wires=active_wires_1)
    pooling_layer(weights[3], wires=active_wires_1)
    
    # --- Measurement ---
    # Measure the remaining active qubits (e.g., wires 3 and 7)
    # We measure PauliZ expectation to get continuous values
    active_wires_2 = [3, 7] 
    return [qml.expval(qml.PauliZ(w)) for w in active_wires_2]

# =============================================================================
# 🧠 6. HYBRID MODEL
# =============================================================================
class HybridQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Weights shape: 4 layers (Conv1, Pool1, Conv2, Pool2), 6 params each
        weight_shapes = {"weights": (4, 6)}
        self.qcnn_layer = qml.qnn.TorchLayer(qcnn_circuit, weight_shapes)
        
        # Classical Post-processing
        self.head = nn.Sequential(
            nn.Linear(2, 16), # 2 Quantum outputs
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # x: (Batch, 8) -> 8 time steps window
        q_out = self.qcnn_layer(x) # (Batch, 2)
        return self.head(q_out)

# =============================================================================
# ⚡ 7. TRAINING & OPTIMIZATION
# =============================================================================
def objective(trial, df):
    # Optimize Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    
    train_loader, test_loader, _ = get_loaders(df, batch_size)
    if len(train_loader) == 0: raise optuna.TrialPruned()
    
    model = HybridQCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Quick Loop
    for epoch in range(3):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X.to(DEVICE))
            loss = criterion(pred, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                val_loss += criterion(model(X.to(DEVICE)), y.to(DEVICE)).item()
        
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return val_loss

def train_final(df, best_params):
    logger.info(f"🚀 Training QCNN with: {best_params}")
    train_loader, test_loader, meta = get_loaders(df, best_params["batch_size"])
    
    model = HybridQCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    history = {'train': [], 'val': []}
    best_loss = float('inf')
    wait = 0
    
    pbar = tqdm(range(CONFIG["EPOCHS"]), desc="Training")
    for epoch in pbar:
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X.to(DEVICE))
                val_loss += criterion(pred, y.to(DEVICE)).item()
        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(test_loader)
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        
        scheduler.step(avg_val)
        pbar.set_postfix({'Train': f"{avg_train:.5f}", 'Val': f"{avg_val:.5f}"})
        
        if avg_val < best_loss:
            best_loss = avg_val
            wait = 0
            torch.save(model.state_dict(), DIRS["models"] / "best_qcnn.pth")
        else:
            wait += 1
            if wait >= CONFIG["PATIENCE"]: break

    # Evaluation
    model.load_state_dict(torch.load(DIRS["models"] / "best_qcnn.pth"))
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            p = model(X.to(DEVICE)).cpu().numpy()
            preds.extend(p)
            actuals.extend(y.numpy())
            
    # Inverse Transform Returns
    pred_ret = meta["scaler_out"].inverse_transform(np.array(preds).reshape(-1, 1))
    
    # Reconstruct Price: P_t = P_{t-1} * exp(r_t)
    prev_prices = np.concatenate(([meta["price_train_last"]], meta["prices_test"][:-1]))
    pred_prices = prev_prices * np.exp(pred_ret.ravel())
    
    # Metrics
    mape = mean_absolute_percentage_error(meta["prices_test"], pred_prices) * 100
    r2 = r2_score(meta["prices_test"], pred_prices)
    rmse = np.sqrt(mean_squared_error(meta["prices_test"], pred_prices))
    
    # Artifacts
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Val')
    plt.title("QCNN Learning Curve")
    plt.legend()
    plt.savefig(DIRS["plots"] / "convergence.png")
    plt.close()
    
    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(meta["dates_test"], meta["prices_test"], label="Actual", color="black", alpha=0.6)
    plt.plot(meta["dates_test"], pred_prices, label="QCNN Forecast", color="#FF3366", linewidth=1.5)
    plt.title(f"NEPSE QCNN Forecast (R²={r2:.3f})")
    plt.legend()
    plt.savefig(DIRS["plots"] / "forecast.png")
    plt.show()
    
    pd.DataFrame({
        "Date": meta["dates_test"],
        "Actual": meta["prices_test"],
        "Predicted": pred_prices
    }).to_csv(DIRS["predictions"] / "qcnn_preds.csv", index=False)
    
    return {"MAPE": mape, "R2": r2, "RMSE": rmse}

# =============================================================================
# 🟢 EXECUTION
# =============================================================================
if __name__ == "__main__":
    input_file = "/content/nepse.json"
    if os.path.exists(input_file):
        df = process_data(input_file)
        
        logger.info("🔍 Tuning QCNN...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: objective(t, df), n_trials=CONFIG["N_TRIALS"])
        
        logger.success(f"🏆 Best: {study.best_params}")
        
        metrics = train_final(df, study.best_params)
        
        print("\n" + "═"*50)
        print("✨ QCNN RESEARCH RESULTS ✨")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"R²:   {metrics['R2']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print("═"*50)
        
        shutil.make_archive("/content/qcnn_results", 'zip', BASE_DIR)
        print("📦 Results: /content/qcnn_results.zip")
    else:
        print("❌ Upload nepse.json")


# ⬇️ Installing Quantum Research Stack...
# 🚀 QCNN Initialized. Mode: DEV | Device: cuda

# 🔍 Tuning QCNN...

# 🏆 Best: {'lr': 0.00015956148621406653, 'batch_size': 32}

# 🚀 Training QCNN with: {'lr': 0.00015956148621406653, 'batch_size': 32}

# Training: 100%
#  5/5 [00:32<00:00,  6.44s/it, Train=0.15313, Val=0.35941]


# ══════════════════════════════════════════════════
# ✨ QCNN RESEARCH RESULTS ✨
# MAPE: 1.30%
# R²:   0.7551
# RMSE: 46.08
# ══════════════════════════════════════════════════
# 📦 Results: /content/qcnn_results.zip