# =============================================================================
# 🔬 HYBRID QUANTUM-LSTM (QLSTM) FOR NEPSE FORECASTING
# Architecture: Feature Engineering -> LSTM Encoder -> Variational Quantum Circuit -> Linear Head
# Optimized for: Google Colab T4 GPU
# =============================================================================

# 1. INSTALL DEPENDENCIES (If not already installed)
import sys
import subprocess
import os

def install_packages():
    packages = ["pennylane", "torch", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "tqdm"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"⬇️ Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install_packages()

# 2. IMPORTS
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# ==========================================
# 🎛️ 3. GLOBAL CONFIGURATION & MODE SWITCH
# ==========================================
# OPTIONS: "DEV" (Fast debugging) | "PROD" (Research Paper Quality)
MODE = "PROD"

if MODE == "PROD":
    CONFIG = {
        "SEQ_LEN": 30,          # Lookback window
        "PRED_LEN": 1,
        "BATCH_SIZE": 32,
        "HIDDEN_SIZE": 64,      # LSTM units
        "NUM_LAYERS": 2,        # LSTM layers
        "N_QUBITS": 8,          # Quantum Width
        "Q_LAYERS": 3,          # Quantum Depth (Strongly Entangling)
        "DROPOUT": 0.2,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 100,          # Sufficient for convergence
        "PATIENCE": 15,         # Early stopping
        "DATA_LIMIT": None,     # Use ALL data
        "SEED": 42
    }
else:  # DEV MODE
    CONFIG = {
        "SEQ_LEN": 20,
        "PRED_LEN": 1,
        "BATCH_SIZE": 16,
        "HIDDEN_SIZE": 32,
        "NUM_LAYERS": 1,
        "N_QUBITS": 8,
        "Q_LAYERS": 1,          # Shallow circuit for speed
        "DROPOUT": 0.0,
        "LEARNING_RATE": 0.01,  # High LR to see movement quickly
        "EPOCHS": 3,            # Very quick run
        "PATIENCE": 2,
        "DATA_LIMIT": 600,      # Small subset
        "SEED": 42
    }

CONFIG["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG["SEED"])

print(f"🚀 RUNNING IN [{MODE}] MODE on {CONFIG['DEVICE']}")
print(f"ℹ️  Config: {json.dumps({k:v for k,v in CONFIG.items() if k!='DEVICE'}, indent=2)}")

# ==========================================
# 📊 4. FEATURE ENGINEERING PIPELINE
# ==========================================
def compute_technical_indicators(df):
    """Generates semantic market features for the LSTM."""
    df = df.copy()
    df = df.sort_values('f_date').reset_index(drop=True)

    # 1. RSI (Momentum)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD (Trend)
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26

    # 3. Bollinger Bands (Volatility)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['STD_20'] = df['close'].rolling(window=20).std()
    df['Upper_Band'] = (df['close'] - df['SMA_20']) / (df['STD_20'] + 1e-8) # Normalized Z-score style

    # 4. Log Returns (Stationarity)
    df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))

    # 5. Volume Trend
    df['Vol_Change'] = df['volume'].pct_change()

    # Cleanup
    df = df.replace([np.inf, -np.inf], 0).fillna(method='bfill').fillna(0)
    return df

def load_and_process_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    with open(filepath, 'r') as f:
        raw = json.load(f)

    data_list = raw['data'] if 'data' in raw else raw
    df = pd.DataFrame(data_list)

    # Convert types
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['f_date'] = pd.to_datetime(df['f_date'])
    df = df.dropna(subset=['close'])

    # Feature Engineering
    df = compute_technical_indicators(df)

    # MODE Truncation
    if CONFIG["DATA_LIMIT"] and len(df) > CONFIG["DATA_LIMIT"]:
        print(f"✂️  DEV MODE: Truncating data from {len(df)} to last {CONFIG['DATA_LIMIT']} rows.")
        df = df.tail(CONFIG["DATA_LIMIT"]).reset_index(drop=True)

    # Define Features
    feature_cols = ['close', 'RSI', 'MACD', 'Upper_Band', 'Log_Ret', 'Vol_Change']
    return df, feature_cols

# ==========================================
# ⚛️ 5. QUANTUM CIRCUIT
# ==========================================
dev = qml.device("lightning.qubit", wires=CONFIG["N_QUBITS"])

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    """
    Hybrid QLSTM Kernel:
    1. Encodes LSTM latent space into rotation angles.
    2. Entangles qubits to find non-linear correlations.
    3. Measures expectation values.
    """
    qml.templates.AngleEmbedding(inputs, wires=range(CONFIG["N_QUBITS"]))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(CONFIG["N_QUBITS"]))
    return [qml.expval(qml.PauliZ(i)) for i in range(CONFIG["N_QUBITS"])]

# ==========================================
# 🧠 6. HYBRID MODEL CLASS
# ==========================================
class HybridQLSTM(nn.Module):
    def __init__(self, input_dim):
        super(HybridQLSTM, self).__init__()

        self.hidden_size = CONFIG["HIDDEN_SIZE"]
        self.num_layers = CONFIG["NUM_LAYERS"]
        self.n_qubits = CONFIG["N_QUBITS"]

        # 1. Classical LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=CONFIG['DROPOUT'] if self.num_layers > 1 else 0
        )

        # 2. Classical -> Quantum Projection
        self.pre_quantum = nn.Linear(self.hidden_size, self.n_qubits)

        # 3. Variational Quantum Layer
        weight_shapes = {"weights": (CONFIG["Q_LAYERS"], self.n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # 4. Quantum -> Classical Head
        self.post_quantum = nn.Linear(self.n_qubits, 1)

    def forward(self, x):
        # LSTM
        # x: (Batch, Seq_Len, Features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(CONFIG["DEVICE"])
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(CONFIG["DEVICE"])

        out, _ = self.lstm(x, (h0, c0))

        # Extract last hidden state
        last_hidden = out[:, -1, :] # (Batch, Hidden)

        # Project to quantum angles (-pi to pi)
        q_input = torch.atan(self.pre_quantum(last_hidden)) * np.pi

        # Quantum Pass
        q_out = self.q_layer(q_input) # (Batch, n_qubits)

        # Final Prediction
        return self.post_quantum(q_out)

# ==========================================
# 📉 7. DATASET CLASS
# ==========================================
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = CONFIG["SEQ_LEN"]

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i : i + self.seq_len], self.y[i + self.seq_len]

# ==========================================
# 🚀 8. TRAINING ENGINE
# ==========================================
def run_pipeline():
    input_file = "/content/nepse.json"

    # 1. Load Data
    try:
        df, features = load_and_process_data(input_file)
        print(f"✅ Data Processed. Shape: {df.shape}")
    except Exception as e:
        print(e)
        return

    # 2. Scale Data (Important for LSTM & Quantum)
    scaler_X = MinMaxScaler(feature_range=(-1, 1)) # -1 to 1 better for Quantum angles
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_all = scaler_X.fit_transform(df[features])
    y_all = scaler_y.fit_transform(df[['close']])

    # 3. Train/Test Split (Chronological)
    train_size = int(len(df) * 0.80)
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]

    # 4. DataLoaders
    train_ds = StockDataset(X_train, y_train)
    test_ds = StockDataset(X_test, y_test)

    if len(train_ds) == 0 or len(test_ds) == 0:
        print("❌ Not enough data for Sequence Length. Reduce SEQ_LEN or add data.")
        return

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    # 5. Model Init
    model = HybridQLSTM(input_dim=len(features)).to(CONFIG["DEVICE"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # 6. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    wait = 0

    print(f"🔥 Starting Training for {CONFIG['EPOCHS']} Epochs...")
    pbar = tqdm(range(CONFIG["EPOCHS"]))

    for epoch in pbar:
        # -- TRAIN --
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(CONFIG["DEVICE"]), y_b.to(CONFIG["DEVICE"])
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # -- VALIDATE --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(CONFIG["DEVICE"]), y_b.to(CONFIG["DEVICE"])
                pred = model(X_b)
                val_loss += criterion(pred, y_b).item()

        avg_val = val_loss / len(test_loader)
        scheduler.step(avg_val)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        pbar.set_postfix({'Train': f"{avg_train:.5f}", 'Val': f"{avg_val:.5f}"})

        # Early Stopping
        if avg_val < best_loss:
            best_loss = avg_val
            wait = 0
            torch.save(model.state_dict(), "best_qlstm_nepse.pth")
        else:
            wait += 1
            if wait >= CONFIG["PATIENCE"]:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break

    # 7. Final Evaluation & Plotting
    print("\n📊 Generating Research Artifacts...")
    model.load_state_dict(torch.load("best_qlstm_nepse.pth"))
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(CONFIG["DEVICE"])
            p = model(X_b)
            preds.extend(p.cpu().numpy())
            actuals.extend(y_b.numpy())

    # Inverse Scaling
    preds_real = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
    actuals_real = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))

    # Metrics
    mape = mean_absolute_percentage_error(actuals_real, preds_real)
    r2 = r2_score(actuals_real, preds_real)
    rmse = np.sqrt(mean_squared_error(actuals_real, preds_real))

    print("-" * 30)
    print(f"✅ Final Metrics ({MODE}):")
    print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")
    print(f"R²  : {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)

    # Plot: Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"Convergence (Mode: {MODE})")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot: Prediction vs Actual
    plt.figure(figsize=(12, 6))
    # Plot last 200 points if in PROD, all if in DEV
    limit = len(preds_real) if MODE == "DEV" else min(200, len(preds_real))

    plt.plot(actuals_real[-limit:], label='Actual NEPSE', color='black', alpha=0.7)
    plt.plot(preds_real[-limit:], label='Hybrid QLSTM', color='#E91E63', linewidth=1.5)
    plt.title(f"Forecast Visualization ({MODE})")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# EXECUTE
if __name__ == "__main__":
    run_pipeline()


# ⬇️ Installing scikit-learn...
# 🚀 RUNNING IN [DEV] MODE on cuda
# ℹ️  Config: {
#   "SEQ_LEN": 20,
#   "PRED_LEN": 1,
#   "BATCH_SIZE": 16,
#   "HIDDEN_SIZE": 32,
#   "NUM_LAYERS": 1,
#   "N_QUBITS": 8,
#   "Q_LAYERS": 1,
#   "DROPOUT": 0.0,
#   "LEARNING_RATE": 0.01,
#   "EPOCHS": 3,
#   "PATIENCE": 2,
#   "DATA_LIMIT": 600,
#   "SEED": 42
# }
# ✂️  DEV MODE: Truncating data from 6394 to last 600 rows.
# ✅ Data Processed. Shape: (600, 15)
# /tmp/ipython-input-4048431369.py:126: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
#   df = df.replace([np.inf, -np.inf], 0).fillna(method='bfill').fillna(0)
# 🔥 Starting Training for 3 Epochs...
#  67%
#  2/3 [01:32<00:31, 31.70s/it, Train=0.16249, Val=0.25490]
# ⏹️  Early stopping at epoch 2

# 📊 Generating Research Artifacts...
# ------------------------------
# ✅ Final Metrics (DEV):
# MAPE: 0.0262 (2.62%)
# R²  : 0.5332
# RMSE: 82.8326
# ------------------------------