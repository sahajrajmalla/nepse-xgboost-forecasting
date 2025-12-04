# =============================================================================
# ⚛️ FINAL SOTA: DATA RE-UPLOADING QUANTUM REGRESSOR (DR-VQR)
# Architecture: "Quantum MLP" with Serial Data Encoding
# Fix: Corrected Batch Broadcasting in QNode
# =============================================================================

# 1. INSTALL
import sys, subprocess, os
def install():
    print("⬇️ Installing Quantum Research Stack...")
    packages = ["pennylane", "torch", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "optuna", "loguru", "tqdm"]
    for p in packages:
        try: __import__(p)
        except: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
install()

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import optuna
from loguru import logger
from pathlib import Path
from datetime import datetime
import shutil
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# 🎛️ 3. CONFIGURATION
# ==========================================
# "DEV" = Fast Debugging (2 mins)
# "PROD" = Full Research Run (20-30 mins)
MODE = "DEV" 

if MODE == "PROD":
    CONFIG = {
        "SEQ_LEN": 10,
        "N_QUBITS": 4,
        "EPOCHS": 100,
        "PATIENCE": 15,
        "BATCH_SIZE": 32,
        "N_TRIALS": 20,
        "LEARNING_RATE": 1e-3,
        "TRAIN_SIZE": 800,
        "SEED": 42
    }
else:
    CONFIG = {
        "SEQ_LEN": 5,
        "N_QUBITS": 4,
        "EPOCHS": 10,        # Short run to verify fix
        "PATIENCE": 3,
        "BATCH_SIZE": 16,
        "N_TRIALS": 3,
        "LEARNING_RATE": 5e-3,
        "TRAIN_SIZE": 200,
        "SEED": 42
    }

# Paths
BASE_DIR = Path("/content/drvqr_final")
DIRS = {k: BASE_DIR / k for k in ["models", "results", "predictions", "logs", "plots", "params"]}
for p in DIRS.values(): p.mkdir(parents=True, exist_ok=True)

# Logging
logger.remove()
logger.add(lambda msg: print(msg), format="{message}", level="INFO")

# Seeding
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 DR-VQR Initialized in [{MODE}] Mode on {DEVICE}")

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
    df = df.dropna()
    
    # Features
    df['Return'] = df['close'].pct_change().fillna(0)
    df['Vol'] = df['Return'].rolling(5).std().fillna(0)
    df['RSI'] = 100 - (100 / (1 + df['close'].diff().clip(lower=0).rolling(14).mean() / df['close'].diff().clip(upper=0).abs().rolling(14).mean()))
    df = df.fillna(0)
    
    # Regime Selection
    if len(df) > CONFIG["TRAIN_SIZE"] + 50: 
        df = df.tail(CONFIG["TRAIN_SIZE"] + 50).reset_index(drop=True)
    
    return df

class FlatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def get_loaders(df, batch_size):
    # Flattened window input
    feature_cols = ['close', 'Return', 'Vol', 'RSI']
    data = df[feature_cols].values
    target = df['close'].shift(-1).fillna(method='ffill').values
    
    X, y = [], []
    for i in range(len(data) - CONFIG["SEQ_LEN"]):
        X.append(data[i:i+CONFIG["SEQ_LEN"]].flatten())
        y.append(target[i+CONFIG["SEQ_LEN"]])
        
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split
    split = int(len(X) * 0.85)
    
    # Scaling
    scaler_x = MinMaxScaler(feature_range=(-1, 1)) 
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_train = scaler_x.fit_transform(X[:split])
    y_train = scaler_y.fit_transform(y[:split])
    
    X_test = scaler_x.transform(X[split:])
    y_test = scaler_y.transform(y[split:])
    
    train_loader = DataLoader(FlatDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(FlatDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Metadata
    y_test_raw = y[split:]
    test_dates = df['f_date'].iloc[split+CONFIG["SEQ_LEN"]:].values
    
    return train_loader, test_loader, scaler_y, test_dates, y_test_raw

# =============================================================================
# ⚛️ 5. FIXED QUANTUM CIRCUIT (BATCH-READY)
# =============================================================================
dev = qml.device("lightning.qubit", wires=CONFIG["N_QUBITS"])

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def reuploading_circuit(inputs, weights):
    """
    Data Re-uploading Circuit.
    inputs: (Batch, N_QUBITS)
    weights: (Layers, N_QUBITS, 3)
    """
    # Iterate over layers
    for l in range(weights.shape[0]):
        # 1. Data Encoding (Re-uploading)
        # AngleEmbedding broadcasts correctly over batch dimension
        qml.templates.AngleEmbedding(inputs, wires=range(CONFIG["N_QUBITS"]), rotation='Y')
        qml.templates.AngleEmbedding(inputs, wires=range(CONFIG["N_QUBITS"]), rotation='Z')
        
        # 2. Trainable Weights
        # We use a simplified StrongEntangling structure for the trainable part
        # qml.Rot is simpler to manage manually:
        for q in range(CONFIG["N_QUBITS"]):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        
        # 3. Entanglement Ring
        for q in range(CONFIG["N_QUBITS"]):
            qml.CNOT(wires=[q, (q+1) % CONFIG["N_QUBITS"]])
                
    return [qml.expval(qml.PauliZ(i)) for i in range(CONFIG["N_QUBITS"])]

# =============================================================================
# 🧠 6. HYBRID MODEL (MANUAL WEIGHTS)
# =============================================================================
class DRVQR(nn.Module):
    def __init__(self, input_dim, n_layers):
        super().__init__()
        
        # Compressor: High-dim Classical -> Low-dim Quantum
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, CONFIG["N_QUBITS"]),
            nn.Tanh() # Normalize to [-1, 1] for quantum embedding
        )
        
        # Quantum Weights (Manually managed to avoid TorchLayer bugs)
        # Shape: (Layers, Qubits, 3 parameters per Rot gate)
        self.q_weights = nn.Parameter(torch.randn(n_layers, CONFIG["N_QUBITS"], 3))
        
        # Head
        self.head = nn.Linear(CONFIG["N_QUBITS"], 1)

    def forward(self, x):
        # 1. Classical Compression
        # x: (Batch, Input_Dim) -> (Batch, N_Qubits)
        q_input = self.compressor(x) * np.pi # Scale to [-pi, pi]
        
        # 2. Quantum Pass (Explicit call to QNode)
        # This returns a list of tensors [ (Batch,), (Batch,), ... ]
        # We stick them to get (Batch, N_Qubits)
        q_out_list = reuploading_circuit(q_input, self.q_weights)
        q_out = torch.stack(q_out_list, dim=1).float()
        
        # 3. Classical Head
        return self.head(q_out)

# =============================================================================
# ⚡ 7. OPTIMIZATION
# =============================================================================
def objective(trial, df):
    params = {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    }
    
    train_loader, test_loader, _, _, _ = get_loaders(df, CONFIG["BATCH_SIZE"])
    if len(train_loader) == 0: raise optuna.TrialPruned()
    
    input_dim = CONFIG["SEQ_LEN"] * 4
    model = DRVQR(input_dim, params["n_layers"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    for epoch in range(3):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()
            
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                val_loss += criterion(model(X.to(DEVICE)), y.to(DEVICE)).item()
                
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        
    return val_loss

def train_final(df, best_params):
    logger.info(f"🚀 Training Final DR-VQR...")
    train_loader, test_loader, scaler_y, test_dates, y_test_raw = get_loaders(df, CONFIG["BATCH_SIZE"])
    
    input_dim = CONFIG["SEQ_LEN"] * 4
    model = DRVQR(input_dim, best_params["n_layers"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
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
            torch.save(model.state_dict(), DIRS["models"] / "best_drvqr.pth")
        else:
            wait += 1
            if wait >= CONFIG["PATIENCE"]: break
            
    # Eval
    model.load_state_dict(torch.load(DIRS["models"] / "best_drvqr.pth"))
    model.eval()
    preds = []
    
    with torch.no_grad():
        for X, y in test_loader:
            p = model(X.to(DEVICE)).cpu().numpy()
            preds.extend(p)
            
    # Inverse Scale
    pred_prices = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
    actual_prices = y_test_raw
    
    # Match lengths
    min_len = min(len(pred_prices), len(test_dates))
    pred_prices = pred_prices[:min_len]
    actual_prices = actual_prices[:min_len]
    test_dates = test_dates[:min_len]
    
    # Metrics
    mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
    r2 = r2_score(actual_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    
    # Directional Accuracy
    actual_diff = np.diff(actual_prices, axis=0)
    pred_diff = np.diff(pred_prices, axis=0)
    da = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100

    # Plot
    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(test_dates, actual_prices, label='Actual', color='black', alpha=0.6)
    plt.plot(test_dates, pred_prices, label='DR-VQR Forecast', color='#d62728', linewidth=1.5)
    plt.title(f"NEPSE DR-VQR Forecast (R²={r2:.3f})")
    plt.legend()
    plt.savefig(DIRS["plots"] / "forecast.png")
    plt.show()
    
    pd.DataFrame({'Date': test_dates, 'Actual': actual_prices.ravel(), 'Pred': pred_prices.ravel()}).to_csv(DIRS["predictions"]/"final.csv", index=False)
    
    return {"MAPE": mape, "R2": r2, "RMSE": rmse, "DA": da}

# =============================================================================
# 🟢 MAIN
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists("/content/nepse.json"):
        print("❌ Upload nepse.json")
    else:
        df = process_data("/content/nepse.json")
        
        logger.info("🔍 Tuning...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: objective(t, df), n_trials=CONFIG["N_TRIALS"])
        
        logger.success(f"🏆 Best: {study.best_params}")
        
        metrics = train_final(df, study.best_params)
        
        print("\n" + "═"*50)
        print("✨ DR-VQR FINAL RESULTS ✨")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"R²:   {metrics['R2']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"Dir Acc: {metrics['DA']:.1f}%")
        print("═"*50)
        
        shutil.make_archive("/content/drvqr_results", 'zip', BASE_DIR)
        print("📦 Results: /content/drvqr_results.zip")



# ⬇️ Installing Quantum Research Stack...
# 🚀 DR-VQR Initialized in [DEV] Mode on cpu

# 🔍 Tuning...

# 🏆 Best: {'n_layers': 3, 'lr': 0.0016348702723323575}

# 🚀 Training Final DR-VQR...

# Training: 100%
#  10/10 [00:28<00:00,  2.86s/it, Train=0.02992, Val=0.03214]


# ══════════════════════════════════════════════════
# ✨ DR-VQR FINAL RESULTS ✨
# MAPE: 2.64%
# R²:   0.1487
# RMSE: 82.56
# Dir Acc: 63.9%
# ══════════════════════════════════════════════════
# 📦 Results: /content/drvqr_results.zip