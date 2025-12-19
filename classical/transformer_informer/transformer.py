"""
Transformer / Informer forecasting Colab script (single cell).
Paste entire file into one Colab cell and run (GPU recommended).

Data: /content/nepse.json  (expects {"data":[{"f_date":..., "close":...}, ...]})
Saves artifacts to /content/models/<timestamp>/

Toggle model type at top: MODEL_TYPE = "transformer" or "informer"
"""

# -------------------- INSTALL / IMPORTS --------------------
import sys, subprocess, os, math, json, pickle, random
from datetime import datetime
from pathlib import Path

# install required packages in Colab (no-op if already present)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "scikit-learn", "pandas", "matplotlib"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------- USER SETTINGS --------------------
MODEL_TYPE = "transformer"   # "transformer" or "informer"
DATA_PATH = "/content/nepse.json"
WINDOW = 60
LAST_N = 600
BATCH_SIZE = 32
D_MODEL = 64               # 64 or 128
N_HEADS = 4                # 4 or 8
ENC_LAYERS = 3             # 2-4
DEC_LAYERS = 2             # used in seq2seq if desired (not needed for encoder-only)
FFN_DIM = 256              # feed-forward dim
DROPOUT = 0.1
LR = 1e-3
EPOCHS = 60
PATIENCE = 10              # early stopping patience
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = Path("/content/models") / datetime.now().strftime("%Y%m%d_%H%M%S") / MODEL_TYPE
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- DATA LOADING & PREPROCESSING --------------------
def load_preprocess(data_path=DATA_PATH, last_n=LAST_N, window=WINDOW):
    # Load json (handles {"data": [...]} or list)
    with open(data_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        rows = raw["data"]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError("Unsupported JSON structure")
    df = pd.DataFrame(rows)
    # locate date and close
    date_col = None
    close_col = None
    for c in df.columns:
        if c.lower() in ("f_date", "date", "datetime", "time"):
            date_col = c
        if c.lower() in ("close", "price", "close_price"):
            close_col = c
    if date_col is None or close_col is None:
        raise ValueError("Could not find 'f_date' and 'close' columns")
    df = df[[date_col, close_col]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    if len(df) > last_n:
        df = df.tail(last_n).reset_index(drop=True)
    # scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["close"]]).flatten()
    # create sequences
    X, y, dates_y = [], [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
        y.append(scaled[i+window])
        dates_y.append(df["date"].iloc[i+window])
    X = np.array(X)    # (n_samples, window)
    y = np.array(y)
    dates_y = np.array(dates_y)
    # splits (time-ordered): train 80%, val 10%, test 10%
    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    # ensure val uses last WINDOW points from train -> start_val = n_train - window
    start_val = max(0, n_train - window)
    end_val = n_train + n_val
    # test uses last WINDOW points from val
    start_test = max(0, n_train + n_val - window)
    end_test = n
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[start_val:end_val], y[start_val:end_val]
    X_test, y_test = X[start_test:end_test], y[start_test:end_test]
    dates_test = dates_y[start_test:end_test]
    return {
        "df": df,
        "scaler": scaler,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "dates_test": dates_test
    }

# -------------------- HELPERS --------------------
def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # avoid divide by zero for mape
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-8, y_true))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "Directional_Accuracy": dir_acc}

# -------------------- POSITIONAL ENCODING --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd dimension handling
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# -------------------- TRANSFORMER (encoder-only) --------------------
class TransformerForecast(nn.Module):
    def __init__(self, window=WINDOW, d_model=D_MODEL, n_heads=N_HEADS, n_layers=ENC_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=1000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, activation="relu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # pooling and head
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2, 1))

    def forward(self, x):
        # x: (batch, window) or (batch, window, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, seq, _ = x.shape
        x = self.input_proj(x)  # (b, seq, d_model)
        x = self.pos_enc(x)
        enc = self.encoder(x)   # (b, seq, d_model)
        enc = self.norm(enc)
        # use last token representation
        last = enc[:, -1, :]    # (b, d_model)
        out = self.head(last)   # (b, 1)
        return out.squeeze(-1)

# -------------------- INFORMER (simplified) --------------------
class ProbSparseAttention(nn.Module):
    """
    Simplified ProbSparse attention approximation:
    - sample a subset of keys for each batch to compute attention scores (probabilistic sparsity)
    - combine with global average value as global context
    This is a light approximation for Informer-style attention for efficiency demonstration.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.05, sample_k=32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.sample_k = sample_k

    def forward(self, q, k, v, mask=None):
        # q,k,v: (b, seq_len, d_model)
        b, seq, _ = q.shape
        Q = self.q_proj(q).view(b, seq, self.n_heads, self.head_dim).transpose(1,2)  # (b, heads, seq, head_dim)
        K = self.k_proj(k).view(b, seq, self.n_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(v).view(b, seq, self.n_heads, self.head_dim).transpose(1,2)
        # sample subset of key indices (same for all queries in this batch to simplify)
        k_sample = min(self.sample_k, seq)
        idx = torch.randperm(seq)[:k_sample].to(q.device)
        K_sample = K[:, :, idx, :]   # (b, heads, k_sample, head_dim)
        V_sample = V[:, :, idx, :]
        # compute scores between Q and sampled K
        scores = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, heads, seq, k_sample)
        attn = torch.softmax(scores, dim=-1)
        context_sample = torch.matmul(attn, V_sample)  # (b, heads, seq, head_dim)
        # global context via average V
        V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, seq, -1)  # (b, heads, seq, head_dim)
        # combine sample-based and global context
        context = 0.8 * context_sample + 0.2 * V_mean
        context = context.transpose(1,2).contiguous().view(b, seq, self.d_model)
        out = self.out_proj(context)
        out = self.dropout(out)
        return out

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, ffn_dim=256, dropout=0.05, sample_k=32):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout=dropout, sample_k=sample_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (b, seq, d_model)
        z = self.attn(x, x, x)
        x = self.norm1(x + z)
        z2 = self.ff(x)
        x = self.norm2(x + z2)
        return x

class InformerForecast(nn.Module):
    """
    Simplified Informer:
    - input projection
    - distilling conv layers to compress sequence length
    - stacked InformerEncoderLayer
    - pooling and head
    """
    def __init__(self, window=WINDOW, d_model=D_MODEL, enc_layers=2, heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.05, distill=True):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.distill = distill
        self.conv_distill = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        ) if distill else None
        self.pos_enc = PositionalEncoding(d_model, max_len=1000)
        self.encoder_layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads=heads, ffn_dim=ffn_dim, dropout=dropout, sample_k=min(32, window)) for _ in range(enc_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model//2, 1))

    def forward(self, x):
        # x: (b, seq) or (b, seq, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, seq, _ = x.shape
        x = self.input_proj(x)        # (b, seq, d_model)
        if self.distill:
            # conv distill expects (b, d_model, seq)
            xd = x.permute(0,2,1)
            xd = self.conv_distill(xd)   # (b, d_model, seq//2)
            x = xd.permute(0,2,1)        # (b, seq2, d_model)
        x = self.pos_enc(x)
        for lyr in self.encoder_layers:
            x = lyr(x)
        x = self.norm(x)
        # use last token representation
        last = x[:, -1, :]
        out = self.head(last).squeeze(-1)
        return out

# -------------------- TRAIN / VALID / EVAL LOOP --------------------
def train_model(model, train_loader, val_loader, lr=LR, epochs=EPOCHS, patience=PATIENCE, save_dir=SAVE_DIR):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best_val = float("inf")
    best_epoch = -1
    train_losses = []
    val_losses = []
    no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        count = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE).float()
            yb = yb.to(DEVICE).float()
            optimizer.zero_grad()
            yhat = model(Xb)
            loss = F.mse_loss(yhat, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * Xb.size(0)
            count += Xb.size(0)
        train_loss = running / max(1, count)
        train_losses.append(train_loss)

        # validation
        model.eval()
        vrunning = 0.0
        vcount = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(DEVICE).float()
                yv = yv.to(DEVICE).float()
                ypv = model(Xv)
                vloss = F.mse_loss(ypv, yv)
                vrunning += vloss.item() * Xv.size(0)
                vcount += Xv.size(0)
        val_loss = vrunning / max(1, vcount)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # early stopping / save best
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            # save model state_dict
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # load best model
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=DEVICE))
    return model, {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch}

def evaluate_model(model, test_loader, scaler):
    model.to(DEVICE)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE).float()
            yhat = model(Xb).cpu().numpy().flatten()
            preds.append(yhat)
            trues.append(yb.numpy().flatten())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds_inv = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    trues_inv = scaler.inverse_transform(trues.reshape(-1,1)).flatten()
    m = metrics(trues_inv, preds_inv)
    return trues_inv, preds_inv, m

def forecast_future_autoregr(model, full_scaled_series, scaler, window=WINDOW, days=30):
    # full_scaled_series: 1D numpy scaled array (entire dataset)
    model.to(DEVICE)
    model.eval()
    cur = full_scaled_series[-window:].copy()
    fut_scaled = []
    with torch.no_grad():
        for i in range(days):
            x = torch.FloatTensor(cur).unsqueeze(0).to(DEVICE)  # (1, window)
            yhat = model(x).cpu().numpy().flatten()[0]
            fut_scaled.append(yhat)
            cur = np.concatenate([cur[1:], np.array([yhat])])
    fut = scaler.inverse_transform(np.array(fut_scaled).reshape(-1,1)).flatten()
    return fut, fut_scaled

# -------------------- RUN PIPELINE --------------------
def run_pipeline():
    print("Loading data...")
    data = load_preprocess(DATA_PATH, last_n=LAST_N, window=WINDOW)
    scaler = data["scaler"]
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    dates_test = data["dates_test"]

    print(f"Shapes: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")

    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE)

    # model selection
    if MODEL_TYPE.lower() == "transformer":
        print("Building Transformer model...")
        model = TransformerForecast(window=WINDOW, d_model=D_MODEL, n_heads=N_HEADS, n_layers=ENC_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT)
    elif MODEL_TYPE.lower() == "informer":
        print("Building Informer model (simplified)...")
        model = InformerForecast(window=WINDOW, d_model=D_MODEL, enc_layers=ENC_LAYERS, heads=N_HEADS, ffn_dim=FFN_DIM, dropout=0.05, distill=True)
    else:
        raise ValueError("MODEL_TYPE must be 'transformer' or 'informer'")

    print(f"Device: {DEVICE}")
    model.to(DEVICE)

    # train
    print("Training...")
    model, history = train_model(model, train_loader, val_loader, lr=LR, epochs=EPOCHS, patience=PATIENCE, save_dir=SAVE_DIR)

    # evaluate
    print("Evaluating on test set...")
    y_true, y_pred, met = evaluate_model(model, test_loader, scaler)
    print("\nTest metrics:")
    for k, v in met.items():
        if k in ("MAPE", "Directional_Accuracy"):
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")

    # plots: loss curves
    plt.figure(figsize=(8,4))
    plt.plot(history["train_losses"], label="train_loss")
    plt.plot(history["val_losses"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    loss_plot = SAVE_DIR / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=150)
    plt.show()

    # plot actual vs predicted
    plt.figure(figsize=(10,4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted (Test)")
    pred_plot = SAVE_DIR / "actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(pred_plot, dpi=150)
    plt.show()

    # save artifacts
    torch.save(model.state_dict(), SAVE_DIR / "best_model_weights.pt")
    with open(SAVE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    np.savez(SAVE_DIR / "test_preds.npz", y_true=y_true, y_pred=y_pred, dates=dates_test.astype("datetime64[ns]"))

    # forecasting next 30 days autoregressively (use full scaled series from data)
    full_scaled = scaler.transform(data["df"][["close"]]).flatten()
    print("\nForecasting next 30 days autoregressively...")
    future, future_scaled = forecast_future_autoregr(model, full_scaled, scaler, window=WINDOW, days=30)
    for i, v in enumerate(future, 1):
        print(f"{i:2d}: {v:.2f}")

    # save forecasts
    np.savez(SAVE_DIR / "future_forecast.npz", forecast=future, forecast_scaled=np.array(future_scaled))

    print(f"\nAll artifacts saved to: {SAVE_DIR}")
    print("Done.")

# -------------------- ENTRYPOINT --------------------
if __name__ == "__main__":
    run_pipeline()