# Colab-ready single-cell script: Patch-Transformer (PatchTST) forecasting
# Paste this entire cell into Google Colab and run.
# Data path expected: /content/nepse.json
# Model and outputs saved to: /content/models

import os
import json
import math
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ----------------------------
# Config (adjustable choices)
# ----------------------------
data_path = "/content/nepse.json"
model_dir = "/content/models"
os.makedirs(model_dir, exist_ok=True)

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model / training hyperparams (choose from allowed ranges)
WINDOW = 60  # context length
PATCH_LEN = 8  # 8 or 16
STRIDE = PATCH_LEN // 2
D_MODEL = 128  # 64 or 128
N_HEADS = 8  # 4 or 8
ENC_LAYERS = 4  # 3-4
DIM_FF = 512  # 256-512
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 80  # 50-100
LR = 1e-4  # 1e-3 or 1e-4
OPTIM = "AdamW"  # "Adam" or "AdamW"
PATIENCE = 12  # early stopping patience

FORECAST_HORIZON = 30  # autoregressive forecast length

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# Utilities: data & sequences
# ----------------------------
def load_and_preprocess(path, n_last=600, feature="close"):
    with open(path, "r") as f:
        js = json.load(f)
    # expecting structure { "data": [ { "f_date": "...", "close": ... }, ... ] }
    df = pd.DataFrame(js.get("data", js))
    if "f_date" in df.columns:
        df["f_date"] = pd.to_datetime(df["f_date"])
        df = df.sort_values("f_date", ascending=True).reset_index(drop=True)
    else:
        raise ValueError("JSON must contain 'f_date' as date field.")
    if feature not in df.columns:
        raise ValueError(f"JSON must contain '{feature}' field.")
    series = df[feature].astype(float).values
    # use only last n_last points
    if len(series) < n_last:
        raise ValueError(f"Need at least {n_last} data points, got {len(series)}")
    series = series[-n_last:]
    return series, df.iloc[-n_last:].reset_index(drop=True)

def create_train_val_test_splits(series, window=WINDOW):
    """
    Splits raw series points into train/val/test segments (80/10/10),
    and then builds sequences so that:
      - validation sequences start with last `window` points of train
      - test sequences start with last `window` points of validation
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
    All arrays are numpy float32.
    """
    n = len(series)
    train_end = int(n * 0.8)  # index where train ends (exclusive for points)
    val_end = int(n * 0.9)

    # starts for sequences are chosen so that y = start + window
    def build_ranges(start_idx, end_point_idx):
        # want y indices from start_idx + window ... end_point_idx - 1
        starts = []
        s = start_idx
        max_start = end_point_idx - 1 - window  # inclusive
        while s <= max_start:
            starts.append(s)
            s += 1
        return starts

    train_starts = build_ranges(0, train_end)
    val_starts = build_ranges(train_end - window, val_end)
    test_starts = build_ranges(val_end - window, n)

    def build_xy(starts):
        X = []
        y = []
        for s in starts:
            x = series[s : s + window]
            target = series[s + window]
            X.append(x)
            y.append(target)
        if len(X) == 0:
            return np.empty((0, window), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train, y_train = build_xy(train_starts)
    X_val, y_val = build_xy(val_starts)
    X_test, y_test = build_xy(test_starts)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ----------------------------
# Patchify & Model
# ----------------------------
def patchify_tensor(x, patch_len=PATCH_LEN, stride=STRIDE):
    """
    x: (batch, seq_len) numpy or torch tensor
    returns: (batch, n_patches, patch_len)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    patches = x.unfold(dimension=1, size=patch_len, step=stride).contiguous()
    return patches

class PatchTST(nn.Module):
    def __init__(self, seq_len=WINDOW, patch_len=PATCH_LEN, stride=STRIDE,
                 d_model=D_MODEL, n_heads=N_HEADS, num_layers=ENC_LAYERS,
                 dim_feedforward=DIM_FF, dropout=DROPOUT):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True,
                                                   norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len) or (batch, seq_len, 1)
        if x.dim() == 3:
            x = x.squeeze(-1)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride).contiguous()
        emb = self.proj(patches)
        emb = emb + self.pos_emb
        enc = self.encoder(emb)
        enc = self.norm(enc)
        pooled = enc.mean(dim=1)
        out = self.head(pooled).squeeze(-1)
        return out

# ----------------------------
# Training & evaluation
# ----------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, opt_name=OPTIM,
                model_dir=model_dir, device=device, patience=PATIENCE):
    model = model.to(device)
    if opt_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        optimizer = Adam(model.parameters(), lr=lr)
    # Some torch versions don't accept 'verbose' in ReduceLROnPlateau; use safe fallback
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, verbose=True)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_val = float("inf")
    best_epoch = -1
    early_count = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                pred = model(xb)
                loss = nn.MSELoss()(pred, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs}  Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}")

        # early stopping
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            early_count = 0
            best_path = os.path.join(model_dir, "best_model_patchtst.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  New best model saved (val {best_val:.6f})")
        else:
            early_count += 1
            if early_count >= patience:
                print(f"Early stopping (epoch {epoch}). Best epoch: {best_epoch}")
                break

    best_model = model
    best_path = os.path.join(model_dir, "best_model_patchtst.pth")
    if os.path.exists(best_path):
        best_model.load_state_dict(torch.load(best_path, map_location=device))
    return best_model, history

def evaluate_model(model, dataloader, scaler=None, device=device):
    model = model.to(device)
    model.eval()
    preds = []
    trues = []
    last_inputs = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            out = model(xb)
            preds.append(out.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
            last_inputs.append(xb[:, -1].detach().cpu().numpy())
    preds = np.concatenate(preds).reshape(-1)
    trues = np.concatenate(trues).reshape(-1)
    last_inputs = np.concatenate(last_inputs).reshape(-1)

    if scaler is not None:
        preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
        trues_inv = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(-1)
        last_inputs_inv = scaler.inverse_transform(last_inputs.reshape(-1,1)).reshape(-1)
    else:
        preds_inv = preds
        trues_inv = trues
        last_inputs_inv = last_inputs

    rmse = math.sqrt(mean_squared_error(trues_inv, preds_inv))
    mae = mean_absolute_error(trues_inv, preds_inv)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((trues_inv - preds_inv) / np.where(trues_inv==0, np.nan, trues_inv))) * 100.0
        if np.isnan(mape):
            mape = np.nan
    r2 = r2_score(trues_inv, preds_inv)

    actual_dir = np.sign(trues_inv - last_inputs_inv)
    pred_dir = np.sign(preds_inv - last_inputs_inv)
    dir_match = (actual_dir == pred_dir)
    dir_acc = np.mean(dir_match.astype(float)) * 100.0

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "R2": r2,
        "DirectionalAcc(%)": dir_acc
    }
    return metrics, trues_inv, preds_inv

# ----------------------------
# Forecasting (autoregressive)
# ----------------------------
def forecast_future(model, last_window_scaled, steps=FORECAST_HORIZON, scaler=None, device=device):
    model = model.to(device)
    model.eval()
    seq = last_window_scaled.copy().astype(np.float32).tolist()
    preds_scaled = []
    with torch.no_grad():
        for _ in range(steps):
            x_in = torch.tensor(np.array(seq[-WINDOW:], dtype=np.float32)).unsqueeze(0).to(device)
            out = model(x_in).detach().cpu().numpy().reshape(-1)[0]
            preds_scaled.append(out)
            seq.append(float(out))
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    if scaler is not None:
        preds = scaler.inverse_transform(preds_scaled).reshape(-1)
    else:
        preds = preds_scaled.reshape(-1)
    return preds, preds_scaled.reshape(-1)

# ----------------------------
# Main: load, prepare, train, eval, forecast
# ----------------------------
def main():
    print("Device:", device)
    print("Loading data...")
    series, df_used = load_and_preprocess(data_path, n_last=600, feature="close")
    print("Series length used:", len(series))

    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_train_val_test_splits(series_scaled, window=WINDOW)
    print("Sequence counts - train/val/test:", len(X_train), len(X_val), len(X_test))

    scaler_path = os.path.join(model_dir, "scaler.save")
    joblib.dump(scaler, scaler_path)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = PatchTST(seq_len=WINDOW, patch_len=PATCH_LEN, stride=STRIDE,
                     d_model=D_MODEL, n_heads=N_HEADS, num_layers=ENC_LAYERS,
                     dim_feedforward=DIM_FF, dropout=DROPOUT)
    print(model)

    best_model, history = train_model(model, train_loader, val_loader,
                                     epochs=EPOCHS, lr=LR, opt_name=OPTIM,
                                     model_dir=model_dir, device=device, patience=PATIENCE)

    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="train_mse")
    plt.plot(history["val_loss"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    loss_curve_path = os.path.join(model_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, bbox_inches="tight")
    plt.close()

    metrics, trues_inv, preds_inv = evaluate_model(best_model, test_loader, scaler=scaler, device=device)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    test_out = np.vstack([trues_inv, preds_inv]).T
    test_pred_path = os.path.join(model_dir, "test_predictions.csv")
    np.savetxt(test_pred_path, test_out, delimiter=",", header="actual,predicted", comments="")

    plt.figure(figsize=(10,5))
    plt.plot(trues_inv, label="Actual")
    plt.plot(preds_inv, label="Predicted")
    plt.xlabel("Test sample index")
    plt.ylabel("Close price")
    plt.legend()
    actual_pred_path = os.path.join(model_dir, "actual_vs_pred.png")
    plt.savefig(actual_pred_path, bbox_inches="tight")
    plt.close()

    last_window_scaled = series_scaled[-WINDOW:]
    forecast_vals, forecast_scaled = forecast_future(best_model, last_window_scaled, steps=FORECAST_HORIZON, scaler=scaler, device=device)
    forecast_path = os.path.join(model_dir, "forecast.csv")
    np.savetxt(forecast_path, forecast_vals, delimiter=",", header="forecast", comments="")

    final_model_path = os.path.join(model_dir, "final_model_patchtst.pth")
    torch.save(best_model.state_dict(), final_model_path)

    np.savez(os.path.join(model_dir, "training_history.npz"), train_loss=np.array(history["train_loss"]), val_loss=np.array(history["val_loss"]))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model, scaler, predictions, plots, and forecast to", model_dir)
    print("Forecast (next steps):")
    print(forecast_vals)

if __name__ == "__main__":
    main()