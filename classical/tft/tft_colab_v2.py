"""
Temporal Fusion Transformer (TFT) - Google Colab ready script
Single-cell runnable script. Saves model, scaler, plots and test data.

Usage in Colab:
- Upload /content/nepse.json
- Run the cell containing this script (it installs dependencies automatically)
"""

# Install required packages (quiet)
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "pytorch-lightning", "scikit-learn", "pandas", "matplotlib"])  # installs in Colab

# All imports
import os
import json
from datetime import datetime
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pl.seed_everything(42)

# --------------------- DATA PREPARATION ---------------------

def load_and_preprocess(json_path='/content/nepse.json', last_n=600, window_size=60):
    # load json (handles {"data": [...]} or list)
    with open(json_path, 'r') as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'data' in raw:
        rows = raw['data']
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError('Unsupported JSON structure')

    df = pd.DataFrame(rows)
    # find date and close columns
    date_col = None
    for c in df.columns:
        if c.lower() in ('f_date', 'date', 'datetime', 'time'):
            date_col = c
            break
    close_col = None
    for c in df.columns:
        if c.lower() in ('close', 'price', 'close_price'):
            close_col = c
            break
    if date_col is None or close_col is None:
        raise ValueError('Could not find required columns in JSON')

    df = df[[date_col, close_col]].copy()
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna().sort_values('date').reset_index(drop=True)

    if len(df) > last_n:
        df = df.tail(last_n).reset_index(drop=True)

    # scaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']]).flatten()

    # create sequences X: window_size, y: next value
    X = []
    y = []
    dates_y = []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size])
        dates_y.append(df['date'].iloc[i+window_size])
    X = np.array(X)  # shape (n_samples, window_size)
    y = np.array(y)
    dates_y = np.array(dates_y)

    # splits indexes based on n_samples
    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    # To ensure val uses last window of train + full val: start_val = n_train - window_size
    start_val = max(0, n_train - 0)  # we will include overlap by preparing datasets below
    # But sequences X are already with window included; so for val we want sequences starting at n_train - window_size
    start_val = max(0, n_train - window_size)
    end_val = n_train + n_val

    start_test = max(0, n_train + n_val - window_size)
    end_test = n

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[start_val:end_val]
    y_val = y[start_val:end_val]

    X_test = X[start_test:end_test]
    y_test = y[start_test:end_test]

    dates_test = dates_y[start_test:end_test]

    return {
        'df': df,
        'scaler': scaler,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'dates_test': dates_test,
        'window_size': window_size,
    }

# --------------------- MODEL COMPONENTS ---------------------

class GRN(nn.Module):
    """Gated Residual Network used in TFT-like architectures"""
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(output_size, output_size), nn.Sigmoid())
        if input_size == output_size:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: (..., input_size)
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        gated = self.gate(x)
        return gated * x + (1 - gated) * residual


class VariableSelection(nn.Module):
    """Simple variable selection network for univariate series (keeps extensibility)
    For single variable it reduces to a linear projection + GRN.
    """
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)
        self.grn = GRN(hidden_size, hidden_size, hidden_size, dropout)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        b, t, _ = x.shape
        x_flat = x.view(b * t, 1)
        h = self.proj(x_flat)
        h = self.grn(h)
        h = h.view(b, t, -1)
        return h


class TemporalFusionTransformer(nn.Module):
    """Simplified TFT-like model for univariate forecasting
    Contains: variable selection, LSTM encoder/decoder, attention, GRN fusion
    """
    def __init__(self,
                 input_size=60,
                 hidden_size=64,
                 lstm_layers=2,
                 attn_heads=4,
                 dropout=0.1,
                 output_size=1):
        super().__init__()
        self.window_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attn_heads = attn_heads
        self.dropout = dropout

        # variable selection / input projection
        self.var_select = VariableSelection(input_size, hidden_size, dropout=dropout)

        # encoder LSTM processes the encoder sequence
        self.encoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)

        # decoder LSTM will generate one-step predictions (we'll run 1-step decoder)
        # we use a small decoder that takes last encoder hidden as initial state
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout)

        # multi-head attention: queries from decoder, keys/values from encoder outputs
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_heads, dropout=dropout, batch_first=True)

        # final layers
        self.post_grn = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, output_size))

    def forward(self, x):
        # x: (batch, seq_len) or (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # variable selection projection
        v = self.var_select(x)  # (b, seq_len, hidden)

        # encoder
        enc_out, (h_n, c_n) = self.encoder_lstm(v)  # enc_out: (b, seq_len, hidden)

        # prepare decoder input: take last encoder timestep representation as decoder input
        # decode one step: use last encoder output as input to decoder
        dec_in = enc_out[:, -1:, :]
        dec_out, _ = self.decoder_lstm(dec_in, (h_n, c_n))  # dec_out: (b, 1, hidden)

        # attention: query=dec_out, key/value=enc_out
        # MultiheadAttention expects (b, seq_len, embed) when batch_first=True
        attn_out, attn_weights = self.attention(dec_out, enc_out, enc_out)

        # fusion
        fused = dec_out.squeeze(1) + attn_out.squeeze(1)
        fused = self.post_grn(fused)

        out = self.output_layer(fused)
        # out shape (b, output_size)
        return out

# --------------------- LIGHTNING MODULE ---------------------

class TFTLightning(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X).squeeze(1)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X).squeeze(1)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X).squeeze(1)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return {'y_hat': y_hat.detach().cpu().numpy(), 'y': y.detach().cpu().numpy()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# --------------------- HELPERS ---------------------

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# --------------------- METRICS & PLOTTING ---------------------

def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    dir_acc = float(np.mean(np.diff(y_true) * np.diff(y_pred) > 0) * 100)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Directional_Accuracy': dir_acc}

# --------------------- MAIN EXECUTION ---------------------

if __name__ == '__main__':
    # Parameters
    JSON_PATH = '/content/nepse.json'
    WINDOW = 60
    LAST_N = 600
    BATCH = 32
    HIDDEN = 64
    LAYERS = 2
    HEADS = 4
    DROPOUT = 0.1
    LR = 1e-3
    MAX_EPOCHS = 60

    print('\nLoading and preprocessing data...')
    data = load_and_preprocess(JSON_PATH, last_n=LAST_N, window_size=WINDOW)

    scaler = data['scaler']
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    dates_test = data['dates_test']

    print(f"Shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH)

    # create model
    model = TemporalFusionTransformer(input_size=WINDOW, hidden_size=HIDDEN, lstm_layers=LAYERS, attn_heads=HEADS, dropout=DROPOUT, output_size=1)
    lit = TFTLightning(model, lr=LR)

    # callbacks and trainer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'models/{timestamp}_tft'
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=model_dir, filename='best_model', save_top_k=1, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    # ensure `devices` is an int > 0 for CPUAccelerator
    devices = 1 if torch.cuda.is_available() else 1
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=[early_stop, checkpoint_callback], accelerator='auto', devices=devices, enable_progress_bar=True)

    print('\nStarting training...')
    trainer.fit(lit, train_loader, val_loader)

    # load best model
    best_path = checkpoint_callback.best_model_path
    print('\nBest checkpoint:', best_path)
    # Try to load checkpoint robustly. For Lightning .ckpt use load_from_checkpoint,
    # otherwise handle plain state_dict (.pt) and strip possible 'model.' prefixes.
    if best_path and os.path.exists(best_path):
        try:
            if best_path.endswith('.ckpt'):
                lit = TFTLightning.load_from_checkpoint(best_path, model=model, lr=LR)
                model = lit.model
            else:
                ckpt = torch.load(best_path, map_location='cpu')
                # checkpoint may contain nested 'state_dict'
                state = ckpt.get('state_dict', ckpt)
                # strip 'model.' prefix if present
                if any(k.startswith('model.') for k in state.keys()):
                    new_state = {k.replace('model.', '', 1): v for k, v in state.items()}
                else:
                    new_state = state
                model.load_state_dict(new_state)
            print('✓ Loaded best model weights')
        except Exception as e:
            print('Warning: failed to load via primary method:', str(e))
            # fallback: try to extract state_dict and strip prefixes
            ckpt = torch.load(best_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            new_state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v for k, v in state.items()}
            model.load_state_dict(new_state)
            print('✓ Loaded best model weights (fallback)')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {best_path}')

    # Test predictions
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            yhat = model(Xb.float()).squeeze(1).cpu().numpy()
            preds.append(yhat)
            trues.append(yb.numpy())
    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()

    # inverse scale
    preds_inv = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    trues_inv = scaler.inverse_transform(trues.reshape(-1,1)).flatten()

    metrics = compute_metrics(trues_inv, preds_inv)
    print('\nTest metrics:')
    for k,v in metrics.items():
        if k == 'MAPE' or k == 'Directional_Accuracy':
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")

    # save scaler
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # save test data
    np.savez(os.path.join(model_dir, 'test_data.npz'), y_true=trues_inv, y_pred=preds_inv, dates=dates_test)

    # plots
    # training/validation loss plot
    metrics_history = trainer.callback_metrics
    # trainer stores logged metrics but not full loss arrays by default; but the checkpoint callback and PL logger could be used. For simplicity, use trainer.logged_metrics if available.
    # Instead, load from trainer.lr_schedulers? To keep simple, plot epoch-wise metrics from trainer.logged_metrics not guaranteed.

    # Plot predictions
    plt.figure(figsize=(12,5))
    plt.plot(trues_inv, label='Actual')
    plt.plot(preds_inv, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted (Test)')
    plt.tight_layout()
    pred_plot = os.path.join(model_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_plot, dpi=150)
    plt.show()

    # Forecast next 30 days autoregressively
    last_seq = np.array(list(data['X_test'][-1]))  # scaled sequence
    future_scaled = []
    cur_seq = last_seq.copy()
    with torch.no_grad():
        for i in range(30):
            x = torch.FloatTensor(cur_seq).unsqueeze(0)  # (1, window)
            yhat = model(x).squeeze(1).cpu().numpy()[0]
            future_scaled.append(yhat)
            # slide
            cur_seq = np.concatenate([cur_seq[1:], np.array([yhat])])
    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1,1)).flatten()

    print('\nNext 30 days forecasts:')
    for i, val in enumerate(future, 1):
        print(f"{i:2d}: {val:.2f}")

    # Save model weights and artifacts
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_weights.pt'))
    print(f"\nSaved artifacts to {model_dir}")
    print('Training complete.')
