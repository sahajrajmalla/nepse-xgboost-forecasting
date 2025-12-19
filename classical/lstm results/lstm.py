"""
LSTM index prediction pipeline (TensorFlow / Keras)

Usage example:
python lstm.py --data_path data.csv --date_col Date --value_col Close \
	--window_size 60 --epochs 50 --batch_size 32 --model_dir models/

This script follows best-practices for time-series forecasting:
- time-ordered train/validation split
- configurable scaler (MinMax or Standard)
- windowed sequences generation
- reproducible randomness
- model checkpointing, early stopping, LR reduction
- save scaler and model artifacts
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib

import sys
try:
	import tensorflow as tf
	from tensorflow.keras.models import Sequential, load_model
	from tensorflow.keras.layers import LSTM, Dense, Dropout
	from tensorflow.keras.callbacks import (
		ModelCheckpoint,
		EarlyStopping,
		ReduceLROnPlateau,
	)
except Exception as e:
	print("\nMissing or broken TensorFlow installation.")
	print("To install dependencies in a new virtual environment (recommended), run:")
	print("  python -m venv .venv")
	print("  .\\.venv\\Scripts\\Activate.ps1  # PowerShell")
	print("  python -m pip install --upgrade pip")
	print("  pip install -r requirements.txt\n")
	print("Original import error:\n", str(e))
	sys.exit(1)


@dataclass
class Config:
	data_path: str
	date_col: str
	value_col: str
	window_size: int = 60
	epochs: int = 50
	batch_size: int = 32
	val_split: float = 0.1
	test_split: float = 0.1
	scale: str = "minmax"  # or 'standard'
	model_dir: str = "models"
	seed: int = 42
	patience: int = 8
	max_samples: int = 600  # Limit dataset to 600 samples


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	tf.random.set_seed(seed)


def load_time_series(path: str, date_col: str, value_col: str, max_samples: int = 600) -> pd.DataFrame:
	# Robust loader: handles CSV, JSON (including top-level {"data": [...]})
	if path.endswith('.json'):
		with open(path, 'r', encoding='utf-8') as fh:
			raw = json.load(fh)
		if isinstance(raw, dict) and 'data' in raw and isinstance(raw['data'], list):
			df = pd.DataFrame(raw['data'])
		elif isinstance(raw, list):
			df = pd.DataFrame(raw)
		else:
			df = pd.json_normalize(raw)
	else:
		df = pd.read_csv(path)

	# Limit to max_samples
	if len(df) > max_samples:
		df = df.iloc[:max_samples].copy()

	# Normalize columns for flexible matching
	columns = list(df.columns)
	col_map = {c.lower(): c for c in columns}

	# Resolve date column: exact -> case-insensitive -> substring
	date_col_used = None
	if date_col in df.columns:
		date_col_used = date_col
	elif date_col.lower() in col_map:
		date_col_used = col_map[date_col.lower()]
	else:
		matches = [c for c in columns if date_col.lower() in c.lower()]
		if len(matches) == 1:
			date_col_used = matches[0]

	if date_col_used:
		df[date_col_used] = pd.to_datetime(df[date_col_used], errors="coerce")
		df = df.sort_values(by=date_col_used).reset_index(drop=True)
	else:
		df = df.reset_index(drop=True)

	# Resolve value column similarly: exact -> case-insensitive -> substring
	value_col_used = None
	if value_col in df.columns:
		value_col_used = value_col
	elif value_col.lower() in col_map:
		value_col_used = col_map[value_col.lower()]
	else:
		matches = [c for c in columns if value_col.lower() in c.lower()]
		if len(matches) == 1:
			value_col_used = matches[0]
		elif len(matches) > 1:
			# pick the best match (prefer exact 'close' or first occurrence)
			for preferred in ('close', 'close_price', 'price', 'last'):
				for c in matches:
					if preferred in c.lower():
						value_col_used = c
						break
				if value_col_used:
					break
			if not value_col_used:
				value_col_used = matches[0]

	if value_col_used is None:
		raise ValueError(
			f"Value column '{value_col}' not found in data. Available columns: {columns}"
		)

	# Coerce numeric values if possible
	df[value_col_used] = pd.to_numeric(df[value_col_used], errors='coerce')

	series = df[[value_col_used]].copy()
	# Keep original requested name for downstream code expectations
	series.columns = [value_col]
	return series


def get_scaler(kind: str):
	if kind == "minmax":
		return MinMaxScaler()
	elif kind == "standard":
		return StandardScaler()
	else:
		raise ValueError("Unknown scaler kind, choose 'minmax' or 'standard'")


def scale_series(scaler, series: pd.DataFrame) -> np.ndarray:
	return scaler.fit_transform(series.values.reshape(-1, 1))


def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
	X, y = [], []
	for i in range(window_size, len(data)):
		X.append(data[i - window_size : i, 0])
		y.append(data[i, 0])
	X_arr = np.array(X)
	y_arr = np.array(y)
	X_arr = X_arr.reshape((X_arr.shape[0], X_arr.shape[1], 1))
	return X_arr, y_arr


def time_train_val_test_split(data: np.ndarray, val_split: float, test_split: float):
	n = len(data)
	test_n = int(n * test_split)
	val_n = int(n * val_split)
	train_end = n - val_n - test_n
	train = data[:train_end]
	val = data[train_end : train_end + val_n]
	test = data[train_end + val_n :]
	return train, val, test


def build_lstm_model(input_shape: Tuple[int, int], units: int = 64, dropout: float = 0.2) -> tf.keras.Model:
	model = Sequential()
	model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
	model.add(Dropout(dropout))
	model.add(LSTM(units // 2))
	model.add(Dropout(dropout))
	model.add(Dense(1))
	model.compile(optimizer="adam", loss="mse")
	return model


def train_and_evaluate(cfg: Config) -> dict:
	set_seed(cfg.seed)

	os.makedirs(cfg.model_dir, exist_ok=True)

	series = load_time_series(cfg.data_path, cfg.date_col, cfg.value_col, cfg.max_samples)

	scaler = get_scaler(cfg.scale)
	scaled = scale_series(scaler, series)

	# Save scaler for later
	scaler_path = os.path.join(cfg.model_dir, "scaler.save")
	joblib.dump(scaler, scaler_path)

	# Split time-ordered data
	train_s, val_s, test_s = time_train_val_test_split(scaled, cfg.val_split, cfg.test_split)

	# Create sequences (train/val/test)
	X_train, y_train = create_sequences(train_s, cfg.window_size)
	X_val, y_val = create_sequences(np.concatenate([train_s[-cfg.window_size:], val_s]), cfg.window_size)
	X_test, y_test = create_sequences(np.concatenate([val_s[-cfg.window_size:], test_s]), cfg.window_size)

	model = build_lstm_model((X_train.shape[1], X_train.shape[2]), units=128, dropout=0.2)

	# Callbacks
	checkpoint_path = os.path.join(cfg.model_dir, "best_model.keras")
	callbacks = [
		ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
		EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
		ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
	]

	history = model.fit(
		X_train,
		y_train,
		validation_data=(X_val, y_val),
		epochs=cfg.epochs,
		batch_size=cfg.batch_size,
		callbacks=callbacks,
		verbose=2,
	)

	# Evaluate
	preds_test = model.predict(X_test).flatten()
	# invert scaling
	preds_test_inv = scaler.inverse_transform(preds_test.reshape(-1, 1)).flatten()
	y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

	rmse = np.sqrt(mean_squared_error(y_test_inv, preds_test_inv))
	mape = mean_absolute_percentage_error(y_test_inv, preds_test_inv)

	# Save final model and training metadata
	final_model_path = os.path.join(cfg.model_dir, "final_model.keras")
	model.save(final_model_path)

	meta = {
		"rmse": float(rmse),
		"mape": float(mape),
		"history": {k: [float(x) for x in v] for k, v in history.history.items()},
		"scaler_path": scaler_path,
		"final_model_path": final_model_path,
		"best_model_path": checkpoint_path,
	}
	with open(os.path.join(cfg.model_dir, "training_meta.json"), "w") as f:
		json.dump(meta, f, indent=2)

	# Save test arrays for later evaluation/plotting
	try:
		np.savez(os.path.join(cfg.model_dir, "test_data.npz"), X_test=X_test, y_test=y_test, scaled_full=scaled.flatten())
	except Exception:
		pass

	print(f"Test RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
	return meta


def parse_args() -> Config:
	p = argparse.ArgumentParser(description="Train an LSTM for index prediction")
	p.add_argument("--data_path", required=True)
	p.add_argument("--date_col", default="Date")
	p.add_argument("--value_col", default="Close")
	p.add_argument("--window_size", type=int, default=60)
	p.add_argument("--epochs", type=int, default=50)
	p.add_argument("--batch_size", type=int, default=32)
	p.add_argument("--val_split", type=float, default=0.1)
	p.add_argument("--test_split", type=float, default=0.1)
	p.add_argument("--scale", choices=["minmax", "standard"], default="minmax")
	p.add_argument("--model_dir", default="models")
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--patience", type=int, default=8)
	p.add_argument("--max_samples", type=int, default=600, help="Maximum number of samples to use from dataset")
	args = p.parse_args()
	return Config(
		data_path=args.data_path,
		date_col=args.date_col,
		value_col=args.value_col,
		window_size=args.window_size,
		epochs=args.epochs,
		batch_size=args.batch_size,
		val_split=args.val_split,
		test_split=args.test_split,
		scale=args.scale,
		model_dir=args.model_dir,
		seed=args.seed,
		patience=args.patience,
		max_samples=args.max_samples,
	)


if __name__ == "__main__":
	cfg = parse_args()
	meta = train_and_evaluate(cfg)
	print("Training complete. Metadata written to model folder.")

# === FINAL EVALUATION & PLOTTING (optional) ===
# This block is optional and will run only if a training_meta.json and test_data.npz exist.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

meta_path = os.path.join("models", "training_meta.json")
if not os.path.exists(meta_path):
    print("No training metadata found at 'models/training_meta.json' — skipping final evaluation.")
else:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    final_model_path = meta.get("final_model_path", os.path.join("models", "final_model.keras"))
    scaler_path = meta.get("scaler_path", os.path.join("models", "scaler.save"))
    test_npz = os.path.join(os.path.dirname(final_model_path), "test_data.npz")

    if not os.path.exists(final_model_path):
        print(f"Final model not found at '{final_model_path}' — skipping evaluation.")
    elif not os.path.exists(scaler_path):
        print(f"Scaler not found at '{scaler_path}' — cannot invert scaling, skipping evaluation.")
    elif not os.path.exists(test_npz):
        print(f"Test data not found at '{test_npz}' — skipping evaluation.")
    else:
        print(f"Loading model from: {final_model_path}")
        model = load_model(final_model_path)
        scaler = joblib.load(scaler_path)
        d = np.load(test_npz)
        X_test = d["X_test"]
        y_test = d["y_test"]
        scaled_full = d.get("scaled_full")

        # Predict and invert scaling
        y_pred_scaled = model.predict(X_test).reshape(-1, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Print scorecard
        print("\n" + "="*50)
        print("FINAL MODEL PERFORMANCE ON TEST DATA")
        print("="*50)
        print(f"RMSE  : {np.sqrt(mean_squared_error(y_true, y_pred)):6.2f} points")
        print(f"MAE   : {mean_absolute_error(y_true, y_pred):6.2f} points")
        print(f"MAPE  : {np.mean(np.abs((y_true - y_pred) / y_true)) * 100:6.2f}%")
        print(f"R²    : {r2_score(y_true, y_pred):.4f}")
        directional = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        print(f"Directional Accuracy: {directional:5.2f}%")
        print("="*50)

        # Plot actual vs predicted
        plt.figure(figsize=(12,6))
        plt.plot(y_true, label="Actual", color="blue")
        plt.plot(y_pred, label="Predicted", color="red", alpha=0.8)
        plt.title("Actual vs Predicted (Test)")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        out_fig = os.path.join(os.path.dirname(final_model_path), "prediction_vs_actual.png")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        print(f"Saved prediction plot to: {out_fig}")

        # Optional: forecast next 30 steps if we have the full scaled series
        if scaled_full is not None:
            last_60 = np.array(scaled_full[-60:]).reshape(60, 1)
            current_batch = last_60.reshape(1, last_60.shape[0], 1)
            future_preds = []
            for i in range(30):
                next_pred = model.predict(current_batch)[0]
                future_preds.append(next_pred[0])
                current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)
            future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
            print("\nNext 30-step forecast:")
            for i, pred in enumerate(future_preds, 1):
                print(f"Day +{i:2d}: {pred:8.2f}")

'''Test RMSE: 72.0909, MAPE: 0.0209
Training complete. Metadata written to model folder.
Loading model from: models/final_model.keras
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 164ms/step

==================================================
FINAL MODEL PERFORMANCE ON TEST DATA
==================================================
RMSE  :  72.09 points
MAE   :  56.94 points
MAPE  :   2.09%
R²    : 0.7492
Directional Accuracy: 50.85%
==================================================
Saved prediction plot to: models/prediction_vs_actual.png
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step

Next 30-step forecast:
Day + 1:  2613.38
Day + 2:  2616.31
Day + 3:  2619.17
Day + 4:  2622.10
Day + 5:  2625.15
Day + 6:  2628.31
Day + 7:  2631.55
Day + 8:  2634.83
Day + 9:  2638.12
Day +10:  2641.41
Day +11:  2644.66
Day +12:  2647.88
Day +13:  2651.03
Day +14:  2654.14
Day +15:  2657.18
Day +16:  2660.16
Day +17:  2663.08
Day +18:  2665.95
Day +19:  2668.75
Day +20:  2671.50
Day +21:  2674.20
Day +22:  2676.84
Day +23:  2679.43
Day +24:  2681.97
Day +25:  2684.46
Day +26:  2686.90
Day +27:  2689.29
Day +28:  2691.64
Day +29:  2693.94
Day +30:  2696.19'''