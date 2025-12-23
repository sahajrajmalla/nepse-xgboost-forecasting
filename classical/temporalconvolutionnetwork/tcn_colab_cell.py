"""
Single Google Colab cell content for training a Temporal Convolutional Network (TCN)
for time-series forecasting (univariate index like NEPSE closing prices).

Instructions: Paste the entire contents of this file into one Google Colab code cell
and run. The dataset is loaded from `/content/nepse.json` by default — change the
`DATA_PATH` or `COLUMN_NAME` variables below if needed.

This script uses only standard libraries: TensorFlow, NumPy, Pandas, scikit-learn,
and Matplotlib. It implements a causal, dilated, residual TCN suitable for
one-step-ahead forecasting.
"""

# FORCE CPU in Colab (remove or comment out if you WANT GPU)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# User-changeable settings
# -----------------------------
DATA_PATH = '/content/nepse.json'  # path in Colab to the uploaded dataset
# If JSON doesn't work for you, change to '/content/nepse.csv' and adapt read_csv below
COLUMN_NAME = 'close'  # change this to the column name for closing prices in your file
USE_LAST_N = 600  # use only the latest 600 points
LOOKBACK = 30  # input window size
PRED_HORIZON = 1  # steps ahead to predict (1)
TEST_SIZE_RATIO = 0.2  # validation split ratio
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_series(path, column_name='Close', use_last_n=600):
    """Load a univariate series from a JSON or CSV file and return the latest
    `use_last_n` values in chronological order (oldest -> newest).
    """
    if path.endswith('.json'):
        # read json; result can be a dict or DataFrame depending on structure
        raw = pd.read_json(path, orient='records')
        if isinstance(raw, pd.DataFrame):
            df = raw
        else:
            try:
                df = pd.json_normalize(raw)
            except Exception:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # find a list of records inside the JSON structure
                records = None
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                            records = v
                            break
                if records is None:
                    raise ValueError('Unable to parse JSON file into records')
                df = pd.json_normalize(records)

        # if the DataFrame contains a nested 'data' column with lists/dicts,
        # normalize that column into a flat table (common layout in some APIs)
        if 'data' in df.columns:
            try:
                # if the 'data' cell contains a list of dicts in the first row
                first = df['data'].iloc[0]
                if isinstance(first, list):
                    df = pd.json_normalize(first)
                elif isinstance(first, dict):
                    # column may contain dict per row -> expand
                    df = pd.json_normalize(df['data'].tolist())
            except Exception:
                pass
    else:
        df = pd.read_csv(path)

    # robust, case-insensitive column detection
    cols_lower = {c.lower(): c for c in df.columns}
    if column_name not in df.columns:
        key = column_name.lower()
        if key in cols_lower:
            column_name = cols_lower[key]
        else:
            # try to find any column containing 'close'
            alt = [c for c in df.columns if 'close' in c.lower()]
            if len(alt) > 0:
                column_name = alt[0]
                print(f"Column name not found; using '{column_name}' instead.")
            else:
                # helpful debug output
                print('Available columns:', list(df.columns))
                raise ValueError(f"Column '{column_name}' not found in data.")

    # Keep only the timestamp and the value column if present
    # Ensure chronological order
    try:
        # support common date column names including 'f_date' from your file
        date_candidates = [c for c in df.columns if c.lower() in ('date', 'timestamp', 'f_date')]
        if len(date_candidates) > 0:
            date_col = date_candidates[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        else:
            # If no date column, assume the file is already chronological
            df = df.reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)

    series = pd.to_numeric(df[column_name], errors='coerce').dropna()
    if len(series) < use_last_n:
        raise ValueError(f"Not enough data points: found {len(series)}, required {use_last_n}.")

    # take the last `use_last_n` points and ensure chronological order
    series = series.values[-use_last_n:]
    return series.astype('float32')


def create_windows(series, lookback=30, horizon=1):
    """Create sliding windows X (lookback) and y (horizon steps ahead).
    Returns arrays X shape=(n_samples, lookback, 1) and y shape=(n_samples, 1).
    """
    X, y = [], []
    for i in range(len(series) - lookback - (horizon - 1)):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback + (horizon - 1)])
    X = np.array(X)
    y = np.array(y)
    # reshape to (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((-1, 1))
    return X, y


def fit_scaler_on_train(X_train_raw, y_train_raw):
    """Fit MinMaxScaler using only training data to avoid data leakage.
    For a univariate series, flatten the windows and include targets.
    """
    flat = np.concatenate([X_train_raw.reshape(-1), y_train_raw.reshape(-1)])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(flat.reshape(-1, 1))
    return scaler


def scale_windows(X, y, scaler):
    # scaler expects shape (-1,1)
    nsamples, ntimesteps, nfeat = X.shape
    X_flat = X.reshape(-1, 1)
    Xs = scaler.transform(X_flat).reshape(nsamples, ntimesteps, nfeat)
    ys = scaler.transform(y)
    return Xs, ys


def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    # 1st causal, dilated conv + ReLU
    conv1 = layers.Conv1D(filters, kernel_size, padding='causal',
                          dilation_rate=dilation_rate, activation='relu')(x)
    drop1 = layers.SpatialDropout1D(dropout_rate)(conv1)
    # 2nd causal, dilated conv (no activation here)
    conv2 = layers.Conv1D(filters, kernel_size, padding='causal',
                          dilation_rate=dilation_rate)(drop1)
    drop2 = layers.SpatialDropout1D(dropout_rate)(conv2)

    # Residual connection: if number of channels differs, use 1x1 conv
    if x.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding='same')(x)
    else:
        res = x

    out = layers.Activation('relu')(layers.add([res, drop2]))
    return out


def build_tcn_model(input_shape, num_filters=32, kernel_size=3,
                    dilations=(1, 2, 4, 8), dropout_rate=0.2):
    """Build a simple stacked residual TCN.
    - input_shape: (timesteps, features)
    - dilations: iterable of dilation rates per residual block
    """
    inputs = Input(shape=input_shape)
    x = inputs
    # initial 1x1 conv to increase dimensionality
    x = layers.Conv1D(num_filters, 1, padding='causal')(x)

    for d in dilations:
        x = residual_tcn_block(x, filters=num_filters, kernel_size=kernel_size,
                               dilation_rate=d, dropout_rate=dropout_rate)

    # global pooling and dense output for one-step prediction
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def main():
    print('Loading series from', DATA_PATH)
    series = load_series(DATA_PATH, column_name=COLUMN_NAME, use_last_n=USE_LAST_N)
    print('Series loaded — using latest', USE_LAST_N, 'points.')

    # Create windows first (raw values), then split to avoid leakage
    X_raw, y_raw = create_windows(series, lookback=LOOKBACK, horizon=PRED_HORIZON)
    n_samples = X_raw.shape[0]
    train_size = int(n_samples * (1 - TEST_SIZE_RATIO))

    X_train_raw = X_raw[:train_size]
    y_train_raw = y_raw[:train_size]
    X_val_raw = X_raw[train_size:]
    y_val_raw = y_raw[train_size:]

    print(f'Total windows: {n_samples}, Train: {X_train_raw.shape[0]}, Val: {X_val_raw.shape[0]}')

    # Fit scaler only on training raw values (flattened)
    scaler = fit_scaler_on_train(X_train_raw, y_train_raw)
    X_train, y_train = scale_windows(X_train_raw, y_train_raw, scaler)
    X_val, y_val = scale_windows(X_val_raw, y_val_raw, scaler)

    # Build model
    input_shape = (LOOKBACK, 1)
    model = build_tcn_model(input_shape, num_filters=32, kernel_size=3,
                            dilations=(1, 2, 4, 8), dropout_rate=0.2)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mae'])
    model.summary()

    # Callbacks
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=2
    )

    # Predict on validation set
    y_pred_scaled = model.predict(X_val)
    # inverse transform predictions and ground truth
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_val)

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print('\nEvaluation on validation set:')
    print(f'MSE: {mse:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'MAE: {mae:.6f}')

    # Plot actual vs predicted (validation)
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.flatten(), label='Actual')
    plt.plot(y_pred.flatten(), label='Predicted')
    plt.title('Actual vs Predicted (Validation set)')
    plt.xlabel('Sample')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training vs validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('\nModel and evaluation complete. The trained model is available as `model`.')


if __name__ == '__main__':
    main()
