"""
N-BEATS Time Series Forecasting for NEPSE Stock Index
Complete standalone script for Google Colab
Based on PyTorch implementation with best practices
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import pickle

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages for Google Colab"""
    packages = [
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            __import__(package.split('[')[0].replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package} installed")

install_packages()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')


# ============================================================================
# N-BEATS MODEL ARCHITECTURE
# ============================================================================

class NBeatsBlock(nn.Module):
    """N-BEATS basic block (Trend or Seasonality)"""
    
    def __init__(self, input_size, output_size, hidden_size=128, n_layers=4):
        """
        Args:
            input_size: Number of input time steps
            output_size: Number of output time steps (forecast horizon)
            hidden_size: Number of hidden units (at least 128)
            n_layers: Number of dense layers
        """
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.fc = nn.Sequential(*layers)
        self.residual_connection = (input_size == output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        out = self.fc(x)
        
        if self.residual_connection:
            out = out + x
        
        return out


class NBeatsStack(nn.Module):
    """N-BEATS stack (multiple blocks)"""
    
    def __init__(self, input_size, output_size, n_blocks=2, hidden_size=128):
        """
        Args:
            input_size: Number of input time steps
            output_size: Number of output time steps
            n_blocks: Number of blocks in this stack
            hidden_size: Number of hidden units per block
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, output_size, hidden_size=hidden_size)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x):
        """Forward pass through stack - each block processes input independently"""
        stack_output = torch.zeros(x.size(0), self.blocks[0].fc[-1].out_features, device=x.device)
        for block in self.blocks:
            stack_output = stack_output + block(x)
        return stack_output


class NBeatsModel(nn.Module):
    """N-BEATS: Neural Basis Expansion Analysis with Trend and Seasonality"""
    
    def __init__(self, input_size=60, output_size=1, hidden_size=128, 
                 n_stacks=2, n_blocks=2, learning_rate=0.001):
        """
        Args:
            input_size: Sequence length (60)
            output_size: Forecast horizon (1)
            hidden_size: Hidden layer size (128+)
            n_stacks: Number of stacks per type (Trend, Seasonality)
            n_blocks: Number of blocks per stack
            learning_rate: Learning rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.learning_rate = learning_rate
        
        # Trend stacks
        self.trend_stacks = nn.ModuleList([
            NBeatsStack(input_size, output_size, n_blocks=n_blocks, hidden_size=hidden_size)
            for _ in range(n_stacks)
        ])
        
        # Seasonality stacks
        self.seasonality_stacks = nn.ModuleList([
            NBeatsStack(input_size, output_size, n_blocks=n_blocks, hidden_size=hidden_size)
            for _ in range(n_stacks)
        ])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        trend_output = torch.zeros(x.size(0), self.output_size, device=x.device)
        seasonality_output = torch.zeros(x.size(0), self.output_size, device=x.device)
        
        # Trend stacks
        for trend_stack in self.trend_stacks:
            trend_output = trend_output + trend_stack(x)
        
        # Seasonality stacks
        for seasonality_stack in self.seasonality_stacks:
            seasonality_output = seasonality_output + seasonality_stack(x)
        
        # Combine
        output = trend_output + seasonality_output
        
        return output


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataProcessor:
    """Handle data loading, preprocessing, and sequence generation"""
    
    def __init__(self, json_file="nepse.json", window_size=60, num_points=600):
        """
        Args:
            json_file: Path to JSON file
            window_size: Sliding window size
            num_points: Number of data points to use (last 600)
        """
        self.json_file = json_file
        self.window_size = window_size
        self.num_points = num_points
        self.scaler = MinMaxScaler()
        self.original_data = None
        self.scaled_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
    
    def load_data(self):
        """Load and preprocess NEPSE JSON data"""
        print(f"\n{'='*70}")
        print("STEP 1: LOADING DATA")
        print(f"{'='*70}")
        
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"File not found: {self.json_file}")
        
        # Load JSON
        with open(self.json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle structure: {"data": [...]} or direct array
        if isinstance(raw_data, dict) and 'data' in raw_data:
            data_list = raw_data['data']
        else:
            data_list = raw_data if isinstance(raw_data, list) else [raw_data]
        
        df = pd.DataFrame(data_list)
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Detect and extract close price column (case-insensitive)
        close_col = None
        for col in df.columns:
            if col.lower() in ['close', 'price', 'close_price']:
                close_col = col
                break
        
        if close_col is None:
            raise ValueError(f"Could not find close/price column. Available: {df.columns.tolist()}")
        
        # Detect date column (f_date, date, datetime)
        date_col = None
        for col in df.columns:
            if col.lower() in ['f_date', 'date', 'datetime', 'time']:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"Could not find date column. Available: {df.columns.tolist()}")
        
        print(f"  Date column: {date_col}")
        print(f"  Close column: {close_col}")
        
        # Create working dataframe
        df_work = df[[date_col, close_col]].copy()
        df_work.columns = ['date', 'close']
        
        # Convert to datetime
        df_work['date'] = pd.to_datetime(df_work['date'])
        df_work['close'] = pd.to_numeric(df_work['close'], errors='coerce')
        
        # Remove NaN
        df_work = df_work.dropna()
        
        # Sort ascending by date
        df_work = df_work.sort_values('date').reset_index(drop=True)
        
        # Take last num_points
        if len(df_work) > self.num_points:
            df_work = df_work.tail(self.num_points).reset_index(drop=True)
            print(f"✓ Using last {self.num_points} data points")
        else:
            print(f"⚠ Only {len(df_work)} rows available (requested {self.num_points})")
        
        self.original_data = df_work
        
        print(f"✓ Data shape: {df_work.shape}")
        print(f"  Date range: {df_work['date'].min()} to {df_work['date'].max()}")
        print(f"  Close range: {df_work['close'].min():.2f} to {df_work['close'].max():.2f}")
        
        return df_work
    
    def scale_data(self):
        """Apply MinMaxScaler"""
        print(f"\n{'='*70}")
        print("STEP 2: SCALING DATA")
        print(f"{'='*70}")
        
        close_values = self.original_data['close'].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(close_values).flatten()
        
        print(f"✓ Scaled data using MinMaxScaler")
        print(f"  Scaled range: [{self.scaled_data.min():.4f}, {self.scaled_data.max():.4f}]")
        
        return self.scaled_data
    
    def create_sequences(self):
        """Create sliding window sequences"""
        print(f"\n{'='*70}")
        print("STEP 3: CREATING SEQUENCES")
        print(f"{'='*70}")
        
        X, y = [], []
        
        for i in range(len(self.scaled_data) - self.window_size):
            X.append(self.scaled_data[i:i + self.window_size])
            y.append(self.scaled_data[i + self.window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✓ Created sequences")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y):
        """Time-ordered train/val/test split (80/10/10)"""
        print(f"\n{'='*70}")
        print("STEP 4: TRAIN/VAL/TEST SPLIT")
        print(f"{'='*70}")
        
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)
        
        # Time-ordered split
        self.X_train = X[:n_train]
        self.y_train = y[:n_train]
        
        self.X_val = X[n_train:n_train + n_val]
        self.y_val = y[n_train:n_train + n_val]
        
        self.X_test = X[n_train + n_val:]
        self.y_test = y[n_train + n_val:]
        
        print(f"✓ Split into time-ordered sets:")
        print(f"  Train: {len(self.X_train)} samples")
        print(f"  Val:   {len(self.X_val)} samples")
        print(f"  Test:  {len(self.X_test)} samples")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def get_dataloaders(self, batch_size=32):
        """Create PyTorch DataLoaders"""
        print(f"\n{'='*70}")
        print("STEP 5: CREATING DATALOADERS")
        print(f"{'='*70}")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(self.X_train)
        y_train_t = torch.FloatTensor(self.y_train)
        
        X_val_t = torch.FloatTensor(self.X_val)
        y_val_t = torch.FloatTensor(self.y_val)
        
        X_test_t = torch.FloatTensor(self.X_test)
        y_test_t = torch.FloatTensor(self.y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✓ Created dataloaders (batch_size={batch_size})")
        
        return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """Training loop with early stopping and learning rate scheduling"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate=0.001):
        """
        Args:
            model: N-BEATS model
            device: Device to train on
            learning_rate: Initial learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience = 0
        self.max_patience = 10
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch.unsqueeze(1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch.unsqueeze(1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # Learning rate scheduling
        self.scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.patience = 0
            return True  # Improved
        else:
            self.patience += 1
            return False
    
    def should_stop(self):
        """Check early stopping condition"""
        return self.patience >= self.max_patience
    
    def train(self, train_loader, val_loader, epochs=100, model_dir='models'):
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"STEP 6: TRAINING N-BEATS MODEL")
        print(f"{'='*70}")
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"{timestamp}_nbeats")
        os.makedirs(model_path, exist_ok=True)
        
        print(f"✓ Model checkpoint directory: {model_path}")
        print(f"✓ Device: {self.device.upper()}")
        print(f"✓ Training on {len(train_loader.dataset)} samples")
        print(f"✓ Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if self.should_stop():
                print(f"\n✓ Early stopping at epoch {epoch}")
                break
            
            # Save best model
            if val_loss:
                checkpoint_path = os.path.join(model_path, 'best_model.pt')
                torch.save(self.model.state_dict(), checkpoint_path)
        
        print(f"✓ Training complete!")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
        
        return model_path
    
    def test(self, test_loader, scaler):
        """Evaluate on test set and compute metrics"""
        print(f"\n{'='*70}")
        print("STEP 7: EVALUATING ON TEST SET")
        print(f"{'='*70}")
        
        self.model.eval()
        y_pred_list = []
        y_test_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                y_pred_list.append(y_pred.cpu().numpy())
                y_test_list.append(y_batch.numpy())
        
        # Concatenate
        y_pred_scaled = np.concatenate(y_pred_list).flatten()
        y_test_scaled = np.concatenate(y_test_list).flatten()
        
        # Inverse transform
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        y_test_direction = np.diff(y_test) > 0
        y_pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
        
        # Print metrics
        print(f"\n✓ TEST SET METRICS:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return y_test, y_pred, {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_loss_curves(train_losses, val_losses, model_path):
    """Plot training vs validation loss"""
    print(f"\n{'='*70}")
    print("STEP 8: PLOTTING LOSS CURVES")
    print(f"{'='*70}")
    
    plt.figure(figsize=(12, 5))
    
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (MSE)', fontsize=11)
    plt.title('N-BEATS: Training vs Validation Loss', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(model_path, 'loss_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved loss curves to: {plot_path}")
    plt.show()


def plot_predictions(y_test, y_pred, model_path):
    """Plot actual vs predicted test values"""
    print(f"\n{'='*70}")
    print("STEP 9: PLOTTING ACTUAL VS PREDICTED")
    print(f"{'='*70}")
    
    plt.figure(figsize=(14, 6))
    
    x_axis = np.arange(len(y_test))
    plt.plot(x_axis, y_test, label='Actual', linewidth=2, marker='o', markersize=4)
    plt.plot(x_axis, y_pred, label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.7)
    
    plt.xlabel('Test Sample Index', fontsize=11)
    plt.ylabel('NEPSE Close Price', fontsize=11)
    plt.title('N-BEATS: Actual vs Predicted (Test Set)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(model_path, 'actual_vs_predicted.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved prediction plot to: {plot_path}")
    plt.show()


# ============================================================================
# FORECASTING
# ============================================================================

def forecast_autoregressive(model, last_sequence, scaler, num_days=30, device='cpu'):
    """Autoregressive forecasting for next num_days"""
    print(f"\n{'='*70}")
    print(f"STEP 10: FORECASTING NEXT {num_days} DAYS")
    print(f"{'='*70}")
    
    model.eval()
    forecasts = []
    current_seq = last_sequence.copy()
    
    with torch.no_grad():
        for day in range(num_days):
            # Prepare input tensor
            X_input = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
            
            # Predict next value (scaled)
            y_pred_scaled = model(X_input).cpu().numpy()[0, 0]
            
            # Inverse transform
            y_pred = scaler.inverse_transform([[y_pred_scaled]])[0, 0]
            forecasts.append(y_pred)
            
            # Update sequence (sliding window)
            current_seq = np.append(current_seq[1:], y_pred_scaled)
    
    print(f"\n✓ FORECASTS FOR NEXT {num_days} DAYS:")
    print(f"{'Day':<6} {'Forecast':<15}")
    print("-" * 21)
    
    for i, forecast in enumerate(forecasts, 1):
        print(f"{i:<6} {forecast:>14.2f}")
    
    return np.array(forecasts)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("N-BEATS TIME SERIES FORECASTING FOR NEPSE STOCK INDEX")
    print("="*70)
    
    # ===== DATA LOADING AND PREPROCESSING =====
    data_processor = DataProcessor(
        json_file="nepse.json",
        window_size=60,
        num_points=600
    )
    
    # Load data
    df = data_processor.load_data()
    
    # Scale data
    data_processor.scale_data()
    
    # Create sequences
    X, y = data_processor.create_sequences()
    
    # Train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.split_data(X, y)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_processor.get_dataloaders(batch_size=32)
    
    # ===== MODEL CREATION =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("CREATING N-BEATS MODEL")
    print(f"{'='*70}")
    
    model = NBeatsModel(
        input_size=60,
        output_size=1,
        hidden_size=128,
        n_stacks=2,
        n_blocks=2,
        learning_rate=0.001
    )
    
    print(f"✓ Model created:")
    print(f"  Input size: 60")
    print(f"  Output size: 1")
    print(f"  Hidden size: 128")
    print(f"  Stacks per type: 2 (Trend + Seasonality)")
    print(f"  Blocks per stack: 2")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ===== TRAINING =====
    trainer = Trainer(model, device=device, learning_rate=0.001)
    model_path = trainer.train(train_loader, val_loader, epochs=100, model_dir='models')
    
    # ===== EVALUATION =====
    y_test_actual, y_test_pred, metrics = trainer.test(test_loader, data_processor.scaler)
    
    # ===== PLOTTING =====
    plot_loss_curves(trainer.train_losses, trainer.val_losses, model_path)
    plot_predictions(y_test_actual, y_test_pred, model_path)
    
    # ===== FORECASTING =====
    # Get last sequence from test data
    last_sequence = X_test[-1]
    forecasts = forecast_autoregressive(
        model, last_sequence, data_processor.scaler, num_days=30, device=device
    )
    
    # ===== SAVE ARTIFACTS =====
    print(f"\n{'='*70}")
    print("SAVING ARTIFACTS")
    print(f"{'='*70}")
    
    # Save model weights
    model_weights_path = os.path.join(model_path, 'model_weights.pt')
    torch.save(model.state_dict(), model_weights_path)
    print(f"✓ Saved model weights to: {model_weights_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_path, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(data_processor.scaler, f)
    print(f"✓ Saved scaler to: {scaler_path}")
    
    # Save test predictions
    test_results = {
        'y_test': y_test_actual,
        'y_pred': y_test_pred,
        'metrics': metrics,
        'forecasts': forecasts
    }
    
    test_results_path = os.path.join(model_path, 'test_results.pkl')
    with open(test_results_path, 'wb') as f:
        pickle.dump(test_results, f)
    print(f"✓ Saved test results to: {test_results_path}")
    
    # ===== COMPLETION MESSAGE =====
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Training complete. Best model saved to: {model_path}/")
    print(f"  Model weights: model_weights.pt")
    print(f"  Scaler: scaler.pkl")
    print(f"  Test results: test_results.pkl")
    print(f"  Loss curves: loss_curves.png")
    print(f"  Predictions: actual_vs_predicted.png")


if __name__ == "__main__":
    main()
