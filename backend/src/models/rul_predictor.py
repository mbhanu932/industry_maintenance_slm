#!/usr/bin/env python3
"""
RUL Prediction Model for NASA C-MAPSS Turbofan Engines

Implements a deep learning model for Remaining Useful Life (RUL) prediction
using LSTM networks with attention mechanism.

Based on research:
- "Long Short-Term Memory Network for Remaining Useful Life Estimation"
- "Deep Learning for Predictive Maintenance"

Usage:
    python src/models/rul_predictor.py --train --data data/processed/cmappss_fd001_train.parquet
    python src/models/rul_predictor.py --predict --model models/rul_predictor_fd001.h5
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
import pickle

# ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    logger.info("PyTorch available - using deep learning models")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - falling back to sklearn models")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RULModelConfig:
    """Configuration for RUL prediction model."""
    # Sequence parameters
    sequence_length: int = 30  # Number of time steps to look back
    stride: int = 1  # Stride for sliding window
    
    # Model architecture
    hidden_size: int = 100
    num_layers: int = 2
    dropout: float = 0.2
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # Feature selection
    sensor_columns: List[str] = None  # Will be set dynamically
    
    # Max RUL (clip RUL values to this maximum)
    max_rul: int = 125


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class TurbofanDataset(Dataset):
        """PyTorch Dataset for turbofan engine data."""
        
        def __init__(self, sequences: np.ndarray, targets: np.ndarray):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]
    
    
    class LSTMRULPredictor(nn.Module):
        """LSTM-based RUL prediction model with attention."""
        
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super(LSTMRULPredictor, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Attention layer
            self.attention = nn.Linear(hidden_size, 1)
            
            # Fully connected layers
            self.fc1 = nn.Linear(hidden_size, 50)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(50, 1)
        
        def forward(self, x):
            # LSTM forward pass
            lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size)
            
            # Attention mechanism
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            
            # Fully connected layers
            out = self.fc1(context_vector)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out.squeeze()


# ═══════════════════════════════════════════════════════════════════════════════
# RUL PREDICTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RULPredictor:
    """
    RUL Prediction model for turbofan engines.
    
    Supports both deep learning (LSTM) and traditional ML (sklearn) models.
    """
    
    def __init__(self, config: RULModelConfig = None):
        self.config = config or RULModelConfig()
        self.model = None
        self.scaler = MinMaxScaler()
        self.sensor_columns = None
        
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from dataframe."""
        if self.sensor_columns is None:
            # Auto-detect sensor columns
            self.sensor_columns = [col for col in df.columns if col.startswith('sensor_')]
            logger.info(f"Using {len(self.sensor_columns)} sensors: {self.sensor_columns[:5]}...")
        
        return df[self.sensor_columns].values
    
    def _create_sequences(self, features: np.ndarray, rul: np.ndarray, 
                         unit_numbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features: Sensor features (n_samples, n_features)
            rul: RUL values (n_samples,)
            unit_numbers: Unit IDs (n_samples,)
            
        Returns:
            sequences: Shape (n_sequences, sequence_length, n_features)
            targets: Shape (n_sequences,)
        """
        sequences = []
        targets = []
        
        unique_units = np.unique(unit_numbers)
        
        for unit in unique_units:
            unit_mask = unit_numbers == unit
            unit_features = features[unit_mask]
            unit_rul = rul[unit_mask]
            
            # Create sliding window sequences
            for i in range(len(unit_features) - self.config.sequence_length):
                seq_start = i
                seq_end = i + self.config.sequence_length
                
                sequences.append(unit_features[seq_start:seq_end])
                targets.append(unit_rul[seq_end])
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training or inference.
        
        Args:
            df: DataFrame with sensor data and RUL
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            sequences: Feature sequences
            targets: RUL targets
            unit_numbers: Unit IDs
        """
        logger.info("Preparing data...")
        
        # Extract features
        features = self._extract_features(df)
        
        # Normalize
        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        # Get RUL and clip to max
        rul = df['rul'].values
        rul = np.clip(rul, 0, self.config.max_rul)
        
        # Get unit numbers
        unit_numbers = df['unit_number'].values
        
        # Create sequences
        sequences, targets = self._create_sequences(features, rul, unit_numbers)
        
        logger.info(f"Created {len(sequences)} sequences")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Target shape: {targets.shape}")
        
        return sequences, targets, unit_numbers
    
    def build_model(self, input_size: int):
        """Build the RUL prediction model."""
        if TORCH_AVAILABLE:
            logger.info("Building LSTM model...")
            self.model = LSTMRULPredictor(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        else:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            logger.info("Building Gradient Boosting model (sklearn fallback)...")
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """
        Train the RUL prediction model.
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING RUL PREDICTION MODEL")
        logger.info("="*80 + "\n")
        
        # Prepare training data
        train_sequences, train_targets, _ = self.prepare_data(train_df, fit_scaler=True)
        
        # Build model
        input_size = train_sequences.shape[2]  # Number of features
        self.build_model(input_size)
        
        if TORCH_AVAILABLE and isinstance(self.model, LSTMRULPredictor):
            # PyTorch training
            self._train_pytorch(train_sequences, train_targets, val_df)
        else:
            # Sklearn training
            self._train_sklearn(train_sequences, train_targets)
    
    def _train_pytorch(self, train_sequences: np.ndarray, train_targets: np.ndarray, 
                       val_df: pd.DataFrame = None):
        """Train PyTorch LSTM model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {device}")
        
        self.model.to(device)
        
        # Create data loader
        train_dataset = TurbofanDataset(train_sequences, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            if val_df is not None:
                val_loss = self._validate_pytorch(val_df, criterion, device)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'models/best_rul_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {avg_train_loss:.4f}")
        
        logger.info("✓ Training complete")
    
    def _validate_pytorch(self, val_df: pd.DataFrame, criterion, device) -> float:
        """Validate PyTorch model."""
        val_sequences, val_targets, _ = self.prepare_data(val_df, fit_scaler=False)
        
        val_dataset = TurbofanDataset(val_sequences, val_targets)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_sequences, batch_targets in val_loader:
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = self.model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _train_sklearn(self, train_sequences: np.ndarray, train_targets: np.ndarray):
        """Train sklearn model (fallback)."""
        # Flatten sequences for sklearn
        train_flat = train_sequences.reshape(train_sequences.shape[0], -1)
        
        logger.info(f"Training sklearn model on {train_flat.shape} data...")
        self.model.fit(train_flat, train_targets)
        logger.info("✓ Training complete")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL for given data.
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            predictions: Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        sequences, _, _ = self.prepare_data(df, fit_scaler=False)
        
        if TORCH_AVAILABLE and isinstance(self.model, LSTMRULPredictor):
            # PyTorch prediction
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            
            with torch.no_grad():
                sequences_tensor = torch.FloatTensor(sequences).to(device)
                predictions = self.model(sequences_tensor).cpu().numpy()
        else:
            # Sklearn prediction
            sequences_flat = sequences.reshape(sequences.shape[0], -1)
            predictions = self.model.predict(sequences_flat)
        
        return predictions
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_df: Test data with true RUL values
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logger.info("\n--- Evaluating Model ---")
        
        # Get predictions
        predictions = self.predict(test_df)
        
        # Get true values
        _, true_rul, _ = self.prepare_data(test_df, fit_scaler=False)
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(true_rul, predictions))
        mae = mean_absolute_error(true_rul, predictions)
        
        # Compute score (asymmetric metric used in NASA challenge)
        # Errors are penalized more heavily for late predictions
        errors = predictions - true_rul
        score = np.sum(np.where(errors < 0, 
                               np.exp(-errors/13) - 1, 
                               np.exp(errors/10) - 1))
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'score': score
        }
        
        logger.info(f"RMSE: {rmse:.2f} cycles")
        logger.info(f"MAE: {mae:.2f} cycles")
        logger.info(f"NASA Score: {score:.2f}")
        
        return metrics
    
    def save(self, model_path: str):
        """Save trained model."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if TORCH_AVAILABLE and isinstance(self.model, LSTMRULPredictor):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'scaler': self.scaler,
                'sensor_columns': self.sensor_columns
            }, model_path)
        else:
            joblib.dump({
                'model': self.model,
                'config': self.config,
                'scaler': self.scaler,
                'sensor_columns': self.sensor_columns
            }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """Load trained model."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if TORCH_AVAILABLE:
            checkpoint = torch.load(model_path)
            self.config = checkpoint['config']
            self.scaler = checkpoint['scaler']
            self.sensor_columns = checkpoint['sensor_columns']
            
            # Rebuild model
            input_size = len(self.sensor_columns)
            self.build_model(input_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            checkpoint = joblib.load(model_path)
            self.config = checkpoint['config']
            self.model = checkpoint['model']
            self.scaler = checkpoint['scaler']
            self.sensor_columns = checkpoint['sensor_columns']
        
        logger.info(f"Model loaded from {model_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RUL prediction model")
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    parser.add_argument('--data', type=str, required=True, help='Data file path')
    parser.add_argument('--model', type=str, default='models/rul_predictor.pt', help='Model file path')
    parser.add_argument('--subset', type=str, default='FD001', help='Dataset subset')
    
    args = parser.parse_args()
    
    predictor = RULPredictor()
    
    if args.train:
        # Load training data
        train_df = pd.read_parquet(args.data)
        
        # Split into train/val
        units = train_df['unit_number'].unique()
        val_units = np.random.choice(units, size=int(len(units) * 0.2), replace=False)
        
        train_split = train_df[~train_df['unit_number'].isin(val_units)]
        val_split = train_df[train_df['unit_number'].isin(val_units)]
        
        # Train
        predictor.train(train_split, val_split)
        
        # Save
        predictor.save(args.model)
    
    elif args.predict or args.evaluate:
        # Load model
        predictor.load(args.model)
        
        # Load data
        test_df = pd.read_parquet(args.data)
        
        if args.evaluate:
            metrics = predictor.evaluate(test_df)
            print(f"\nMetrics: {metrics}")
        else:
            predictions = predictor.predict(test_df)
            print(f"\nPredictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:10]}")


if __name__ == "__main__":
    main()
