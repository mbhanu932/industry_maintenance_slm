#!/usr/bin/env python3
"""
NASA C-MAPSS Turbofan Engine Degradation Dataset Preprocessor

Processes raw C-MAPSS data files and prepares them for model training and inference.

Dataset Structure:
- 4 subsets: FD001, FD002, FD003, FD004
- 21 sensor measurements
- 3 operational settings
- Unit number, cycle, and RUL (Remaining Useful Life)

Usage:
    python src/data/cmappss_preprocessor.py --input data/raw --output data/processed
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# C-MAPSS SENSOR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SENSOR_COLUMNS = [
    'T30',      # Total temperature at HPC outlet
    'T50',      # Total temperature at LPT outlet
    'P30',      # Total pressure at HPC outlet
    'Ps30',     # Static pressure at HPC outlet
    'phi',      # Ratio of fuel flow to Ps30
    'Nr',       # Corrected core speed
    'BPR',      # Bypass Ratio
    'htBleed',  # Bleed Enthalpy
    'W31',      # HPT coolant bleed
    'W32',      # LPT coolant bleed
    'T24',      # Total temperature at LPC outlet
    'T50',      # Total temperature at LPT outlet
    'P20',      # Total pressure in bypass-duct
    'P50',      # Total pressure at LPT outlet
    'Nf',       # Physical fan speed
    'Nc',       # Physical core speed
    'epr',      # Engine pressure ratio (P50/P2)
    'farB',     # Burner fuel-air ratio
    'Nf_dmd',   # Demanded fan speed
    'PCNfR_dmd', # Demanded corrected fan speed
    'smFan',    # Fan stall margin
    'smHPC',    # HPC stall margin
    'w22',      # 22nd sensor (varies by subset)
    'w25',      # 25th sensor (varies by subset)
]

# Common sensor indices (used across all subsets)
COMMON_SENSOR_INDICES = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

# Operational settings
OP_SETTING_COLUMNS = ['op_setting_1', 'op_setting_2', 'op_setting_3']


@dataclass
class CMAPSSConfig:
    """Configuration for C-MAPSS dataset subset."""
    name: str
    num_operational_conditions: int
    num_fault_modes: int
    num_sensors: int
    train_file: str
    test_file: str
    rul_file: str


# Dataset configurations
DATASET_CONFIGS = {
    'FD001': CMAPSSConfig(
        name='FD001',
        num_operational_conditions=1,
        num_fault_modes=1,
        num_sensors=21,
        train_file='train_FD001.txt',
        test_file='test_FD001.txt',
        rul_file='RUL_FD001.txt'
    ),
    'FD002': CMAPSSConfig(
        name='FD002',
        num_operational_conditions=6,
        num_fault_modes=1,
        num_sensors=21,
        train_file='train_FD002.txt',
        test_file='test_FD002.txt',
        rul_file='RUL_FD002.txt'
    ),
    'FD003': CMAPSSConfig(
        name='FD003',
        num_operational_conditions=1,
        num_fault_modes=2,
        num_sensors=21,
        train_file='train_FD003.txt',
        test_file='test_FD003.txt',
        rul_file='RUL_FD003.txt'
    ),
    'FD004': CMAPSSConfig(
        name='FD004',
        num_operational_conditions=6,
        num_fault_modes=2,
        num_sensors=21,
        train_file='train_FD004.txt',
        test_file='test_FD004.txt',
        rul_file='RUL_FD004.txt'
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class CMAPSSPreprocessor:
    """Preprocess NASA C-MAPSS turbofan engine degradation dataset."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, subset: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw C-MAPSS data files.
        
        Returns:
            train_df: Training data with all engine run-to-failure data
            test_df: Test data with truncated engine runs
            rul_df: True RUL values for test engines
        """
        config = DATASET_CONFIGS.get(subset)
        if not config:
            raise ValueError(f"Unknown subset: {subset}. Choose from {list(DATASET_CONFIGS.keys())}")
        
        logger.info(f"Loading {subset} dataset...")
        
        # Column names for C-MAPSS data
        column_names = ['unit_number', 'cycle'] + OP_SETTING_COLUMNS + [f'sensor_{i}' for i in range(1, 22)]
        
        # Load training data
        train_path = self.input_dir / config.train_file
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        
        train_df = pd.read_csv(train_path, sep=' ', header=None, names=column_names)
        train_df = train_df.dropna(axis=1, how='all')  # Remove empty columns
        
        # Load test data
        test_path = self.input_dir / config.test_file
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        test_df = pd.read_csv(test_path, sep=' ', header=None, names=column_names)
        test_df = test_df.dropna(axis=1, how='all')
        
        # Load RUL values
        rul_path = self.input_dir / config.rul_file
        if not rul_path.exists():
            raise FileNotFoundError(f"RUL file not found: {rul_path}")
        
        rul_df = pd.read_csv(rul_path, sep=' ', header=None, names=['rul'])
        rul_df = rul_df.dropna()
        
        logger.info(f"  Train: {len(train_df)} rows, {train_df['unit_number'].nunique()} engines")
        logger.info(f"  Test: {len(test_df)} rows, {test_df['unit_number'].nunique()} engines")
        logger.info(f"  RUL: {len(rul_df)} values")
        
        return train_df, test_df, rul_df
    
    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Remaining Useful Life for each cycle.
        
        RUL = max_cycle - current_cycle
        """
        logger.info("Computing RUL for each engine cycle...")
        
        # Get max cycle for each engine
        max_cycles = df.groupby('unit_number')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycle']
        
        # Merge and compute RUL
        df = df.merge(max_cycles, on='unit_number')
        df['rul'] = df['max_cycle'] - df['cycle']
        df = df.drop('max_cycle', axis=1)
        
        return df
    
    def normalize_sensors(self, df: pd.DataFrame, sensor_columns: List[str], 
                         scaler: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize sensor readings using min-max scaling.
        
        Returns:
            df: DataFrame with normalized sensors
            scaler: Dictionary with min/max values for each sensor
        """
        logger.info("Normalizing sensor readings...")
        
        if scaler is None:
            # Compute min/max from training data
            scaler = {}
            for col in sensor_columns:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    # Handle constant sensors
                    if min_val == max_val:
                        scaler[col] = {'min': 0, 'max': 1, 'constant': True}
                        df[col] = 0.5
                    else:
                        scaler[col] = {'min': min_val, 'max': max_val, 'constant': False}
                        df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            # Apply existing scaler
            for col in sensor_columns:
                if col in df.columns and col in scaler:
                    scale_info = scaler[col]
                    if scale_info['constant']:
                        df[col] = 0.5
                    else:
                        df[col] = (df[col] - scale_info['min']) / (scale_info['max'] - scale_info['min'])
        
        return df, scaler
    
    def add_health_index(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Compute health index based on sensor degradation patterns.
        
        Health index ranges from 0 (healthy) to 1 (degraded)
        """
        logger.info("Computing health index...")
        
        # Group by unit and compute rolling statistics
        df['health_index'] = 0.0
        
        for unit in df['unit_number'].unique():
            unit_mask = df['unit_number'] == unit
            unit_data = df[unit_mask].copy()
            
            # Compute degradation score based on sensor deviations
            degradation_scores = []
            for sensor in sensor_columns[:14]:  # Use first 14 sensors
                if sensor in unit_data.columns:
                    initial_val = unit_data[sensor].iloc[:10].mean()
                    current_val = unit_data[sensor]
                    deviation = abs(current_val - initial_val).rolling(window=5, min_periods=1).mean()
                    degradation_scores.append(deviation)
            
            if degradation_scores:
                health_index = pd.concat(degradation_scores, axis=1).mean(axis=1)
                health_index = (health_index - health_index.min()) / (health_index.max() - health_index.min() + 1e-8)
                df.loc[unit_mask, 'health_index'] = health_index.values
        
        return df
    
    def add_risk_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify engine risk level based on RUL and health index.
        
        Risk levels:
        - NORMAL: RUL > 100 cycles
        - ATTENTION: RUL 50-100 cycles
        - HIGH: RUL 20-50 cycles
        - CRITICAL: RUL < 20 cycles
        """
        logger.info("Classifying risk levels...")
        
        conditions = [
            (df['rul'] > 100),
            (df['rul'].between(50, 100)),
            (df['rul'].between(20, 50)),
            (df['rul'] < 20)
        ]
        choices = ['NORMAL', 'ATTENTION', 'HIGH', 'CRITICAL']
        
        df['risk_level'] = np.select(conditions, choices, default='NORMAL')
        
        return df
    
    def add_anomaly_score(self, df: pd.DataFrame, sensor_columns: List[str]) -> pd.DataFrame:
        """
        Compute anomaly score based on sensor deviation from normal operation.
        
        Anomaly score ranges from 0 (normal) to 1 (highly anomalous)
        """
        logger.info("Computing anomaly scores...")
        
        # For each engine, compare current sensor values to initial values
        df['anomaly_score'] = 0.0
        
        for unit in df['unit_number'].unique():
            unit_mask = df['unit_number'] == unit
            unit_data = df[unit_mask].copy()
            
            # Compute baseline (first 20 cycles)
            baseline = unit_data.iloc[:20][sensor_columns[:14]].mean()
            
            # Compute deviation
            deviations = []
            for i, sensor in enumerate(sensor_columns[:14]):
                if sensor in unit_data.columns:
                    current = unit_data[sensor]
                    baseline_val = baseline.iloc[i] if i < len(baseline) else 0
                    deviation = abs(current - baseline_val) / (abs(baseline_val) + 1e-8)
                    deviations.append(deviation)
            
            if deviations:
                anomaly_score = pd.concat(deviations, axis=1).mean(axis=1)
                anomaly_score = anomaly_score.clip(0, 1)
                df.loc[unit_mask, 'anomaly_score'] = anomaly_score.values
        
        # Classify anomaly status
        df['anomaly_status'] = df['anomaly_score'].apply(
            lambda x: 'ANOMALOUS' if x > 0.5 else 'NORMAL'
        )
        
        return df
    
    def add_failure_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate failure probability for 24h and 48h windows.
        
        Based on RUL and health indicators.
        """
        logger.info("Estimating failure probabilities...")
        
        # Simple heuristic: failure probability increases as RUL decreases
        max_rul = df['rul'].max()
        
        df['failure_probability_24h'] = 1.0 / (1.0 + np.exp((df['rul'] - 10) / 5.0))
        df['failure_probability_48h'] = 1.0 / (1.0 + np.exp((df['rul'] - 20) / 10.0))
        
        # Clip probabilities
        df['failure_probability_24h'] = df['failure_probability_24h'].clip(0, 1)
        df['failure_probability_48h'] = df['failure_probability_48h'].clip(0, 1)
        
        return df
    
    def preprocess_dataset(self, subset: str = 'FD001', 
                          save_intermediate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Full preprocessing pipeline for C-MAPSS dataset.
        
        Steps:
        1. Load raw data
        2. Compute RUL
        3. Normalize sensors
        4. Add health indicators
        5. Add risk levels
        6. Add anomaly scores
        7. Add failure probabilities
        8. Save processed data
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PREPROCESSING C-MAPSS {subset} DATASET")
        logger.info(f"{'='*80}\n")
        
        # Load raw data
        train_df, test_df, rul_df = self.load_raw_data(subset)
        
        # Get sensor columns
        sensor_columns = [col for col in train_df.columns if col.startswith('sensor_')]
        logger.info(f"Found {len(sensor_columns)} sensor columns")
        
        # Process training data
        logger.info("\n--- Processing Training Data ---")
        train_df = self.compute_rul(train_df)
        train_df, scaler = self.normalize_sensors(train_df, sensor_columns)
        train_df = self.add_health_index(train_df, sensor_columns)
        train_df = self.add_risk_level(train_df)
        train_df = self.add_anomaly_score(train_df, sensor_columns)
        train_df = self.add_failure_probability(train_df)
        
        # Process test data
        logger.info("\n--- Processing Test Data ---")
        test_df = self.compute_rul(test_df)
        test_df, _ = self.normalize_sensors(test_df, sensor_columns, scaler)
        test_df = self.add_health_index(test_df, sensor_columns)
        test_df = self.add_risk_level(test_df)
        test_df = self.add_anomaly_score(test_df, sensor_columns)
        test_df = self.add_failure_probability(test_df)
        
        # Save processed data
        if save_intermediate:
            output_file = self.output_dir / f'cmappss_{subset.lower()}_processed.parquet'
            logger.info(f"\nSaving processed data to {output_file}")
            
            # Save train and test separately
            train_output = self.output_dir / f'cmappss_{subset.lower()}_train.parquet'
            test_output = self.output_dir / f'cmappss_{subset.lower()}_test.parquet'
            
            train_df.to_parquet(train_output, index=False)
            test_df.to_parquet(test_output, index=False)
            rul_df.to_parquet(self.output_dir / f'cmappss_{subset.lower()}_rul.parquet', index=False)
            
            logger.info(f"  Saved: {train_output}")
            logger.info(f"  Saved: {test_output}")
            logger.info(f"  Saved: {self.output_dir / f'cmappss_{subset.lower()}_rul.parquet'}")
        
        # Summary statistics
        logger.info(f"\n--- Summary ---")
        logger.info(f"Training engines: {train_df['unit_number'].nunique()}")
        logger.info(f"Test engines: {test_df['unit_number'].nunique()}")
        logger.info(f"Total training cycles: {len(train_df)}")
        logger.info(f"Total test cycles: {len(test_df)}")
        logger.info(f"Risk distribution (train): {train_df['risk_level'].value_counts().to_dict()}")
        logger.info(f"Anomalous engines (train): {(train_df.groupby('unit_number')['anomaly_status'].apply(lambda x: (x == 'ANOMALOUS').any())).sum()}")
        
        return {
            'train': train_df,
            'test': test_df,
            'rul': rul_df,
            'scaler': scaler
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess NASA C-MAPSS dataset")
    parser.add_argument('--input', type=str, default='data/raw',
                       help='Input directory containing raw C-MAPSS files')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for processed files')
    parser.add_argument('--subset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004', 'all'],
                       help='Dataset subset to process')
    
    args = parser.parse_args()
    
    preprocessor = CMAPSSPreprocessor(
        input_dir=Path(args.input),
        output_dir=Path(args.output)
    )
    
    if args.subset == 'all':
        for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
            try:
                preprocessor.preprocess_dataset(subset)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {subset}: {e}")
    else:
        preprocessor.preprocess_dataset(args.subset)
    
    logger.info("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    main()
