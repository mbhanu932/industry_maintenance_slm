#!/usr/bin/env python3
"""
Download NASA C-MAPSS Turbofan Engine Degradation Dataset

Downloads the dataset from the NASA repository and extracts it to the data directory.

Usage:
    python src/data/download_cmappss.py --output data/raw
"""

import urllib.request
import zipfile
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NASA C-MAPSS dataset URL
CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"
CMAPSS_MIRROR = "https://github.com/ericzhao28/NASA-CMAPSS/raw/master/data.zip"

def download_cmappss(output_dir: str = "data/raw"):
    """
    Download and extract NASA C-MAPSS dataset.
    
    Args:
        output_dir: Directory to save extracted files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_path / "cmappss_data.zip"
    
    # Download
    logger.info(f"Downloading NASA C-MAPSS dataset...")
    logger.info(f"Source: {CMAPSS_MIRROR}")
    logger.info(f"Destination: {zip_path}")
    
    try:
        # Try primary URL
        urllib.request.urlretrieve(CMAPSS_MIRROR, zip_path)
    except Exception as e:
        logger.warning(f"Failed to download from primary URL: {e}")
        logger.info("Please download manually from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        logger.info("Look for 'Turbofan Engine Degradation Simulation Data Set'")
        return False
    
    # Extract
    logger.info(f"Extracting to {output_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    # Clean up
    logger.info(f"Cleaning up...")
    os.remove(zip_path)
    
    logger.info(f"✓ Dataset downloaded and extracted to {output_path}")
    logger.info(f"\nFiles extracted:")
    for file in output_path.glob("*.txt"):
        logger.info(f"  - {file.name}")
    
    return True


def verify_dataset(data_dir: str = "data/raw"):
    """Verify that all required C-MAPSS files exist."""
    required_files = [
        'train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt',
        'train_FD002.txt', 'test_FD002.txt', 'RUL_FD002.txt',
        'train_FD003.txt', 'test_FD003.txt', 'RUL_FD003.txt',
        'train_FD004.txt', 'test_FD004.txt', 'RUL_FD004.txt',
    ]
    
    data_path = Path(data_dir)
    missing_files = []
    
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} files:")
        for file in missing_files:
            logger.warning(f"  - {file}")
        return False
    else:
        logger.info("✓ All C-MAPSS dataset files present")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download NASA C-MAPSS dataset")
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory for raw data')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing files, do not download')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.output)
    else:
        download_cmappss(args.output)
