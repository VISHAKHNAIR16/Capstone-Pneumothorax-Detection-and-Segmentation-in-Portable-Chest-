"""
FIXED DATA SPLITTER - WITH RLE MASKS FOR SEGMENTATION
Pneumothorax Segmentation - SIIM-ACR Dataset

CRITICAL FIX:
- Now preserves 'EncodedPixels' column in split CSVs
- This is REQUIRED for segmentation training
- Previous version only had True/False labels (good for classification, bad for segmentation)

Version: 2.2 (SEGMENTATION-READY)
Date: 2025-11-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import logging
from typing import Tuple, Dict, Optional
from datetime import datetime
import sys

# Fix for Windows Unicode issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging for data splitting process"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    log_file = log_path / f"data_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Use UTF-8 encoding for file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


# =====================================================================
# RLE DECODING HELPER
# =====================================================================
def decode_rle_to_mask(rle_string: str, image_shape: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Decode RLE string to binary mask
    
    Args:
        rle_string: RLE encoded string from SIIM CSV
        image_shape: (height, width) of image
    
    Returns:
        Binary mask as numpy array [H, W] with values 0 or 1
    
    Example RLE:
        "387620 23 996 33" means:
        - Start at pixel 387620, mark next 23 pixels as 1
        - Start at pixel 996, mark next 33 pixels as 1
    """
    height, width = image_shape
    
    # Handle negative cases (no pneumothorax)
    if rle_string.strip() == '-1':
        return np.zeros((height, width), dtype=np.uint8)
    
    try:
        # Parse RLE string
        runs = [int(x) for x in rle_string.split()]
        
        # Create flattened mask
        mask = np.zeros(height * width, dtype=np.uint8)
        
        # RLE format: [start1, length1, start2, length2, ...]
        for i in range(0, len(runs), 2):
            start = runs[i] - 1  # SIIM uses 1-indexed pixels
            length = runs[i + 1]
            mask[start:start + length] = 1
        
        # Reshape to 2D (column-major order as per SIIM)
        mask = mask.reshape((width, height)).T
        
        return mask
        
    except Exception as e:
        print(f"Error decoding RLE: {str(e)}")
        return np.zeros((height, width), dtype=np.uint8)


def validate_siim_csv(csv_path: str, logger: logging.Logger) -> bool:
    """Validate SIIM train-rle.csv format and content"""
    try:
        logger.info(f"Validating CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check column names (accounting for SIIM's space in column names)
        if ' EncodedPixels' not in df.columns and 'EncodedPixels' not in df.columns:
            logger.error("CSV missing 'EncodedPixels' column")
            return False
        
        if 'ImageId' not in df.columns:
            logger.error("CSV missing 'ImageId' column")
            return False
        
        # Check for required data
        if df.empty:
            logger.error("CSV is empty")
            return False
        
        # Check for nulls in critical columns
        if df['ImageId'].isnull().any():
            logger.error("Found null values in ImageId column")
            return False
        
        logger.info(f"[OK] CSV validation passed")
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Unique images: {df['ImageId'].nunique()}")
        
        return True
        
    except Exception as e:
        logger.error(f"CSV validation failed: {str(e)}")
        return False


# =====================================================================
# DATA SPLITTING
# =====================================================================
class PneumothoraxDataSplitter:
    """
    SEGMENTATION-READY data splitter for SIIM-ACR pneumothorax dataset
    
    CRITICAL CHANGE FROM PREVIOUS VERSION:
    - Now preserves 'EncodedPixels' column in output CSVs
    - This is REQUIRED for segmentation (not just classification)
    """
    
    def __init__(self,
                 train_rle_csv: str,
                 test_rle_csv: Optional[str] = None,
                 output_dir: str = "data/splits",
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 random_seed: int = 42,
                 logger: Optional[logging.Logger] = None):
        """Initialize data splitter"""
        self.train_rle_csv = train_rle_csv
        self.test_rle_csv = test_rle_csv
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.logger = logger or setup_logging()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data storage
        self.train_df = None
        self.test_df = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
    
    def load_siim_data(self) -> bool:
        """Load SIIM RLE CSV files"""
        try:
            self.logger.info("=" * 70)
            self.logger.info("LOADING SIIM-ACR DATASET")
            self.logger.info("=" * 70)
            
            # Validate CSV before loading
            if not validate_siim_csv(self.train_rle_csv, self.logger):
                return False
            
            # Load training CSV
            self.logger.info(f"Loading training data from: {self.train_rle_csv}")
            self.train_df = pd.read_csv(self.train_rle_csv)
            
            # Handle SIIM's space in column name
            if ' EncodedPixels' in self.train_df.columns:
                self.train_df.rename(columns={' EncodedPixels': 'EncodedPixels'}, inplace=True)
            
            # Create has_pneumothorax column (CORRECT spelling with 'o')
            self.train_df['has_pneumothorax'] = ~self.train_df['EncodedPixels'].astype(str).str.strip().isin(['-1'])
            
            self.logger.info(f"[OK] Loaded {len(self.train_df)} training records")
            self.logger.info(f"[INFO] Positive (has mask): {self.train_df['has_pneumothorax'].sum()}")
            self.logger.info(f"[INFO] Negative (no mask): {(~self.train_df['has_pneumothorax']).sum()}")
            
            # Load test CSV if provided
            if self.test_rle_csv:
                if not validate_siim_csv(self.test_rle_csv, self.logger):
                    self.logger.warning("Test CSV validation failed, proceeding without test set")
                    self.test_rle_csv = None
                else:
                    self.logger.info(f"Loading test data from: {self.test_rle_csv}")
                    self.test_df = pd.read_csv(self.test_rle_csv)
                    if ' EncodedPixels' in self.test_df.columns:
                        self.test_df.rename(columns={' EncodedPixels': 'EncodedPixels'}, inplace=True)
                    self.test_df['has_pneumothorax'] = ~self.test_df['EncodedPixels'].astype(str).str.strip().isin(['-1'])
                    self.logger.info(f"[OK] Loaded {len(self.test_df)} test records")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load SIIM data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def create_stratified_split(self) -> bool:
        """Create stratified train/val/test splits"""
        try:
            self.logger.info("=" * 70)
            self.logger.info("CREATING STRATIFIED SPLITS")
            self.logger.info("=" * 70)
            
            if self.train_df is None:
                self.logger.error("Training data not loaded. Call load_siim_data() first")
                return False
            
            # Get unique images (SIIM has multiple annotations per image)
            # We take the max label: if any annotation is positive, mark as positive
            unique_images = self.train_df.groupby('ImageId').agg({
                'has_pneumothorax': 'max',  # If any annotation positive -> positive
            }).reset_index()
            
            self.logger.info(f"Total unique images: {len(unique_images)}")
            self.logger.info(f"Positive (pneumothorax): {unique_images['has_pneumothorax'].sum()}")
            self.logger.info(f"Negative: {(~unique_images['has_pneumothorax']).sum()}")
            self.logger.info(f"Class distribution: {unique_images['has_pneumothorax'].mean():.2%} positive")
            
            # First split: Train vs (Val + Test)
            if self.test_split > 0:
                train_ids, temp_ids = train_test_split(
                    unique_images['ImageId'],
                    test_size=(self.val_split + self.test_split),
                    stratify=unique_images['has_pneumothorax'],
                    random_state=self.random_seed
                )
                
                # Second split: Val vs Test
                temp_labels = unique_images[unique_images['ImageId'].isin(temp_ids)]['has_pneumothorax']
                val_ratio_adjusted = self.val_split / (self.val_split + self.test_split)
                val_ids, test_ids = train_test_split(
                    temp_ids,
                    test_size=(1 - val_ratio_adjusted),
                    stratify=temp_labels,
                    random_state=self.random_seed
                )
            else:
                # If no test split, split only train/val
                train_ids, val_ids = train_test_split(
                    unique_images['ImageId'],
                    test_size=self.val_split,
                    stratify=unique_images['has_pneumothorax'],
                    random_state=self.random_seed
                )
                test_ids = []
            
            # CRITICAL FIX: Keep ALL columns including EncodedPixels
            self.train_set = self.train_df[self.train_df['ImageId'].isin(train_ids)].copy()
            self.val_set = self.train_df[self.train_df['ImageId'].isin(val_ids)].copy()
            
            if len(test_ids) > 0 and self.test_df is not None:
                self.test_set = self.test_df[self.test_df['ImageId'].isin(test_ids)].copy()
            elif len(test_ids) > 0:
                self.test_set = self.train_df[self.train_df['ImageId'].isin(test_ids)].copy()
            
            self.logger.info(f"\n[OK] Split created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create splits: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def print_split_statistics(self):
        """Print detailed statistics about splits"""
        
        self.logger.info("=" * 70)
        self.logger.info("SPLIT STATISTICS")
        self.logger.info("=" * 70)
        
        if self.train_set is not None:
            unique_train = self.train_set['ImageId'].nunique()
            pos_train = self.train_set['has_pneumothorax'].sum()
            neg_train = len(self.train_set) - pos_train
            self.logger.info(f"\nTRAINING SET:")
            self.logger.info(f"  Total rows: {len(self.train_set)}")
            self.logger.info(f"  Unique images: {unique_train}")
            self.logger.info(f"  Positive: {pos_train} ({pos_train/len(self.train_set):.2%})")
            self.logger.info(f"  Negative: {neg_train} ({neg_train/len(self.train_set):.2%})")
        
        if self.val_set is not None:
            unique_val = self.val_set['ImageId'].nunique()
            pos_val = self.val_set['has_pneumothorax'].sum()
            neg_val = len(self.val_set) - pos_val
            self.logger.info(f"\nVALIDATION SET:")
            self.logger.info(f"  Total rows: {len(self.val_set)}")
            self.logger.info(f"  Unique images: {unique_val}")
            self.logger.info(f"  Positive: {pos_val} ({pos_val/len(self.val_set):.2%})")
            self.logger.info(f"  Negative: {neg_val} ({neg_val/len(self.val_set):.2%})")
        
        if self.test_set is not None:
            unique_test = self.test_set['ImageId'].nunique()
            pos_test = self.test_set['has_pneumothorax'].sum()
            neg_test = len(self.test_set) - pos_test
            self.logger.info(f"\nTEST SET:")
            self.logger.info(f"  Total rows: {len(self.test_set)}")
            self.logger.info(f"  Unique images: {unique_test}")
            self.logger.info(f"  Positive: {pos_test} ({pos_test/len(self.test_set):.2%})")
            self.logger.info(f"  Negative: {neg_test} ({neg_test/len(self.test_set):.2%})")
        
        self.logger.info("=" * 70)
    
    def save_splits(self):
        """
        Save split metadata to files
        
        CRITICAL: Now includes EncodedPixels column for segmentation!
        
        Files created:
        - train_split.csv: ImageId, EncodedPixels, has_pneumothorax
        - val_split.csv: ImageId, EncodedPixels, has_pneumothorax
        - test_split.csv: ImageId, EncodedPixels, has_pneumothorax
        - split_metadata.json: Split configuration and statistics
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info("SAVING SPLITS (WITH RLE MASKS)")
            self.logger.info("=" * 70)
            
            # CRITICAL FIX: Save ImageId, EncodedPixels, has_pneumothorax
            columns_to_save = ['ImageId', 'EncodedPixels', 'has_pneumothorax']
            
            # Save splits as CSVs
            if self.train_set is not None:
                train_path = self.output_dir / 'train_split.csv'
                self.train_set[columns_to_save].to_csv(train_path, index=False)
                self.logger.info(f"[OK] Saved: {train_path} ({len(self.train_set)} records)")
                self.logger.info(f"     Columns: {columns_to_save}")
            
            if self.val_set is not None:
                val_path = self.output_dir / 'val_split.csv'
                self.val_set[columns_to_save].to_csv(val_path, index=False)
                self.logger.info(f"[OK] Saved: {val_path} ({len(self.val_set)} records)")
                self.logger.info(f"     Columns: {columns_to_save}")
            
            if self.test_set is not None:
                test_path = self.output_dir / 'test_split.csv'
                self.test_set[columns_to_save].to_csv(test_path, index=False)
                self.logger.info(f"[OK] Saved: {test_path} ({len(self.test_set)} records)")
                self.logger.info(f"     Columns: {columns_to_save}")
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'random_seed': self.random_seed,
                'split_configuration': {
                    'val_split': self.val_split,
                    'test_split': self.test_split,
                },
                'columns_saved': columns_to_save,
                'statistics': {
                    'train': {
                        'total_records': len(self.train_set) if self.train_set is not None else 0,
                        'unique_images': self.train_set['ImageId'].nunique() if self.train_set is not None else 0,
                        'positive_count': int(self.train_set['has_pneumothorax'].sum()) if self.train_set is not None else 0,
                        'positive_ratio': float(self.train_set['has_pneumothorax'].mean()) if self.train_set is not None else 0,
                    },
                    'val': {
                        'total_records': len(self.val_set) if self.val_set is not None else 0,
                        'unique_images': self.val_set['ImageId'].nunique() if self.val_set is not None else 0,
                        'positive_count': int(self.val_set['has_pneumothorax'].sum()) if self.val_set is not None else 0,
                        'positive_ratio': float(self.val_set['has_pneumothorax'].mean()) if self.val_set is not None else 0,
                    },
                    'test': {
                        'total_records': len(self.test_set) if self.test_set is not None else 0,
                        'unique_images': self.test_set['ImageId'].nunique() if self.test_set is not None else 0,
                        'positive_count': int(self.test_set['has_pneumothorax'].sum()) if self.test_set is not None else 0,
                        'positive_ratio': float(self.test_set['has_pneumothorax'].mean()) if self.test_set is not None else 0,
                    }
                }
            }
            
            metadata_path = self.output_dir / 'split_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"[OK] Saved: {metadata_path}")
            self.logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save splits: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Get train/val/test splits"""
        return self.train_set, self.val_set, self.test_set
    
    def run_full_pipeline(self) -> bool:
        """Run complete splitting pipeline"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STARTING SEGMENTATION-READY DATA SPLITTING PIPELINE")
        self.logger.info("=" * 70 + "\n")
        
        # Step 1: Load data
        if not self.load_siim_data():
            self.logger.error("Pipeline failed at data loading step")
            return False
        
        # Step 2: Create splits
        if not self.create_stratified_split():
            self.logger.error("Pipeline failed at split creation step")
            return False
        
        # Step 3: Print statistics
        self.print_split_statistics()
        
        # Step 4: Save splits
        if not self.save_splits():
            self.logger.error("Pipeline failed at save splits step")
            return False
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("[SUCCESS] SEGMENTATION-READY SPLITS CREATED")
        self.logger.info("[INFO] Split CSVs now include EncodedPixels for masks")
        self.logger.info("=" * 70 + "\n")
        
        return True


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    """
    USAGE INSTRUCTIONS:
    
    1. Update paths below to your SIIM dataset location
    2. Run: python data_splitter_segmentation.py
    3. Check data/splits/ for CSV files WITH EncodedPixels column
    """
    
    # ===== CONFIGURE THESE PATHS =====
    TRAIN_RLE_CSV = r"C:/Users/VISHAKH NAIR/Desktop/CAPSTONE/Capstone-Pneumothorax-Detection-and-Segmentation-in-Portable-Chest-/Data/train-rle.csv"
    TEST_RLE_CSV = None
    OUTPUT_DIR = "data/splits"
    # =================================
    
    # Initialize logger
    logger = setup_logging()
    
    # Create splitter
    splitter = PneumothoraxDataSplitter(
        train_rle_csv=TRAIN_RLE_CSV,
        test_rle_csv=TEST_RLE_CSV,
        output_dir=OUTPUT_DIR,
        val_split=0.15,
        test_split=0.15,
        random_seed=42,
        logger=logger
    )
    
    # Run pipeline
    success = splitter.run_full_pipeline()
    
    if success:
        print("\n[SUCCESS] Segmentation-ready data splitting completed!")
        print(f"[INFO] Splits saved to: {OUTPUT_DIR}")
        print("[INFO] CSV files now include EncodedPixels for mask creation")
        print("[INFO] Log file saved to: logs/")
        
        # Print summary
        train, val, test = splitter.get_splits()
        print(f"\n[SUMMARY]:")
        print(f"   Training images: {train['ImageId'].nunique()}")
        print(f"   Validation images: {val['ImageId'].nunique()}")
        if test is not None:
            print(f"   Test images: {test['ImageId'].nunique()}")
        
        # Show sample of what's in the CSV
        print(f"\n[SAMPLE] train_split.csv first 3 rows:")
        print(train[['ImageId', 'EncodedPixels', 'has_pneumothorax']].head(3))
        
    else:
        print("\n[ERROR] Data splitting failed!")
        print("[INFO] Check logs/ directory for details")
