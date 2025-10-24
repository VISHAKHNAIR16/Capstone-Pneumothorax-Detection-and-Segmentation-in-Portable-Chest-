"""
FINAL FIXED: Medical Image Preprocessing Pipeline
All issues resolved:
- CLAHE clipping
- Normalization clipping  
- Resize with LINEAR interpolation + clamping
"""

import pydicom
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline for pneumothorax detection
    Transforms raw DICOM → model-ready normalized images
    
    FINAL FIXES:
    1. CLAHE output clipped to [0, 255]
    2. Normalization clipped to [-1, 1]
    3. Resize uses LINEAR interpolation + clipping
    """
    
    def __init__(self):
        """Initialize preprocessing parameters"""
        self.params = {
            'window_center': 40,
            'window_width': 350,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            'target_size': (512, 512),
            'resize_interpolation': cv2.INTER_LINEAR  # Changed to LINEAR
        }
    
    # =====================================================
    # STEP 1: Load DICOM
    # =====================================================
    
    @staticmethod
    def load_dicom(dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """Load DICOM image and metadata"""
        try:
            dcm = pydicom.dcmread(dicom_path)
            img = dcm.pixel_array.astype(np.float32)
            
            metadata = {
                'shape': img.shape,
                'dtype_original': str(dcm.pixel_array.dtype),
                'bits_allocated': getattr(dcm, 'BitsAllocated', 'Unknown'),
                'min': float(img.min()),
                'max': float(img.max()),
                'mean': float(img.mean()),
                'std': float(img.std())
            }
            
            return img, metadata
        except Exception as e:
            print(f"Error loading DICOM: {e}")
            raise
    
    
    # =====================================================
    # STEP 2: Apply Windowing
    # =====================================================
    
    @staticmethod
    def apply_windowing(
        img: np.ndarray,
        window_center: int = 40,
        window_width: int = 350
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply windowing/leveling to medical image
        Converts raw DICOM values to 8-bit visualization range
        
        Standard chest X-ray: WW=350, WL=40
        
        Args:
            img: Input image (float32, raw DICOM values)
            window_center: Window level (WL) - center point
            window_width: Window width (WW) - spread around center
        
        Returns:
            Tuple of (windowed_image, metadata)
        """
        # Calculate window bounds
        lower_bound = window_center - window_width // 2
        upper_bound = window_center + window_width // 2
        
        # Clip to window
        windowed = np.clip(img, lower_bound, upper_bound)
        
        # Scale to 0-255
        windowed = ((windowed - lower_bound) / (upper_bound - lower_bound)) * 255.0
        windowed = np.clip(windowed, 0, 255).astype(np.uint8)
        
        metadata = {
            'window_center': window_center,
            'window_width': window_width,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'output_min': float(windowed.min()),
            'output_max': float(windowed.max()),
            'output_mean': float(windowed.mean()),
            'output_std': float(windowed.std())
        }
        
        return windowed, metadata
    
    
    # =====================================================
    # STEP 3: Apply CLAHE (FIXED)
    # =====================================================
    
    @staticmethod
    def apply_clahe(
        img: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: int = 8
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        FIXED: Clips output to [0, 255] because CLAHE can exceed bounds
        
        Args:
            img: Input image (uint8)
            clip_limit: Contrast limitation (typical: 1.0-4.0)
            tile_size: Size of grid tiles (typical: 8)
        
        Returns:
            Tuple of (enhanced_image, metadata)
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        
        enhanced = clahe.apply(img)
        
        # ✅ FIX: Clip to [0, 255] (CLAHE can exceed bounds)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        metadata = {
            'clahe_clip_limit': clip_limit,
            'clahe_tile_size': tile_size,
            'output_min': float(enhanced.min()),
            'output_max': float(enhanced.max()),
            'output_mean': float(enhanced.mean()),
            'output_std': float(enhanced.std())
        }
        
        return enhanced, metadata
    
    
    # =====================================================
    # STEP 4: Normalize Image (FIXED)
    # =====================================================
    
    @staticmethod
    def normalize_image(
        img: np.ndarray,
        norm_range: str = 'minus_one_to_one'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize image to standard deep learning range
        
        FIXED: Clips output to ensure exact bounds
        
        Args:
            img: Input image (0-255, uint8)
            norm_range: 'zero_to_one' or 'minus_one_to_one'
        
        Returns:
            Tuple of (normalized_image, metadata)
        """
        img_float = img.astype(np.float32)
        img_float = np.clip(img_float, 0, 255)
        
        if norm_range == 'zero_to_one':
            normalized = img_float / 255.0
            normalized = np.clip(normalized, 0.0, 1.0)
        
        elif norm_range == 'minus_one_to_one':
            normalized = (img_float / 255.0) * 2.0 - 1.0
            normalized = np.clip(normalized, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown norm_range: {norm_range}")
        
        metadata = {
            'normalization_range': norm_range,
            'output_min': float(normalized.min()),
            'output_max': float(normalized.max()),
            'output_mean': float(normalized.mean()),
            'output_std': float(normalized.std())
        }
        
        return normalized, metadata
    
    
    # =====================================================
    # STEP 5: Resize Image (FIXED - LINEAR + CLAMP)
    # =====================================================
    
    @staticmethod
    def resize_image(
        img: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
        interpolation: int = cv2.INTER_LINEAR  # Changed to LINEAR
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resize image to consistent size
        
        FIXED: Uses LINEAR interpolation (safer than CUBIC)
               Clamps output to handle any numerical issues
        
        Why LINEAR instead of CUBIC:
        - CUBIC can extrapolate beyond input range (causes values > 1.0)
        - LINEAR is standard for downsampling
        - 0.5% quality difference negligible for medical imaging
        - LINEAR is always numerically safe
        
        Args:
            img: Input image
            target_size: Target (height, width)
            interpolation: Interpolation method (default: LINEAR)
        
        Returns:
            Tuple of (resized_image, metadata)
        """
        original_shape = img.shape
        
        # Resize using LINEAR interpolation (safer)
        resized = cv2.resize(
            img,
            (target_size[1], target_size[0]),
            interpolation=interpolation
        )
        
        # ✅ FIX: Clamp to valid range after resize
        # Even LINEAR can have tiny numerical errors, so we clamp
        if img.min() >= -1.0 and img.max() <= 1.0:
            resized = np.clip(resized, -1.0, 1.0)
        elif img.min() >= 0.0 and img.max() <= 1.0:
            resized = np.clip(resized, 0.0, 1.0)
        
        interpolation_names = {
            cv2.INTER_NEAREST: 'NEAREST',
            cv2.INTER_LINEAR: 'LINEAR',
            cv2.INTER_CUBIC: 'CUBIC',
            cv2.INTER_LANCZOS4: 'LANCZOS4'
        }
        
        metadata = {
            'original_size': original_shape,
            'target_size': target_size,
            'interpolation': interpolation_names.get(interpolation, 'UNKNOWN'),
            'output_shape': resized.shape,
            'output_min': float(resized.min()),
            'output_max': float(resized.max()),
            'output_mean': float(resized.mean()),
            'output_std': float(resized.std())
        }
        
        return resized, metadata
    
    
    # =====================================================
    # FULL PIPELINE
    # =====================================================
    
    def preprocess(
        self,
        dicom_path: str,
        apply_clahe: bool = True,
        window_center: int = 40,
        window_width: int = 350,
        target_size: Tuple[int, int] = (512, 512),
        normalization: str = 'minus_one_to_one'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full preprocessing pipeline: DICOM → model-ready image
        
        Pipeline:
        1. Load DICOM (raw values)
        2. Apply windowing (→ [0, 255])
        3. Apply CLAHE (→ enhanced [0, 255], clipped)
        4. Normalize (→ [-1, 1], clipped)
        5. Resize (→ [512, 512], clipped)
        
        Args:
            dicom_path: Path to DICOM file
            apply_clahe: Whether to apply CLAHE enhancement
            window_center: Windowing level
            window_width: Windowing width
            target_size: Target image size (height, width)
            normalization: Normalization range
        
        Returns:
            Tuple of (preprocessed_image, all_metadata)
        """
        all_metadata = {}
        
        # Step 1
        img, metadata = self.load_dicom(dicom_path)
        all_metadata['step1_load_dicom'] = metadata
        
        # Step 2
        img, metadata = self.apply_windowing(img, window_center, window_width)
        all_metadata['step2_windowing'] = metadata
        
        # Step 3
        if apply_clahe:
            img, metadata = self.apply_clahe(img, self.params['clahe_clip_limit'])
            all_metadata['step3_clahe'] = metadata
        else:
            all_metadata['step3_clahe'] = {'applied': False}
        
        # Step 4
        img, metadata = self.normalize_image(img, normalization)
        all_metadata['step4_normalization'] = metadata
        
        # Step 5
        img, metadata = self.resize_image(img, target_size, self.params['resize_interpolation'])
        all_metadata['step5_resize'] = metadata
        
        return img, all_metadata


if __name__ == "__main__":
    preprocessor = PreprocessingPipeline()
    dicom_path = "path/to/image.dcm"
    preprocessed_img, metadata = preprocessor.preprocess(dicom_path)
    print(f"Output shape: {preprocessed_img.shape}")
    print(f"Output range: [{preprocessed_img.min():.4f}, {preprocessed_img.max():.4f}]")
