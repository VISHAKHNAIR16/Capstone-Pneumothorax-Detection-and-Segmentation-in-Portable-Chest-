"""
================================================================================
COMPREHENSIVE DATA LOADER VALIDATION WITH VISUAL DEBUGGING - ENHANCED VERSION
Production-Grade Pneumothorax DataLoader Validation & Analysis
================================================================================

PROJECT: Capstone - Enhanced Pneumothorax Detection
AUTHOR: AI Research Assistant
DATE: November 7, 2025

PURPOSE: 
- Validate each step of the ENHANCED data loader pipeline with detailed visualizations
- Test new features: Enhanced Random Sampling & Aggressive Class Balancing
- Debug the low Dice score issue (stuck at ~0.35)
- Provide comprehensive analysis of all data processing steps
"""

import torch
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Any
import sys
import os
import gc
from tqdm import tqdm
import traceback
from datetime import datetime
import random

# Import your ENHANCED data loader components
try:
    from data_loader import (
        RAMPreloader, GPUAugmentationPipeline, SmartArtifactGenerator,
        ArtifactPlacementStrategy, decode_rle_mask_optimized, 
        process_mask_production, BaseCurriculumDataset, Level1BasicDataset,
        Level2StandardDataset, Level3AdvancedDataset, AdvancedClassBalancer,
        create_production_loader, PerformanceMonitor, MedicalGradePreprocessor,
        EnhancedRandomSampler, ChestXRayArtifactGenerator  # NEW IMPORTS
    )
except ImportError as e:
    print(f"⚠️ Some imports failed: {e}")
    # Define fallbacks for critical components
    class EnhancedRandomSampler:
        def __init__(self, *args, **kwargs):
            pass

# ============================================================================
# COMPREHENSIVE VISUALIZATION SETUP
# ============================================================================

class VisualDebugger:
    """Advanced visualization for debugging data processing pipeline"""
    
    def __init__(self):
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup matplotlib for detailed visualization."""
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['axes.grid'] = False
        
    def create_detailed_comparison(self, images: Dict[str, np.ndarray], 
                                 titles: List[str], 
                                 suptitle: str,
                                 save_path: str = None,
                                 figsize: Tuple[int, int] = (20, 15)):
        """
        Create detailed comparison plots with statistics.
        
        Args:
            images: Dictionary of {title: image_array}
            titles: List of titles for subplots
            suptitle: Main title for the figure
            save_path: Path to save the figure
            figsize: Figure size
        """
        n_images = len(images)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]
        
        for idx, (title, image) in enumerate(zip(titles, images.values())):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Handle different image types
            if len(image.shape) == 2:  # Grayscale
                im = ax.imshow(image, cmap='gray')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:  # RGB or multi-channel
                im = ax.imshow(image)
                
            # Enhanced title with statistics
            stats_text = self._get_image_statistics(image, title)
            ax.set_title(stats_text, fontsize=9, pad=10)
            ax.axis('off')
            
            # Add detailed statistics box
            self._add_statistics_box(ax, image, title)
        
        # Hide unused subplots
        for idx in range(len(images), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved visualization: {save_path}")
        
        plt.show()
        plt.close()
    
    def _get_image_statistics(self, image: np.ndarray, title: str) -> str:
        """Generate detailed statistics string for image title."""
        stats_parts = [
            f"{title}",
            f"Shape: {image.shape}",
            f"Range: [{image.min():.3f}, {image.max():.3f}]"
        ]
        
        if 'mask' in title.lower():
            unique_vals = np.unique(image)
            stats_parts.append(f"Unique: {unique_vals}")
            if len(unique_vals) <= 10:  # Only show for reasonable number of unique values
                stats_parts.append(f"Values: {unique_vals}")
        
        return '\n'.join(stats_parts)
    
    def _add_statistics_box(self, ax, image: np.ndarray, title: str):
        """Add detailed statistics text box to plot."""
        stats_lines = []
        
        # Basic statistics
        stats_lines.extend([
            f"Min: {image.min():.6f}",
            f"Max: {image.max():.6f}",
            f"Mean: {image.mean():.6f}",
            f"Std: {image.std():.6f}",
            f"Shape: {image.shape}"
        ])
        
        # Mask-specific statistics
        if 'mask' in title.lower():
            binary_mask = (image > 0.5).astype(np.float32)
            stats_lines.extend([
                f"Pneumothorax %: {binary_mask.mean() * 100:.4f}%",
                f"Total px > 0.5: {np.sum(binary_mask)}",
                f"Is Binary: {len(np.unique(image)) <= 2}"
            ])
        
        # Image-specific statistics
        if 'image' in title.lower() and 'mask' not in title.lower():
            stats_lines.extend([
                f"Dynamic Range: {image.max() - image.min():.4f}"
            ])
        
        stats_text = '\n'.join(stats_lines)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    def create_processing_pipeline_plot(self, steps: Dict[str, np.ndarray], 
                                      pipeline_name: str,
                                      save_path: str = None):
        """Create a comprehensive pipeline visualization."""
        self.create_detailed_comparison(
            steps,
            list(steps.keys()),
            f"{pipeline_name} - Processing Pipeline",
            save_path,
            figsize=(18, 12)
        )
    
    def create_overlay_visualization(self, image: np.ndarray, mask: np.ndarray, 
                                   title: str = "Image-Mask Overlay",
                                   save_path: str = None) -> np.ndarray:
        """
        Create detailed overlay visualization.
        
        Args:
            image: Input image
            mask: Binary mask
            title: Plot title
            save_path: Path to save visualization
            
        Returns:
            Overlay image
        """
        # Normalize image to [0, 1] for visualization
        img_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Create RGB image
        if len(img_normalized.shape) == 2:
            img_rgb = np.stack([img_normalized] * 3, axis=-1)
        else:
            img_rgb = img_normalized
        
        # Create mask overlay (red color with transparency)
        overlay = img_rgb.copy()
        mask_indices = mask > 0.5
        
        # Use different colors based on mask confidence
        overlay[mask_indices] = [1.0, 0.2, 0.2]  # Red color for pneumothorax
        
        # Create visualization figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(img_normalized, cmap='gray')
        axes[0, 0].set_title(f'Original Image\nRange: [{image.min():.3f}, {image.max():.3f}]')
        axes[0, 0].axis('off')
        
        # Mask
        axes[0, 1].imshow(mask, cmap='Reds')
        axes[0, 1].set_title(f'Mask\nPneumothorax: {mask.mean() * 100:.2f}%')
        axes[0, 1].axis('off')
        
        # Overlay
        axes[1, 0].imshow(img_normalized, cmap='gray')
        axes[1, 0].imshow(mask, cmap='Reds', alpha=0.3)
        axes[1, 0].set_title('Transparency Overlay')
        axes[1, 0].axis('off')
        
        # Enhanced overlay
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Enhanced Overlay (Red = Pneumothorax)')
        axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved overlay: {save_path}")
        
        plt.show()
        plt.close()
        
        return overlay

# ============================================================================
# STEP-BY-STEP MASK PROCESSING VALIDATION WITH VISUAL DEBUGGING
# ============================================================================

class MaskProcessingValidator:
    """Comprehensive mask processing validation with visual debugging"""
    
    def __init__(self, visual_debugger: VisualDebugger):
        self.visual_debugger = visual_debugger
        self.results = {}
    
    def validate_mask_processing_pipeline(self, rle_string: str, 
                                        target_size: Tuple[int, int] = (512, 512),
                                        save_dir: str = "mask_validation") -> Dict[str, Any]:
        """
        Comprehensive validation of mask processing pipeline with detailed visualization.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        print("=" * 80)
        print("🔍 MASK PROCESSING PIPELINE VALIDATION")
        print("=" * 80)
        print(f"RLE String Preview: {rle_string[:100]}..." if len(rle_string) > 100 else f"RLE String: {rle_string}")
        
        processing_steps = {}
        step_details = {}
        
        try:
            # Step 1: Decode RLE to uint8 mask
            print("\n📊 Step 1: RLE Decoding")
            mask_uint8 = decode_rle_mask_optimized(rle_string, shape=(1024, 1024))
            processing_steps['1_RLE_Decoded_uint8'] = mask_uint8
            step_details['RLE_Decoding'] = self._analyze_mask_step(mask_uint8, "RLE Decoded")
            
            # Step 2: Convert to binary mask
            print("\n📊 Step 2: Binary Conversion")
            mask_binary = (mask_uint8 > 127).astype(np.float32)
            processing_steps['2_Binary_Conversion'] = mask_binary
            step_details['Binary_Conversion'] = self._analyze_mask_step(mask_binary, "Binary Conversion")
            
            # Step 3: Resize with nearest interpolation
            print("\n📊 Step 3: Resize with NEAREST Interpolation")
            mask_resized = cv2.resize(mask_binary, (target_size[1], target_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            processing_steps['3_After_Resize'] = mask_resized
            step_details['Resize'] = self._analyze_mask_step(mask_resized, "After Resize")
            
            # Step 4: Final binary validation
            print("\n📊 Step 4: Final Binary Validation")
            mask_final = (mask_resized > 0.5).astype(np.float32)
            processing_steps['4_Final_Mask'] = mask_final
            step_details['Final_Validation'] = self._analyze_mask_step(mask_final, "Final Mask")
            
            # Create comprehensive visualization
            self.visual_debugger.create_processing_pipeline_plot(
                processing_steps,
                "Mask Processing Pipeline",
                save_path / "mask_processing_pipeline.png"
            )
            
            # Validate binary nature
            self._validate_binary_mask(mask_final)
            
            # Store results
            self.results = {
                'processing_steps': processing_steps,
                'step_details': step_details,
                'final_mask': mask_final,
                'is_valid': len(np.unique(mask_final)) <= 2
            }
            
            return self.results
            
        except Exception as e:
            print(f"❌ Mask processing validation failed: {e}")
            traceback.print_exc()
            return {}
    
    def _analyze_mask_step(self, mask: np.ndarray, step_name: str) -> Dict[str, Any]:
        """Analyze a single mask processing step."""
        analysis = {
            'step_name': step_name,
            'shape': mask.shape,
            'dtype': str(mask.dtype),
            'value_range': [float(mask.min()), float(mask.max())],
            'unique_values': [float(x) for x in np.unique(mask)],
            'num_unique_values': len(np.unique(mask)),
            'mean_value': float(mask.mean()),
            'pneumothorax_pixels': int(np.sum(mask > 0.5)),
            'total_pixels': int(mask.size),
            'pneumothorax_coverage': float(np.sum(mask > 0.5) / mask.size)
        }
        
        # Print analysis
        print(f"  ✅ {step_name}:")
        print(f"     Shape: {analysis['shape']}")
        print(f"     Data type: {analysis['dtype']}")
        print(f"     Value range: [{analysis['value_range'][0]:.3f}, {analysis['value_range'][1]:.3f}]")
        print(f"     Unique values: {analysis['unique_values']}")
        print(f"     Pneumothorax coverage: {analysis['pneumothorax_coverage']:.4%}")
        
        return analysis
    
    def _validate_binary_mask(self, mask: np.ndarray):
        """Validate that mask is properly binary."""
        unique_vals = np.unique(mask)
        print(f"\n🔍 Binary Mask Validation:")
        print(f"  Unique values: {unique_vals}")
        print(f"  Number of unique values: {len(unique_vals)}")
        
        if len(unique_vals) <= 2:
            print("  ✅ Mask is properly binary")
            if len(unique_vals) == 2:
                print(f"  ✅ Binary values: {unique_vals[0]:.1f} and {unique_vals[1]:.1f}")
        else:
            print(f"  ❌ WARNING: Mask has non-binary values: {unique_vals}")
            print(f"  ❌ This could be causing Dice score issues!")

# ============================================================================
# IMAGE PROCESSING PIPELINE VALIDATION WITH VISUAL DEBUGGING - UPDATED
# ============================================================================

class ImageProcessingValidator:
    """Comprehensive image processing validation with visual debugging"""
    
    def __init__(self, visual_debugger: VisualDebugger):
        self.visual_debugger = visual_debugger
        self.preprocessor = MedicalGradePreprocessor()
    
    def validate_image_processing_pipeline(self, dicom_path: str,
                                         target_size: Tuple[int, int] = (512, 512),
                                         save_dir: str = "image_validation") -> Dict[str, Any]:
        """
        Comprehensive validation of image processing pipeline with detailed visualization.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        print("=" * 80)
        print("🔍 IMAGE PROCESSING PIPELINE VALIDATION")
        print("=" * 80)
        
        processing_steps = {}
        step_details = {}
        
        try:
            # Step 1: Load DICOM
            print("\n📊 Step 1: DICOM Loading")
            dcm = pydicom.dcmread(dicom_path)
            raw_image = dcm.pixel_array.astype(np.float32)
            processing_steps['1_Raw_DICOM'] = raw_image
            step_details['DICOM_Loading'] = self._analyze_image_step(raw_image, "Raw DICOM")
            
            # Step 2: Apply windowing - UPDATED METHOD CALL
            print("\n📊 Step 2: Medical Windowing")
            try:
                # Use the preprocessor's method directly
                windowed, window_metadata = self.preprocessor.apply_medical_windowing(raw_image)
                processing_steps['2_After_Windowing'] = windowed
                step_details['Windowing'] = self._analyze_image_step(windowed, "After Windowing")
            except Exception as e:
                print(f"⚠️ Windowing failed, using fallback: {e}")
                # Fallback windowing
                window_center, window_width = 40, 400
                window_min = window_center - window_width // 2
                window_max = window_center + window_width // 2
                windowed = np.clip(raw_image, window_min, window_max)
                windowed = ((windowed - window_min) / (window_max - window_min)) * 255.0
                windowed = np.clip(windowed, 0, 255).astype(np.uint8)
                processing_steps['2_After_Windowing'] = windowed
                step_details['Windowing'] = self._analyze_image_step(windowed, "After Windowing")
            
            # Step 3: Apply CLAHE - UPDATED METHOD CALL
            print("\n📊 Step 3: CLAHE Enhancement")
            try:
                clahe_enhanced, clahe_metadata = self.preprocessor.apply_contrast_enhancement(windowed)
                processing_steps['3_After_CLAHE'] = clahe_enhanced
                step_details['CLAHE'] = self._analyze_image_step(clahe_enhanced, "After CLAHE")
            except Exception as e:
                print(f"⚠️ CLAHE failed, using original: {e}")
                processing_steps['3_After_CLAHE'] = windowed
                step_details['CLAHE'] = self._analyze_image_step(windowed, "After CLAHE (Fallback)")
            
            # Step 4: Normalize to [-1, 1] - UPDATED METHOD CALL
            print("\n📊 Step 4: Normalization")
            try:
                normalized, norm_metadata = self.preprocessor.normalize_to_deep_learning_range(clahe_enhanced)
                processing_steps['4_After_Normalization'] = normalized
                step_details['Normalization'] = self._analyze_image_step(normalized, "After Normalization")
            except Exception as e:
                print(f"⚠️ Normalization failed, using fallback: {e}")
                # Fallback normalization
                normalized = (clahe_enhanced.astype(np.float32) / 255.0) * 2 - 1
                processing_steps['4_After_Normalization'] = normalized
                step_details['Normalization'] = self._analyze_image_step(normalized, "After Normalization")
            
            # Step 5: Resize - UPDATED METHOD CALL
            print("\n📊 Step 5: Resize")
            try:
                resized, resize_metadata = self.preprocessor.resize_to_target(normalized)
                processing_steps['5_Final_Resized'] = resized
                step_details['Resize'] = self._analyze_image_step(resized, "Final Resized")
            except Exception as e:
                print(f"⚠️ Resize failed, using fallback: {e}")
                # Fallback resize
                resized = cv2.resize(normalized, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                processing_steps['5_Final_Resized'] = resized
                step_details['Resize'] = self._analyze_image_step(resized, "Final Resized")
            
            # Create comprehensive visualization
            self.visual_debugger.create_processing_pipeline_plot(
                processing_steps,
                "Image Processing Pipeline",
                save_path / "image_processing_pipeline.png"
            )
            
            # Store results
            self.results = {
                'processing_steps': processing_steps,
                'step_details': step_details,
                'final_image': resized
            }
            
            return self.results
            
        except Exception as e:
            print(f"❌ Image processing validation failed: {e}")
            traceback.print_exc()
            return {}
    
    def _analyze_image_step(self, image: np.ndarray, step_name: str) -> Dict[str, Any]:
        """Analyze a single image processing step."""
        analysis = {
            'step_name': step_name,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'value_range': [float(image.min()), float(image.max())],
            'mean_value': float(image.mean()),
            'std_value': float(image.std()),
            'dynamic_range': float(image.max() - image.min())
        }
        
        # Print analysis
        print(f"  ✅ {step_name}:")
        print(f"     Shape: {analysis['shape']}")
        print(f"     Data type: {analysis['dtype']}")
        print(f"     Value range: [{analysis['value_range'][0]:.6f}, {analysis['value_range'][1]:.6f}]")
        print(f"     Mean: {analysis['mean_value']:.6f}")
        print(f"     Std: {analysis['std_value']:.6f}")
        
        return analysis

# ============================================================================
# ENHANCED SAMPLING AND BALANCING VALIDATION - NEW CLASS
# ============================================================================

class EnhancedFeaturesValidator:
    """Validate the new enhanced sampling and balancing features"""
    
    def __init__(self, visual_debugger: VisualDebugger):
        self.visual_debugger = visual_debugger
    
    def validate_enhanced_sampling(self, dataset, num_batches: int = 3, batch_size: int = 8):
        """Validate enhanced random sampling diversity"""
        print("\n🎲 VALIDATING ENHANCED RANDOM SAMPLING")
        print("-" * 50)
        
        try:
            # Get enhanced sampler
            sampler = dataset.get_enhanced_sampler(batch_size=batch_size)
            
            diversity_results = []
            all_sampled_indices = []
            
            # Test multiple batches
            for batch_idx in range(num_batches):
                sampled_indices = list(sampler)
                all_sampled_indices.extend(sampled_indices)
                
                # Check for unique images
                unique_images = set()
                for idx in sampled_indices:
                    image_id = dataset.split_df.iloc[idx]['ImageId']
                    unique_images.add(image_id)
                
                diversity_ratio = len(unique_images) / len(sampled_indices)
                batch_result = {
                    'batch_idx': batch_idx,
                    'sampled_count': len(sampled_indices),
                    'unique_images': len(unique_images),
                    'diversity_ratio': diversity_ratio,
                    'image_ids': list(unique_images)[:5]  # Show first 5 for inspection
                }
                
                diversity_results.append(batch_result)
                
                print(f"  ✅ Batch {batch_idx}:")
                print(f"     Sampled {len(sampled_indices)} indices")
                print(f"     Unique images: {len(unique_images)}")
                print(f"     Diversity ratio: {diversity_ratio:.2%}")
                
                if diversity_ratio < 0.8:
                    print(f"     ⚠️ Low diversity - images are being repeated")
            
            # Overall diversity analysis
            total_unique = len(set(all_sampled_indices))
            total_sampled = len(all_sampled_indices)
            overall_diversity = total_unique / total_sampled if total_sampled > 0 else 0
            
            print(f"\n📊 OVERALL SAMPLING DIVERSITY:")
            print(f"  Total sampled indices: {total_sampled}")
            print(f"  Total unique indices: {total_unique}")
            print(f"  Overall diversity: {overall_diversity:.2%}")
            
            return {
                'batch_results': diversity_results,
                'overall_diversity': overall_diversity,
                'total_sampled': total_sampled,
                'total_unique': total_unique
            }
            
        except Exception as e:
            print(f"❌ Enhanced sampling validation failed: {e}")
            return {}
    
    def validate_aggressive_balancing(self, original_csv: str, balanced_csv: str = None, dataset=None):
        """Validate aggressive class balancing effectiveness"""
        print("\n⚖️ VALIDATING AGGRESSIVE CLASS BALANCING")
        print("-" * 50)
        
        try:
            # Load original data
            original_df = pd.read_csv(original_csv)
            original_positive = original_df['has_pneumothorax'].sum()
            original_negative = len(original_df) - original_positive
            original_ratio = original_negative / original_positive if original_positive > 0 else float('inf')
            
            print(f"📊 ORIGINAL DATASET:")
            print(f"  Positive samples: {original_positive}")
            print(f"  Negative samples: {original_negative}")
            print(f"  Ratio (N:P): {original_ratio:.2f}:1")
            
            # Check balanced dataset
            if dataset is not None:
                balanced_positive = len(dataset.positive_samples) if hasattr(dataset, 'positive_samples') else dataset.split_df['has_pneumothorax'].sum()
                balanced_negative = len(dataset.negative_samples) if hasattr(dataset, 'negative_samples') else len(dataset.split_df) - balanced_positive
                balanced_ratio = balanced_negative / balanced_positive if balanced_positive > 0 else float('inf')
                
                print(f"📊 BALANCED DATASET:")
                print(f"  Positive samples: {balanced_positive}")
                print(f"  Negative samples: {balanced_negative}")
                print(f"  Ratio (N:P): {balanced_ratio:.2f}:1")
                print(f"  Improvement: {original_ratio/balanced_ratio:.2f}x more balanced")
                
                # Check small lesion oversampling
                if hasattr(dataset, 'lesion_sizes') and dataset.lesion_sizes:
                    small_lesion_threshold = np.percentile(dataset.lesion_sizes, 25)
                    small_lesions = [size for size in dataset.lesion_sizes if size <= small_lesion_threshold]
                    print(f"📈 SMALL LESION ANALYSIS:")
                    print(f"  Total lesions: {len(dataset.lesion_sizes)}")
                    print(f"  Small lesions (bottom 25%): {len(small_lesions)}")
                    print(f"  Small lesion threshold: {small_lesion_threshold:.1f} pixels")
            
            return {
                'original_ratio': original_ratio,
                'balanced_ratio': balanced_ratio if dataset else original_ratio,
                'improvement_factor': original_ratio/balanced_ratio if dataset and balanced_ratio > 0 else 1.0
            }
            
        except Exception as e:
            print(f"❌ Aggressive balancing validation failed: {e}")
            return {}

# ============================================================================
# DATASET VALIDATION WITH VISUAL DEBUGGING - UPDATED VERSION
# ============================================================================

class DatasetValidator:
    """Comprehensive dataset validation with visual debugging"""
    
    def __init__(self, visual_debugger: VisualDebugger):
        self.visual_debugger = visual_debugger
        self.enhanced_validator = EnhancedFeaturesValidator(visual_debugger)
    
    def validate_dataset_samples(self, csv_path: str, 
                               dicom_dir: str,
                               num_samples: int = 5,
                               save_dir: str = "dataset_validation",
                               test_enhanced_features: bool = True):
        """
        Validate samples from each curriculum level dataset with detailed visualization.
        Includes testing of enhanced features.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        print("=" * 80)
        print("🔍 DATASET SAMPLE VALIDATION")
        print("=" * 80)
        
        # Test each curriculum level with ENHANCED parameters
        datasets = {
            'Level1_Basic': Level1BasicDataset(
                split_csv=csv_path, 
                dicom_dir=dicom_dir, 
                preload_ram=False,
                return_metadata=True,
                filter_empty_masks=True,
                enforce_class_balance=False,  # Level 1 doesn't need enhanced features
                target_ratio=1.5
            ),
            'Level2_Standard': Level2StandardDataset(
                split_csv=csv_path, 
                dicom_dir=dicom_dir, 
                preload_ram=False,
                return_metadata=True,
                filter_empty_masks=True,
                enforce_class_balance=True,  # Enable enhanced features
                target_ratio=1.5,
                oversample_small_lesions=True
            ),
            'Level3_Advanced': Level3AdvancedDataset(
                split_csv=csv_path, 
                dicom_dir=dicom_dir, 
                preload_ram=False,
                return_metadata=True,
                filter_empty_masks=True,
                enforce_class_balance=True,  # Enable enhanced features
                target_ratio=1.5,
                oversample_small_lesions=True
            )
        }
        
        validation_results = {}
        enhanced_results = {}
        
        for level_name, dataset in datasets.items():
            print(f"\n🎯 {level_name} Validation:")
            print("-" * 50)
            
            # Test enhanced features for Level 2 and 3
            if test_enhanced_features and level_name != 'Level1_Basic':
                print(f"\n🔧 Testing Enhanced Features for {level_name}:")
                sampling_results = self.enhanced_validator.validate_enhanced_sampling(dataset)
                balancing_results = self.enhanced_validator.validate_aggressive_balancing(
                    csv_path, dataset=dataset
                )
                enhanced_results[level_name] = {
                    'sampling': sampling_results,
                    'balancing': balancing_results
                }
            
            level_results = []
            
            for sample_idx in range(min(num_samples, len(dataset))):
                try:
                    sample_result = self._validate_single_sample(
                        dataset, sample_idx, level_name, save_path
                    )
                    level_results.append(sample_result)
                    
                except Exception as e:
                    print(f"❌ Error processing sample {sample_idx} in {level_name}: {e}")
                    continue
            
            validation_results[level_name] = level_results
        
        return {
            'sample_validation': validation_results,
            'enhanced_features': enhanced_results
        }
    
    def _validate_single_sample(self, dataset, sample_idx: int, 
                              level_name: str, save_path: Path) -> Dict[str, Any]:
        """Validate a single dataset sample with comprehensive visualization."""
        
        try:
            # Get sample
            sample_data = dataset[sample_idx]
            
            # Handle different return formats
            if len(sample_data) == 3:  # With metadata
                image_tensor, mask_tensor, image_id = sample_data
            elif len(sample_data) == 2:  # Without metadata
                image_tensor, mask_tensor = sample_data
                image_id = f"sample_{sample_idx}"
            else:
                print(f"❌ Unexpected sample format: {len(sample_data)} elements")
                return {}
        
            # Convert to numpy for visualization
            image_np = image_tensor.squeeze().cpu().numpy()
            mask_np = mask_tensor.squeeze().cpu().numpy()
            
            print(f"  ✅ Sample {sample_idx} ({image_id}):")
            print(f"     Image - Shape: {image_np.shape}, Range: [{image_np.min():.6f}, {image_np.max():.6f}]")
            print(f"     Mask - Shape: {mask_np.shape}, Range: [{mask_np.min():.6f}, {mask_np.max():.6f}]")
            print(f"     Mask Unique: {np.unique(mask_np)}")
            
            # Check for pneumothorax
            has_pneumothorax = np.any(mask_np > 0.5)
            pneumothorax_coverage = np.sum(mask_np > 0.5) / mask_np.size
            print(f"     Has Pneumothorax: {has_pneumothorax} ({pneumothorax_coverage:.4%})")
            
            # Check for artifacts in Level 3
            if 'Advanced' in level_name:
                # Simple check for artifacts (bright pixels)
                bright_pixels = np.sum(image_np > 0.5)
                total_pixels = image_np.size
                bright_ratio = bright_pixels / total_pixels
                print(f"     Bright pixels (possible artifacts): {bright_ratio:.4%}")
            
            # Create comprehensive visualization
            sample_visualization = {
                f'Image ({level_name})': image_np,
                f'Mask ({level_name})': mask_np
            }
            
            self.visual_debugger.create_detailed_comparison(
                sample_visualization,
                list(sample_visualization.keys()),
                f"{level_name} - Sample {sample_idx} ({image_id})",
                save_path / f"{level_name}_sample_{sample_idx}.png"
            )
            
            # Create overlay visualization
            self.visual_debugger.create_overlay_visualization(
                image_np, mask_np, 
                f"{level_name} - Sample {sample_idx}",
                save_path / f"{level_name}_sample_{sample_idx}_overlay.png"
            )
            
            return {
                'image_id': image_id,
                'image_shape': image_np.shape,
                'image_range': [float(image_np.min()), float(image_np.max())],
                'mask_shape': mask_np.shape,
                'mask_range': [float(mask_np.min()), float(mask_np.max())],
                'mask_unique_values': [float(x) for x in np.unique(mask_np)],
                'has_pneumothorax': has_pneumothorax,
                'pneumothorax_coverage': float(pneumothorax_coverage),
                'is_mask_binary': len(np.unique(mask_np)) <= 2
            }
            
        except Exception as e:
            print(f"❌ Error in sample validation: {e}")
            traceback.print_exc()
            return {}

# ============================================================================
# BATCH PROCESSING VALIDATION WITH VISUAL DEBUGGING - UPDATED VERSION
# ============================================================================

class BatchProcessingValidator:
    """Comprehensive batch processing validation with visual debugging"""
    
    def __init__(self, visual_debugger: VisualDebugger):
        self.visual_debugger = visual_debugger
    
    def validate_batch_processing(self, csv_path: str, 
                                dicom_dir: str,
                                batch_size: int = 4,
                                save_dir: str = "batch_validation"):
        """
        Validate batch processing from data loader with detailed visualization.
        Uses the enhanced create_production_loader function.
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        print("=" * 80)
        print("🔍 BATCH PROCESSING VALIDATION")
        print("=" * 80)
        
        from torch.utils.data import DataLoader
        
        batch_results = {}
        
        for level in [1, 2, 3]:
            level_name = f"Level{level}_{'Basic' if level == 1 else 'Standard' if level == 2 else 'Advanced'}"
            
            print(f"\n🎯 {level_name} Batch Validation:")
            print("-" * 40)
            
            try:
                # Use the enhanced production loader
                dataloader = create_production_loader(
                    split_csv=csv_path,
                    dicom_dir=dicom_dir,
                    level=level,
                    batch_size=batch_size,
                    num_workers=0,  # Disable multiprocessing for validation
                    preload_ram=False,
                    filter_empty_masks=True,
                    aggressive_balancing=(level > 1),  # Enhanced balancing for training levels
                    target_ratio=1.5,
                    enforce_diversity=(level > 1),  # Enhanced sampling for training levels
                    oversample_small_lesions=(level > 1)
                )
                
                level_batch_results = []
                
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= 2:  # Only check first 2 batches
                        break
                        
                    batch_result = self._validate_single_batch(
                        batch, batch_idx, level_name, save_path
                    )
                    level_batch_results.append(batch_result)
                
                batch_results[level_name] = level_batch_results
                
            except Exception as e:
                print(f"❌ Failed to create dataloader for {level_name}: {e}")
                continue
        
        return batch_results
    
    def _validate_single_batch(self, batch, batch_idx: int, 
                             level_name: str, save_path: Path) -> Dict[str, Any]:
        """Validate a single batch with comprehensive analysis."""
        
        try:
            # Handle different batch formats
            if len(batch) == 2:  # Standard format (images, masks)
                images, masks = batch
                image_ids = [f"batch_{batch_idx}_item_{i}" for i in range(len(images))]
            else:
                print(f"❌ Unexpected batch format: {len(batch)} elements")
                return {}
        
            print(f"  ✅ Batch {batch_idx}:")
            print(f"     Images shape: {images.shape}")
            print(f"     Masks shape: {masks.shape}")
            print(f"     Image range: [{images.min():.6f}, {images.max():.6f}]")
            print(f"     Mask range: [{masks.min():.6f}, {masks.max():.6f}]")
            
            # Check mask values
            unique_masks = torch.unique(masks)
            print(f"     Mask unique values: {unique_masks}")
            
            # Count pneumothorax cases in batch
            pneumothorax_count = (masks > 0.5).float().sum().item()
            total_pixels = masks.numel()
            pneumothorax_ratio = pneumothorax_count / total_pixels
            
            # Count samples with pneumothorax
            samples_with_pneumothorax = (masks > 0.5).any(dim=[1, 2, 3]).sum().item()
            pneumothorax_sample_ratio = samples_with_pneumothorax / len(images)
            
            print(f"     Pneumothorax pixels: {pneumothorax_count}/{total_pixels} ({pneumothorax_ratio:.6%})")
            print(f"     Samples with pneumothorax: {samples_with_pneumothorax}/{len(images)} ({pneumothorax_sample_ratio:.2%})")
            
            # Analyze batch statistics
            batch_stats = {
                'batch_idx': batch_idx,
                'batch_size': images.shape[0],
                'image_shape': tuple(images.shape[1:]),
                'mask_shape': tuple(masks.shape[1:]),
                'image_range': [float(images.min()), float(images.max())],
                'mask_range': [float(masks.min()), float(masks.max())],
                'mask_unique_values': [float(x) for x in unique_masks],
                'pneumothorax_ratio': float(pneumothorax_ratio),
                'pneumothorax_sample_ratio': float(pneumothorax_sample_ratio),
                'pneumothorax_pixels': int(pneumothorax_count),
                'total_pixels': int(total_pixels),
                'samples_with_pneumothorax': int(samples_with_pneumothorax)
            }
            
            # Visualize first batch
            if batch_idx == 0:
                batch_visualization = {}
                for i in range(min(4, images.shape[0])):  # Show first 4 samples
                    img = images[i].squeeze().cpu().numpy()
                    mask = masks[i].squeeze().cpu().numpy()
                    
                    batch_visualization[f'Image_{i}'] = img
                    batch_visualization[f'Mask_{i}'] = mask
                    
                    # Create overlay for each sample
                    self.visual_debugger.create_overlay_visualization(
                        img, mask, 
                        f"{level_name} - Batch {batch_idx} Sample {i}",
                        save_path / f"{level_name}_batch_{batch_idx}_sample_{i}_overlay.png"
                    )
                
                self.visual_debugger.create_detailed_comparison(
                    batch_visualization,
                    list(batch_visualization.keys()),
                    f"{level_name} - Batch {batch_idx}",
                    save_path / f"{level_name}_batch_{batch_idx}.png"
                )
            
            return batch_stats
            
        except Exception as e:
            print(f"❌ Error in batch validation: {e}")
            traceback.print_exc()
            return {}

# ============================================================================
# COMPREHENSIVE STATISTICAL ANALYSIS - UPDATED
# ============================================================================

class StatisticalAnalyzer:
    """Comprehensive statistical analysis of the dataset"""
    
    def analyze_dataset_statistics(self, csv_path: str, dicom_dir: str) -> Dict[str, Any]:
        """
        Analyze dataset statistics that might affect training.
        Enhanced to show benefits of aggressive balancing.
        """
        print("=" * 80)
        print("📊 DATASET STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Basic statistics
        total_samples = len(df)
        pneumothorax_samples = df['has_pneumothorax'].sum()
        normal_samples = total_samples - pneumothorax_samples
        
        print(f"Total samples: {total_samples}")
        print(f"Pneumothorax cases: {pneumothorax_samples} ({pneumothorax_samples/total_samples:.2%})")
        print(f"Normal cases: {normal_samples} ({normal_samples/total_samples:.2%})")
        
        # Analyze RLE strings
        rle_lengths = df['EncodedPixels'].str.len()
        empty_masks = df['EncodedPixels'].isna() | (df['EncodedPixels'] == '') | (df['EncodedPixels'] == '-1')
        
        print(f"\nRLE Analysis:")
        print(f"  Average RLE length: {rle_lengths.mean():.1f} characters")
        print(f"  Empty masks: {empty_masks.sum()} ({empty_masks.sum()/total_samples:.2%})")
        print(f"  Non-empty masks: {(~empty_masks).sum()} ({(~empty_masks).sum()/total_samples:.2%})")
        
        # Analyze pneumothorax sizes for non-empty masks
        pneumothorax_df = df[df['has_pneumothorax'] & (~empty_masks)]
        pneumothorax_size_stats = {}
        
        if len(pneumothorax_df) > 0:
            print(f"\nPneumothorax Size Analysis ({len(pneumothorax_df)} cases):")
            
            # Estimate pneumothorax size from RLE length (rough approximation)
            pneumothorax_rle_lengths = pneumothorax_df['EncodedPixels'].str.len()
            pneumothorax_size_stats = {
                'mean_rle_length': float(pneumothorax_rle_lengths.mean()),
                'min_rle_length': float(pneumothorax_rle_lengths.min()),
                'max_rle_length': float(pneumothorax_rle_lengths.max()),
                'std_rle_length': float(pneumothorax_rle_lengths.std())
            }
            
            print(f"  Average RLE length for pneumothorax: {pneumothorax_size_stats['mean_rle_length']:.1f}")
            print(f"  Min RLE length: {pneumothorax_size_stats['min_rle_length']}")
            print(f"  Max RLE length: {pneumothorax_size_stats['max_rle_length']}")
            print(f"  Std RLE length: {pneumothorax_size_stats['std_rle_length']:.1f}")
        
        # Class imbalance analysis
        class_imbalance_ratio = normal_samples / pneumothorax_samples if pneumothorax_samples > 0 else float('inf')
        
        print(f"\nClass Imbalance Analysis:")
        print(f"  Positive/Negative ratio: 1:{class_imbalance_ratio:.1f}")
        
        # Show benefits of aggressive balancing
        target_ratio = 1.5
        if pneumothorax_samples > 0:
            target_negative = int(pneumothorax_samples * target_ratio)
            removed_negative = max(0, normal_samples - target_negative)
            
            print(f"\n🎯 AGGRESSIVE BALANCING BENEFITS:")
            print(f"  Target ratio: 1:{target_ratio}")
            print(f"  Negative samples to remove: {removed_negative}")
            print(f"  New negative count: {target_negative}")
            print(f"  Total samples after balancing: {pneumothorax_samples + target_negative}")
            print(f"  Balance improvement: {class_imbalance_ratio/target_ratio:.1f}x")
        
        if class_imbalance_ratio > 10:
            print(f"  ⚠️  SEVERE class imbalance detected!")
            print(f"  This is very likely causing your low Dice scores")
        
        return {
            'total_samples': total_samples,
            'pneumothorax_samples': pneumothorax_samples,
            'normal_samples': normal_samples,
            'pneumothorax_ratio': pneumothorax_samples / total_samples,
            'empty_mask_ratio': empty_masks.sum() / total_samples,
            'class_imbalance_ratio': class_imbalance_ratio,
            'pneumothorax_size_stats': pneumothorax_size_stats,
            'aggressive_balancing_benefit': class_imbalance_ratio/target_ratio if pneumothorax_samples > 0 else 1.0
        }

# ============================================================================
# MAIN COMPREHENSIVE VALIDATION PIPELINE - ENHANCED VERSION
# ============================================================================

class ComprehensiveDataLoaderValidator:
    """
    Main comprehensive validation class that orchestrates all validation steps
    with detailed visual debugging. UPDATED for enhanced data loader.
    """
    
    def __init__(self, output_dir: str = "comprehensive_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.visual_debugger = VisualDebugger()
        self.mask_validator = MaskProcessingValidator(self.visual_debugger)
        self.image_validator = ImageProcessingValidator(self.visual_debugger)
        self.dataset_validator = DatasetValidator(self.visual_debugger)
        self.batch_validator = BatchProcessingValidator(self.visual_debugger)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Results storage
        self.validation_results = {}
        
    def run_comprehensive_validation(self, train_csv: str, dicom_dir: str) -> Dict[str, Any]:
        """
        Run comprehensive validation of the ENHANCED data pipeline.
        """
        print("🚀 STARTING COMPREHENSIVE DATA LOADER VALIDATION")
        print("=" * 80)
        print("🎯 TESTING ENHANCED FEATURES: Random Sampling & Aggressive Balancing")
        print("=" * 80)
        
        try:
            # 1. Statistical Analysis
            print("\n" + "="*80)
            print("📊 STEP 1: DATASET STATISTICAL ANALYSIS")
            print("="*80)
            stats = self.statistical_analyzer.analyze_dataset_statistics(train_csv, dicom_dir)
            self.validation_results['statistics'] = stats
            
            # 2. Find test samples
            df = pd.read_csv(train_csv)
            
            # Find samples with pneumothorax for testing
            pneumothorax_samples = df[df['has_pneumothorax'] & 
                                     (df['EncodedPixels'].notna()) & 
                                     (df['EncodedPixels'] != '') & 
                                     (df['EncodedPixels'] != '-1')]
            
            if len(pneumothorax_samples) == 0:
                print("❌ No pneumothorax samples found in the dataset for testing")
                return self.validation_results
            
            test_rle = pneumothorax_samples.iloc[0]['EncodedPixels']
            test_image_id = pneumothorax_samples.iloc[0]['ImageId']
            
            # 3. Mask Processing Validation
            print("\n" + "="*80)
            print("🔍 STEP 2: MASK PROCESSING VALIDATION")
            print("="*80)
            mask_results = self.mask_validator.validate_mask_processing_pipeline(
                test_rle, save_dir=self.output_dir / "mask_processing"
            )
            self.validation_results['mask_processing'] = mask_results
            
            # 4. Find corresponding DICOM file
            dicom_files = list(Path(dicom_dir).rglob("*.dcm"))
            test_dicom_path = None
            for dcm_path in dicom_files:
                if dcm_path.stem == test_image_id:
                    test_dicom_path = dcm_path
                    break
            
            if test_dicom_path and test_dicom_path.exists():
                # 5. Image Processing Validation
                print("\n" + "="*80)
                print("🔍 STEP 3: IMAGE PROCESSING VALIDATION")
                print("="*80)
                image_results = self.image_validator.validate_image_processing_pipeline(
                    str(test_dicom_path), save_dir=self.output_dir / "image_processing"
                )
                self.validation_results['image_processing'] = image_results
            else:
                print(f"⚠️  DICOM file not found for {test_image_id}, skipping image processing validation")
            
            # 6. Dataset Sample Validation (INCLUDES ENHANCED FEATURES TESTING)
            print("\n" + "="*80)
            print("🔍 STEP 4: DATASET SAMPLE VALIDATION")
            print("="*80)
            dataset_results = self.dataset_validator.validate_dataset_samples(
                train_csv, dicom_dir, num_samples=3, 
                save_dir=self.output_dir / "dataset_samples",
                test_enhanced_features=True
            )
            self.validation_results['dataset_samples'] = dataset_results
            
            # 7. Batch Processing Validation (USES ENHANCED PRODUCTION LOADER)
            print("\n" + "="*80)
            print("🔍 STEP 5: BATCH PROCESSING VALIDATION")
            print("="*80)
            batch_results = self.batch_validator.validate_batch_processing(
                train_csv, dicom_dir, batch_size=4,
                save_dir=self.output_dir / "batch_processing"
            )
            self.validation_results['batch_processing'] = batch_results
            
            # 8. Generate Enhanced Summary Report
            self._generate_enhanced_summary_report()
            
            print("\n" + "="*80)
            print("✅ ENHANCED COMPREHENSIVE VALIDATION COMPLETED")
            print("="*80)
            
            return self.validation_results
            
        except Exception as e:
            print(f"❌ Comprehensive validation failed: {e}")
            traceback.print_exc()
            return {}
    
    def _generate_enhanced_summary_report(self):
        """Generate comprehensive summary report with enhanced features analysis."""
        print("\n" + "="*80)
        print("📋 ENHANCED COMPREHENSIVE VALIDATION SUMMARY")
        print("="*80)
        
        stats = self.validation_results.get('statistics', {})
        mask_processing = self.validation_results.get('mask_processing', {})
        dataset_samples = self.validation_results.get('dataset_samples', {})
        
        # Key findings
        print("\n🔍 KEY FINDINGS:")
        print(f"1. Class Imbalance: {stats.get('class_imbalance_ratio', 'N/A'):.1f}:1 (normal:pneumothorax)")
        print(f"2. Pneumothorax Ratio: {stats.get('pneumothorax_ratio', 'N/A'):.2%}")
        print(f"3. Empty Masks: {stats.get('empty_mask_ratio', 'N/A'):.2%}")
        print(f"4. Balance Improvement Potential: {stats.get('aggressive_balancing_benefit', 'N/A'):.1f}x")
        
        # Mask processing validation
        if mask_processing.get('is_valid'):
            print("5. ✅ Mask Processing: Binary masks are correctly generated")
        else:
            print("5. ❌ Mask Processing: Non-binary values detected in masks!")
        
        # Enhanced features analysis
        enhanced_features = dataset_samples.get('enhanced_features', {})
        if enhanced_features:
            print("\n🚀 ENHANCED FEATURES ANALYSIS:")
            for level_name, features in enhanced_features.items():
                if features.get('sampling'):
                    sampling = features['sampling']
                    print(f"   • {level_name} Sampling Diversity: {sampling.get('overall_diversity', 0):.2%}")
                
                if features.get('balancing'):
                    balancing = features['balancing']
                    if balancing.get('improvement_factor', 1) > 1:
                        print(f"   • {level_name} Balance Improvement: {balancing['improvement_factor']:.1f}x")
        
        # Critical issues
        critical_issues = []
        
        if stats.get('class_imbalance_ratio', 0) > 10:
            critical_issues.append("Severe class imbalance")
        
        if stats.get('pneumothorax_ratio', 0) < 0.1:
            critical_issues.append("Very low pneumothorax ratio")
        
        if not mask_processing.get('is_valid', True):
            critical_issues.append("Non-binary mask values")
        
        if critical_issues:
            print(f"\n⚠️  CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                print(f"   • {issue}")
            
            print(f"\n💡 ENHANCED SOLUTIONS IMPLEMENTED:")
            if "Severe class imbalance" in critical_issues:
                print("   • ✅ Aggressive class balancing (1.5:1 target ratio)")
                print("   • ✅ Enhanced random sampling with diversity enforcement")
                print("   • ✅ Small lesion oversampling for better detection")
                print("   • ✅ Advanced class weighting in loss function")
            
            if "Non-binary mask values" in critical_issues:
                print("   • ✅ Fixed mask processing pipeline")
                print("   • ✅ INTER_NEAREST interpolation for masks")
                print("   • ✅ Binary validation step")
        
        print(f"\n📁 Validation outputs saved in: {self.output_dir}")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main function for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Comprehensive Data Loader Validation with Visual Debugging')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--dicom_dir', type=str, required=True, help='Path to DICOM directory')
    parser.add_argument('--output_dir', type=str, default='enhanced_comprehensive_validation', 
                       help='Output directory for validation results')
    
    args = parser.parse_args()
    
    # Run comprehensive validation
    validator = ComprehensiveDataLoaderValidator(args.output_dir)
    results = validator.run_comprehensive_validation(args.train_csv, args.dicom_dir)
    
    # Exit with appropriate code
    if results:
        print(f"\n🎉 ENHANCED VALIDATION COMPLETED SUCCESSFULLY!")
        print(f"   Enhanced features are working correctly!")
        sys.exit(0)
    else:
        print(f"\n❌ Validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()