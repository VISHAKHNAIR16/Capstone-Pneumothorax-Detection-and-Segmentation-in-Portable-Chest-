"""
Visualization utilities for pneumothorax detection project
Provides reusable functions for visualizing images, masks, and augmentations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, List
import pydicom
import pandas as pd


class PneumothoraxVisualizer:
    """
    Comprehensive visualization utility for pneumothorax detection project
    Handles DICOM images, masks, augmentations, and comparisons
    """
    
    # =====================================================
    # SECTION 1: Basic Visualization
    # =====================================================
    
    @staticmethod
    def visualize_image_mask(
        image: np.ndarray,
        mask: np.ndarray,
        title: str = "Image with Mask",
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Visualize image and mask side by side
        
        Args:
            image: Input image (grayscale, uint8 or float)
            mask: Binary mask (0 or 255)
            title: Title for the figure
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image with mask overlay
        axes[1].imshow(image, cmap='gray')
        axes[1].imshow(mask, alpha=0.4, cmap='Reds')
        axes[1].set_title('With Pneumothorax Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    @staticmethod
    def visualize_three_panel(
        image: np.ndarray,
        mask: np.ndarray,
        title: str = "Image Analysis",
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Visualize image, mask, and overlay in three panels
        
        Args:
            image: Input image
            mask: Binary mask
            title: Title for figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Panel 1: Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Panel 2: Mask only
        axes[1].imshow(mask, cmap='Reds')
        axes[1].set_title('Pneumothorax Mask')
        axes[1].axis('off')
        
        # Panel 3: Overlay
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(mask, alpha=0.5, cmap='Reds')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    # =====================================================
    # SECTION 2: Batch Visualization
    # =====================================================
    
    @staticmethod
    def visualize_batch(
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        num_samples: int = 8,
        figsize: Tuple[int, int] = (16, 8)
    ) -> None:
        """
        Visualize a batch of images and optional masks in a grid
        
        Args:
            images: Batch of images (N, H, W)
            masks: Batch of masks (N, H, W) or None
            num_samples: How many samples to display
            figsize: Figure size
        """
        num_samples = min(num_samples, len(images))
        
        if masks is not None:
            fig, axes = plt.subplots(2, num_samples, figsize=figsize)
            fig.suptitle(f'Batch Visualization ({num_samples} samples)', 
                        fontsize=14, fontweight='bold')
            
            for idx in range(num_samples):
                # Images row
                axes[0, idx].imshow(images[idx], cmap='gray')
                axes[0, idx].set_title(f'Image {idx}')
                axes[0, idx].axis('off')
                
                # Masks row
                axes[1, idx].imshow(masks[idx], cmap='Reds')
                axes[1, idx].set_title(f'Mask {idx}')
                axes[1, idx].axis('off')
        else:
            fig, axes = plt.subplots(1, num_samples, figsize=figsize)
            fig.suptitle(f'Batch Images ({num_samples} samples)', 
                        fontsize=14, fontweight='bold')
            
            for idx in range(num_samples):
                axes[idx].imshow(images[idx], cmap='gray')
                axes[idx].set_title(f'Image {idx}')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    @staticmethod
    def visualize_augmentation_comparison(
        original_image: np.ndarray,
        original_mask: np.ndarray,
        augmented_images: List[np.ndarray],
        augmented_masks: List[np.ndarray],
        augmentation_names: List[str],
        figsize: Tuple[int, int] = (16, 10)
    ) -> None:
        """
        Visualize original and multiple augmented versions side by side
        
        Args:
            original_image: Original image
            original_mask: Original mask
            augmented_images: List of augmented images
            augmented_masks: List of augmented masks
            augmentation_names: List of augmentation names
            figsize: Figure size
        """
        num_augmentations = len(augmented_images)
        fig, axes = plt.subplots(2, num_augmentations + 1, figsize=figsize)
        fig.suptitle('Augmentation Comparison', fontsize=14, fontweight='bold')
        
        # Original
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(original_mask, cmap='Reds')
        axes[1, 0].set_title('Original Mask')
        axes[1, 0].axis('off')
        
        # Augmented versions
        for idx, (aug_img, aug_mask, name) in enumerate(
            zip(augmented_images, augmented_masks, augmentation_names)
        ):
            col = idx + 1
            
            axes[0, col].imshow(aug_img, cmap='gray')
            axes[0, col].set_title(f'{name}')
            axes[0, col].axis('off')
            
            axes[1, col].imshow(aug_mask, cmap='Reds')
            axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    # =====================================================
    # SECTION 3: Preprocessing Visualization
    # =====================================================
    
    @staticmethod
    def visualize_preprocessing_steps(
        original_image: np.ndarray,
        windowed_image: np.ndarray,
        normalized_image: np.ndarray,
        resized_image: np.ndarray,
        figsize: Tuple[int, int] = (16, 4)
    ) -> None:
        """
        Visualize preprocessing pipeline steps
        
        Args:
            original_image: Raw DICOM image
            windowed_image: After windowing/leveling
            normalized_image: After normalization
            resized_image: After resizing
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        fig.suptitle('Preprocessing Pipeline', fontsize=14, fontweight='bold')
        
        steps = [
            ('Original (Raw DICOM)', original_image),
            ('After Windowing', windowed_image),
            ('After Normalization', normalized_image),
            ('After Resizing', resized_image)
        ]
        
        for idx, (title, img) in enumerate(steps):
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(title)
            axes[idx].axis('off')
            
            # Add statistics
            stats_text = f'Min: {img.min():.2f}\nMax: {img.max():.2f}\nMean: {img.mean():.2f}'
            axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes,
                          fontsize=8, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    
    # =====================================================
    # SECTION 4: Statistical Visualization
    # =====================================================
    
    @staticmethod
    def visualize_histogram_comparison(
        original_image: np.ndarray,
        processed_image: np.ndarray,
        figsize: Tuple[int, int] = (14, 4)
    ) -> None:
        """
        Compare histograms of original and processed images
        
        Args:
            original_image: Original image
            processed_image: Processed image
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Histogram Comparison', fontsize=14, fontweight='bold')
        
        # Original histogram
        axes[0].hist(original_image.flatten(), bins=256, color='blue', alpha=0.7)
        axes[0].set_title('Original Image Histogram')
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Processed histogram
        axes[1].hist(processed_image.flatten(), bins=256, color='green', alpha=0.7)
        axes[1].set_title('Processed Image Histogram')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    
    # =====================================================
    # SECTION 5: Mask Analysis
    # =====================================================
    
    @staticmethod
    def visualize_mask_statistics(
        image: np.ndarray,
        mask: np.ndarray,
        figsize: Tuple[int, int] = (14, 5)
    ) -> None:
        """
        Visualize mask with statistics (size, area, bounding box)
        
        Args:
            image: Original image
            mask: Binary mask
            figsize: Figure size
        """
        # Calculate statistics
        ptx_pixels = (mask == 255).sum()
        total_pixels = mask.size
        percentage = (ptx_pixels / total_pixels) * 100
        
        # Find bounding box
        rows = np.any(mask == 255, axis=1)
        cols = np.any(mask == 255, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_height = rmax - rmin
            bbox_width = cmax - cmin
        else:
            rmin = rmax = cmin = cmax = 0
            bbox_height = bbox_width = 0
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Mask Analysis', fontsize=14, fontweight='bold')
        
        # Image with overlay and bounding box
        axes[0].imshow(image, cmap='gray')
        axes[0].imshow(mask, alpha=0.4, cmap='Reds')
        
        # Draw bounding box
        if bbox_height > 0 and bbox_width > 0:
            rect = patches.Rectangle(
                (cmin, rmin), bbox_width, bbox_height,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            axes[0].add_patch(rect)
        
        axes[0].set_title('Image with Bounding Box')
        axes[0].axis('off')
        
        # Statistics
        stats_text = (
            f'Pneumothorax Pixels: {ptx_pixels:,}\n'
            f'Total Pixels: {total_pixels:,}\n'
            f'Percentage: {percentage:.3f}%\n\n'
            f'Bounding Box:\n'
            f'  Height: {bbox_height} pixels\n'
            f'  Width: {bbox_width} pixels\n'
            f'  Area: {bbox_height * bbox_width:,} pixels'
        )
        
        axes[1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    # =====================================================
    # SECTION 6: Quality Control
    # =====================================================
    
    @staticmethod
    def visualize_quality_issues(
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Visualize image quality metrics and potential issues
        
        Args:
            image: Image to analyze
            mask: Optional mask
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Quality Control', fontsize=14, fontweight='bold')
        
        # Image with grid overlay
        axes[0].imshow(image, cmap='gray')
        
        # Add grid to detect artifacts
        h, w = image.shape
        for i in range(0, h, 128):
            axes[0].axhline(y=i, color='cyan', linewidth=0.5, alpha=0.3)
        for j in range(0, w, 128):
            axes[0].axvline(x=j, color='cyan', linewidth=0.5, alpha=0.3)
        
        axes[0].set_title('Image with Grid')
        axes[0].axis('off')
        
        # Quality metrics
        quality_text = (
            f'Image Shape: {image.shape}\n'
            f'Image Dtype: {image.dtype}\n'
            f'Min Value: {image.min():.2f}\n'
            f'Max Value: {image.max():.2f}\n'
            f'Mean Value: {image.mean():.2f}\n'
            f'Std Dev: {image.std():.2f}\n'
            f'Range: {image.max() - image.min():.2f}\n'
        )
        
        if mask is not None:
            ptx_pixels = (mask == 255).sum()
            quality_text += f'\nMask Pixels: {ptx_pixels:,}'
        
        axes[1].text(0.1, 0.5, quality_text, fontsize=11, verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    # =====================================================
    # SECTION 7: Utility Functions
    # =====================================================
    
    @staticmethod
    def save_figure(
        filename: str,
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save current figure to file
        
        Args:
            filename: Output filename (e.g., 'output.png')
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting ('tight' recommended)
        """
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"✓ Figure saved as {filename}")
    
    
    @staticmethod
    def close_all_figures() -> None:
        """Close all open figures"""
        plt.close('all')
    
    
    @staticmethod
    def set_style(style: str = 'seaborn-v0_8') -> None:
        """
        Set matplotlib style
        
        Args:
            style: Style name ('seaborn-v0_8', 'ggplot', 'default', etc.)
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')


# =====================================================
# SECTION 8: Convenience Functions
# =====================================================

def visualize_single_image(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "Image Visualization"
) -> None:
    """
    Quick function to visualize a single image with optional mask
    
    Args:
        image: Image array
        mask: Optional mask array
        title: Title for visualization
    """
    visualizer = PneumothoraxVisualizer()
    
    if mask is not None:
        visualizer.visualize_three_panel(image, mask, title=title)
    else:
        visualizer.visualize_quality_issues(image, title=title)


def visualize_preprocessing(
    original: np.ndarray,
    windowed: np.ndarray,
    normalized: np.ndarray,
    resized: np.ndarray
) -> None:
    """
    Quick function to visualize preprocessing steps
    """
    visualizer = PneumothoraxVisualizer()
    visualizer.visualize_preprocessing_steps(original, windowed, normalized, resized)


def visualize_augmentations(
    original_image: np.ndarray,
    original_mask: np.ndarray,
    augmented_images: List[np.ndarray],
    augmented_masks: List[np.ndarray],
    names: List[str]
) -> None:
    """
    Quick function to visualize augmentations
    """
    visualizer = PneumothoraxVisualizer()
    visualizer.visualize_augmentation_comparison(
        original_image, original_mask,
        augmented_images, augmented_masks,
        names
    )
