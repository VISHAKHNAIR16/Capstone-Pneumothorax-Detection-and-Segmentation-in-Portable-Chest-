"""
Test preprocessing pipeline on random images
Visualize each step of preprocessing
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import PreprocessingPipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'visualize')))
from visualization import PneumothoraxVisualizer


def test_preprocessing_on_random_images(
    dicom_dir: str,
    csv_path: str,
    num_samples: int = 10,
    save_visualizations: bool = True,
    keep_only_one: bool = False  # <-- New flag
):
    """
    Test preprocessing on random images and visualize
    
    Args:
        dicom_dir: Directory containing DICOM files
        csv_path: Path to CSV with image IDs
        num_samples: Number of random images to test
        save_visualizations: Whether to save visualization PNG files
        keep_only_one: If True, stop after processing 1 image (for quick test)
    """
    
    preprocessor = PreprocessingPipeline()
    visualizer = PneumothoraxVisualizer()
    
    # Find DICOM files
    print("Finding DICOM files...")
    dicom_files = []
    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    # Sample random files
    random_files = random.sample(dicom_files, min(num_samples, len(dicom_files)))
    
    # Load CSV for reference
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['EncodedPixels'] = df['EncodedPixels'].astype(str).str.strip()
    
    print(f"\nTesting preprocessing on {len(random_files)} random images...\n")
    
    results = []
    successful = 0
    failed = 0
    
    for idx, dcm_path in enumerate(random_files, 1):
        try:
            print(f"[{idx}/{len(random_files)}] Processing {os.path.basename(dcm_path)}...")
            
            # Preprocess
            preprocessed_img, metadata = preprocessor.preprocess(dcm_path)
            
            # Verify output
            assert preprocessed_img.shape == (512, 512), f"Wrong shape: {preprocessed_img.shape}"
            assert preprocessed_img.min() >= -1.0, f"Min out of range: {preprocessed_img.min()}"
            assert preprocessed_img.max() <= 1.0, f"Max out of range: {preprocessed_img.max()}"
            
            # Visualize
            visualizer.visualize_quality_issues(preprocessed_img)
            
            # Save visualization
            if save_visualizations:
                plt.savefig(f'preprocessed_{idx:02d}.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            results.append({
                'image_id': idx,
                'filename': os.path.basename(dcm_path),
                'shape': preprocessed_img.shape,
                'min': preprocessed_img.min(),
                'max': preprocessed_img.max(),
                'mean': preprocessed_img.mean(),
                'std': preprocessed_img.std(),
                'status': 'SUCCESS'
            })
            
            successful += 1
            print(f"  ✓ Success\n")
            
            if keep_only_one:
                print("Stopping after processing one image as requested.\n")
                break
            
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            results.append({
                'image_id': idx,
                'filename': os.path.basename(dcm_path),
                'status': 'FAILED',
                'error': str(e)
            })
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING TEST SUMMARY")
    print("="*60)
    print(f"Total images tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(results)*100):.1f}%")
    
    if successful > 0:
        print("\nOutput Statistics:")
        successful_results = [r for r in results if r['status'] == 'SUCCESS']
        shapes = [r['shape'] for r in successful_results]
        mins = [r['min'] for r in successful_results]
        maxs = [r['max'] for r in successful_results]
        means = [r['mean'] for r in successful_results]
        stds = [r['std'] for r in successful_results]
        
        print(f"  All shapes: {set(shapes)}")
        print(f"  Min range: [{min(mins):.4f}, {max(mins):.4f}]")
        print(f"  Max range: [{min(maxs):.4f}, {max(maxs):.4f}]")
        print(f"  Mean range: [{min(means):.4f}, {max(means):.4f}]")
        print(f"  Std range: [{min(stds):.4f}, {max(stds):.4f}]")
    print("="*60)
    
    return results



if __name__ == "__main__":
    # Configuration
    DICOM_DIR = "../../../Data/siim-original/dicom-images-train"
    CSV_PATH = "../../../Data/train-rle.csv"
    NUM_SAMPLES = 10
    
    # Run test with keep_only_one=True to only see one image
    results = test_preprocessing_on_random_images(
        DICOM_DIR,
        CSV_PATH,
        num_samples=NUM_SAMPLES,
        save_visualizations=True,
        keep_only_one=True
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('preprocessing_test_results.csv', index=False)
    
    print("\n✓ Results saved to preprocessing_test_results.csv")
