"""
Analytics: Compare different preprocessing methods
Run multiple windowing, CLAHE, normalization, and resize strategies.
Generates detailed comparison CSVs and prints best configurations.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import PreprocessingPipeline

class PreprocessingAnalyzer:
    """
    Analyze and compare different preprocessing configurations
    """

    def __init__(self, dicom_dir: str, num_samples: int = 5, keep_only_one: bool = False):
        self.dicom_dir = dicom_dir
        self.num_samples = num_samples
        self.keep_only_one = keep_only_one
        self.preprocessor = PreprocessingPipeline()

        # Find DICOM files
        self.dicom_files = []
        for root, _, files in os.walk(dicom_dir):
            for file in files:
                if file.endswith('.dcm'):
                    self.dicom_files.append(os.path.join(root, file))
        self.sample_files = random.sample(self.dicom_files, min(num_samples, len(self.dicom_files)))
        print(f"Selected {len(self.sample_files)} images for analysis")

    def analyze_windowing_methods(self) -> pd.DataFrame:
        print("\n" + "="*60)
        print("WINDOWING ANALYSIS")
        print("="*60)

        windowing_configs = [
            ("Standard Chest X-ray", 40, 350),
            ("High Contrast", 40, 200),
            ("Low Contrast", 40, 500),
            ("Alternative", 50, 400),
        ]
        results = []

        test_files = self.sample_files[:1] if self.keep_only_one else self.sample_files[:3]
        for dcm_path in test_files:
            img, _ = self.preprocessor.load_dicom(dcm_path)
            for config_name, wl, ww in windowing_configs:
                windowed, _ = self.preprocessor.apply_windowing(img, wl, ww)
                results.append({
                    'image': os.path.basename(dcm_path),
                    'config': config_name,
                    'window_level': wl,
                    'window_width': ww,
                    'min': windowed.min(),
                    'max': windowed.max(),
                    'mean': windowed.mean(),
                    'std': windowed.std(),
                })
                print(f"✓ {config_name} (WL={wl}, WW={ww}): mean={windowed.mean():.1f}, std={windowed.std():.1f}")
        return pd.DataFrame(results)

    def analyze_clahe_impact(self) -> pd.DataFrame:
        print("\n" + "="*60)
        print("CLAHE IMPACT ANALYSIS")
        print("="*60)
        clahe_configs = [
            ("No CLAHE", None, None),
            ("Low CLAHE", 1.0, 8),
            ("Standard CLAHE", 2.0, 8),
            ("High CLAHE", 4.0, 8),
        ]
        results = []

        test_files = self.sample_files[:1] if self.keep_only_one else self.sample_files[:3]
        for dcm_path in test_files:
            img, _ = self.preprocessor.load_dicom(dcm_path)
            windowed, _ = self.preprocessor.apply_windowing(img)
            for config_name, clip_limit, tile_size in clahe_configs:
                if clip_limit is None:
                    enhanced = windowed.copy()
                else:
                    enhanced, _ = self.preprocessor.apply_clahe(windowed, clip_limit, tile_size)
                results.append({
                    'image': os.path.basename(dcm_path),
                    'config': config_name,
                    'clip_limit': clip_limit,
                    'tile_size': tile_size,
                    'min': enhanced.min(),
                    'max': enhanced.max(),
                    'mean': enhanced.mean(),
                    'std': enhanced.std(),
                })
                print(f"✓ {config_name}: mean={enhanced.mean():.1f}, std={enhanced.std():.1f}")
        return pd.DataFrame(results)

    def analyze_resize_methods(self) -> pd.DataFrame:
        print("\n" + "="*60)
        print("RESIZE INTERPOLATION ANALYSIS")
        print("="*60)
        resize_methods = [
            ("Nearest", cv2.INTER_NEAREST),
            ("Linear", cv2.INTER_LINEAR),
            ("Cubic", cv2.INTER_CUBIC),
            ("Lanczos4", cv2.INTER_LANCZOS4),
        ]
        results = []

        test_files = self.sample_files[:1] if self.keep_only_one else self.sample_files[:3]
        for dcm_path in test_files:
            img, _ = self.preprocessor.load_dicom(dcm_path)
            windowed, _ = self.preprocessor.apply_windowing(img)
            enhanced, _ = self.preprocessor.apply_clahe(windowed)
            normalized, _ = self.preprocessor.normalize_image(enhanced)
            for method_name, interpolation in resize_methods:
                resized, _ = self.preprocessor.resize_image(normalized, (512, 512), interpolation)
                # Simple edge preservation score using Canny
                edges = cv2.Canny((resized * 255).astype(np.uint8), 50, 150).sum()
                results.append({
                    'image': os.path.basename(dcm_path),
                    'method': method_name,
                    'min': resized.min(),
                    'max': resized.max(),
                    'mean': resized.mean(),
                    'std': resized.std(),
                    'edge_preservation': edges
                })
                print(f"✓ {method_name}: mean={resized.mean():.4f}, std={resized.std():.4f}, edges={edges}")
        return pd.DataFrame(results)

    def analyze_normalization_methods(self) -> pd.DataFrame:
        print("\n" + "="*60)
        print("NORMALIZATION ANALYSIS")
        print("="*60)
        norm_configs = [
            ("Zero to One [0, 1]", "zero_to_one"),
            ("Minus One to One [-1, 1]", "minus_one_to_one"),
        ]
        results = []

        test_files = self.sample_files[:1] if self.keep_only_one else self.sample_files[:3]
        for dcm_path in test_files:
            img, _ = self.preprocessor.load_dicom(dcm_path)
            windowed, _ = self.preprocessor.apply_windowing(img)
            enhanced, _ = self.preprocessor.apply_clahe(windowed)
            for config_name, norm_range in norm_configs:
                normalized, _ = self.preprocessor.normalize_image(enhanced, norm_range)
                results.append({
                    'image': os.path.basename(dcm_path),
                    'config': config_name,
                    'norm_range': norm_range,
                    'min': normalized.min(),
                    'max': normalized.max(),
                    'mean': normalized.mean(),
                    'std': normalized.std()
                })
                print(f"✓ {config_name}: range=[{normalized.min():.4f}, {normalized.max():.4f}], mean={normalized.mean():.4f}")
        return pd.DataFrame(results)

    def generate_report(self, output_dir: str = '.'):
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        # Run all analyses
        windowing_results = self.analyze_windowing_methods()
        clahe_results = self.analyze_clahe_impact()
        resize_results = self.analyze_resize_methods()
        norm_results = self.analyze_normalization_methods()

        # Save to CSVs
        windowing_results.to_csv(f'{output_dir}/analysis_windowing.csv', index=False)
        clahe_results.to_csv(f'{output_dir}/analysis_clahe.csv', index=False)
        resize_results.to_csv(f'{output_dir}/analysis_resize.csv', index=False)
        norm_results.to_csv(f'{output_dir}/analysis_normalization.csv', index=False)

        print("\n✓ Results saved:")
        print(f"  - {output_dir}/analysis_windowing.csv")
        print(f"  - {output_dir}/analysis_clahe.csv")
        print(f"  - {output_dir}/analysis_resize.csv")
        print(f"  - {output_dir}/analysis_normalization.csv")

        # Simple summary logic (best configs by std, edge score)
        summary = {
            'Analysis': [
                'Windowing',
                'CLAHE',
                'Resize Interpolation',
                'Normalization'
            ],
            'Best Parameter': [
                windowing_results.loc[windowing_results['std'].idxmax(), 'config'],
                clahe_results.loc[clahe_results['std'].idxmax(), 'config'],
                resize_results.loc[resize_results['edge_preservation'].idxmax(), 'method'],
                norm_results.loc[norm_results['std'].idxmax(), 'config']
            ]
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f'{output_dir}/analysis_summary.csv', index=False)

        print(f"\n  - {output_dir}/analysis_summary.csv")
        print("\n" + summary_df.to_string(index=False))
        return {
            'windowing': windowing_results,
            'clahe': clahe_results,
            'resize': resize_results,
            'normalization': norm_results,
            'summary': summary_df
        }


if __name__ == "__main__":
    # Configuration
    DICOM_DIR = "../../../Data/siim-original/dicom-images-train"
    NUM_SAMPLES = 1000   # Try a few for analytics, or set to 1 for rapid checks
    OUTPUT_DIR = "."
    KEEP_ONLY_ONE = False   # Set to True for quick checks

    analyzer = PreprocessingAnalyzer(DICOM_DIR, NUM_SAMPLES, keep_only_one=KEEP_ONLY_ONE)
    results = analyzer.generate_report(OUTPUT_DIR)
    print("\n✓ Analysis complete!")
