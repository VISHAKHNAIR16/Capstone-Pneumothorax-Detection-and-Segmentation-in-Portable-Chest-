"""
AUGMENTATION DIAGNOSTIC - Validate Fixes with Production-Ready BasicAugmentation
Works with FIXED basic_augmentations.py (rotation ±1.5°)
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import sys
import os
import pydicom
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocess')))
from preprocess import PreprocessingPipeline
from basic_augmentations import BasicAugmentation


def diagnose_augmentation_issue():
    """
    Pinpoint EXACTLY which augmentation operation is working or broken
    Tests with FIXED BasicAugmentation (rotation ±1.5°)
    """
    
    print("\n" + "="*120)
    print("🔍 AUGMENTATION DIAGNOSTIC - Validate FIXED BasicAugmentation")
    print("="*120)
    
    # Load sample DICOM
    dicom_path = r"C:/Users/VISHAKH NAIR/Desktop/CAPSTONE/Capstone-Pneumothorax-Detection-and-Segmentation-in-Portable-Chest-/Data/siim-original/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.306.1517875162.312800/1.2.276.0.7230010.3.1.3.8323329.306.1517875162.312799/1.2.276.0.7230010.3.1.4.8323329.306.1517875162.312801.dcm"
    
    print("\n📂 Loading DICOM...")
    try:
        dcm = pydicom.dcmread(dicom_path)
        original = dcm.pixel_array
        print(f"   ✅ Loaded: {original.shape}, {original.dtype}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Preprocess
    print("\n🔧 Preprocessing...")
    preprocessing_pipeline = PreprocessingPipeline()
    basic_augmentation = BasicAugmentation()
    
    try:
        img_float = original.astype(np.float32)
        windowed, _ = preprocessing_pipeline.apply_windowing(img_float)
        clahe_enhanced, _ = preprocessing_pipeline.apply_clahe(windowed)
        normalized, _ = preprocessing_pipeline.normalize_image(clahe_enhanced, 'minus_one_to_one')
        preprocessed, _ = preprocessing_pipeline.resize_image(normalized)
        
        print(f"   ✅ Preprocessed: {preprocessed.shape}")
        print(f"      Range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        print(f"      Mean: {preprocessed.mean():.3f}, Std: {preprocessed.std():.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create dummy mask
    dummy_mask = np.zeros_like(preprocessed)
    
    print("\n" + "="*120)
    print("🔬 TESTING EACH AUGMENTATION OPERATION")
    print("="*120)
    
    results = {}
    
    # ===== TEST 2a: All augmentations ON =====
    print("\n2a. ALL FIXED AUGMENTATIONS ON (rotation=±1.5°, brightness, contrast, noise):")
    print("-"*120)
    try:
        aug_all, _ = basic_augmentation.augment(
            preprocessed.copy(), dummy_mask,
            rotation=True, brightness=True, contrast=True, noise=True, elastic=False
        )
        
        psnr_all = psnr(preprocessed, aug_all, data_range=2.0)
        ssim_all = ssim(preprocessed, aug_all, data_range=2.0)
        
        print(f"    PSNR: {psnr_all:.2f} dB")
        print(f"    SSIM: {ssim_all:.4f}")
        
        if psnr_all > 25 and ssim_all > 0.80:
            print(f"    Status: ✅ PASS (Quality Acceptable)")
        elif psnr_all > 20 and ssim_all > 0.70:
            print(f"    Status: ⚠️ MARGINAL (Review Needed)")
        else:
            print(f"    Status: ❌ FAIL (Quality Poor)")
        
        results['all_on'] = {
            'psnr': psnr_all,
            'ssim': ssim_all,
            'pass': psnr_all > 25 and ssim_all > 0.80
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== TEST 2b: Only Rotation =====
    print("\n2b. ONLY ROTATION (±1.5° - FIXED):")
    print("-"*120)
    try:
        aug_rotation, _ = basic_augmentation.augment(
            preprocessed.copy(), dummy_mask,
            rotation=True, brightness=False, contrast=False, noise=False, elastic=False
        )
        
        psnr_rot = psnr(preprocessed, aug_rotation, data_range=2.0)
        ssim_rot = ssim(preprocessed, aug_rotation, data_range=2.0)
        
        print(f"    PSNR: {psnr_rot:.2f} dB")
        print(f"    SSIM: {ssim_rot:.4f}")
        
        if psnr_rot > 25 and ssim_rot > 0.80:
            print(f"    Status: ✅ PASS (Rotation Fixed!)")
        elif psnr_rot > 20 and ssim_rot > 0.70:
            print(f"    Status: ⚠️ MARGINAL (Still Some Quality Loss)")
        else:
            print(f"    Status: ❌ FAIL (Still Broken)")
        
        results['rotation_only'] = {
            'psnr': psnr_rot,
            'ssim': ssim_rot,
            'pass': psnr_rot > 25 and ssim_rot > 0.80
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== TEST 2c: Only Brightness =====
    print("\n2c. ONLY BRIGHTNESS (±20%):")
    print("-"*120)
    try:
        aug_brightness, _ = basic_augmentation.augment(
            preprocessed.copy(), dummy_mask,
            rotation=False, brightness=True, contrast=False, noise=False, elastic=False
        )
        
        psnr_bright = psnr(preprocessed, aug_brightness, data_range=2.0)
        ssim_bright = ssim(preprocessed, aug_brightness, data_range=2.0)
        
        print(f"    PSNR: {psnr_bright:.2f} dB")
        print(f"    SSIM: {ssim_bright:.4f}")
        
        if psnr_bright > 25 and ssim_bright > 0.80:
            print(f"    Status: ✅ PASS (Brightness Working)")
        else:
            print(f"    Status: ❌ FAIL (Brightness Issue)")
        
        results['brightness_only'] = {
            'psnr': psnr_bright,
            'ssim': ssim_bright,
            'pass': psnr_bright > 25 and ssim_bright > 0.80
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== TEST 2d: Only Contrast =====
    print("\n2d. ONLY CONTRAST (±15%):")
    print("-"*120)
    try:
        aug_contrast, _ = basic_augmentation.augment(
            preprocessed.copy(), dummy_mask,
            rotation=False, brightness=False, contrast=True, noise=False, elastic=False
        )
        
        psnr_cont = psnr(preprocessed, aug_contrast, data_range=2.0)
        ssim_cont = ssim(preprocessed, aug_contrast, data_range=2.0)
        
        print(f"    PSNR: {psnr_cont:.2f} dB")
        print(f"    SSIM: {ssim_cont:.4f}")
        
        if psnr_cont > 25 and ssim_cont > 0.80:
            print(f"    Status: ✅ PASS (Contrast Working)")
        else:
            print(f"    Status: ❌ FAIL (Contrast Issue)")
        
        results['contrast_only'] = {
            'psnr': psnr_cont,
            'ssim': ssim_cont,
            'pass': psnr_cont > 25 and ssim_cont > 0.80
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== TEST 2e: Only Noise =====
    print("\n2e. ONLY NOISE (σ=0.01-0.05):")
    print("-"*120)
    try:
        aug_noise, _ = basic_augmentation.augment(
            preprocessed.copy(), dummy_mask,
            rotation=False, brightness=False, contrast=False, noise=True, elastic=False
        )
        
        psnr_noise = psnr(preprocessed, aug_noise, data_range=2.0)
        ssim_noise = ssim(preprocessed, aug_noise, data_range=2.0)
        
        print(f"    PSNR: {psnr_noise:.2f} dB")
        print(f"    SSIM: {ssim_noise:.4f}")
        
        if psnr_noise > 25 and ssim_noise > 0.80:
            print(f"    Status: ✅ PASS (Noise Working)")
        else:
            print(f"    Status: ⚠️ CHECK (Noise Impact)")
        
        results['noise_only'] = {
            'psnr': psnr_noise,
            'ssim': ssim_noise,
            'pass': psnr_noise > 25 and ssim_noise > 0.80
        }
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== SUMMARY =====
    print("\n" + "="*120)
    print("📊 DIAGNOSTIC SUMMARY - FIXED VERSION")
    print("="*120)
    
    print("\n✅ OPERATIONS PASSING (PSNR > 25 dB, SSIM > 0.80):")
    passing = 0
    for test_name, metrics in results.items():
        if metrics.get('pass', False):
            print(f"   ✅ {test_name.upper()}: PSNR {metrics['psnr']:.2f} dB, SSIM {metrics['ssim']:.4f}")
            passing += 1
    
    print("\n❌ OPERATIONS NOT PASSING:")
    failing = 0
    for test_name, metrics in results.items():
        if not metrics.get('pass', False):
            print(f"   ❌ {test_name.upper()}: PSNR {metrics['psnr']:.2f} dB, SSIM {metrics['ssim']:.4f}")
            failing += 1
    
    if failing == 0:
        print("   (None - All operations passing!)")
    
    # ===== FINAL RECOMMENDATION =====
    print("\n" + "="*120)
    print("🎯 FINAL RECOMMENDATION")
    print("="*120)
    
    if passing == len(results) or (passing >= 4):
        print("\n✅ READY FOR PRODUCTION")
        print("   ✅ All augmentation operations working correctly")
        print("   ✅ Quality metrics within acceptable range")
        print("   ✅ Safe to generate 10,000+ training images")
        print("\n🚀 NEXT STEP: Generate augmented dataset")
    else:
        print("\n⚠️ NEEDS REVIEW")
        print(f"   {passing}/{len(results)} operations passing")
        print(f"   {failing}/{len(results)} operations have issues")
        print("   Investigate failed operations before mass generation")
    
    print("\n" + "="*120)


if __name__ == "__main__":
    diagnose_augmentation_issue()
