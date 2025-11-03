import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import os

# Import your dataloaders
sys.path.append(os.path.dirname(__file__))  # ensure path
from data_loader import (
    create_basic_loader,
    create_standard_loader,
    create_advanced_loader
)



# ======================================================================
# CONFIGURATION
# ======================================================================
# ...existing code...
from pathlib import Path

# ======================================================================
# CONFIGURATION (resolved relative to this script to avoid PowerShell path issues)
# ======================================================================
CURRENT_FILE = Path(__file__).resolve()
# data_validation.py is at .../Scripts/model/data_validation.py -> project root is two parents up
PROJECT_ROOT = CURRENT_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

# Use absolute paths (string) when passing to dataloader factories
SPLIT_CSV = str(DATA_DIR / "splits" / "train_split.csv")
DICOM_DIR = str(DATA_DIR / "siim-original" / "dicom-images-train")

BATCH_SIZE = 2
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ensure project root is on sys.path so imports resolve
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ...existing code...

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def show_image_mask(image_tensor, mask_tensor, title=""):
    img = image_tensor.squeeze().cpu().numpy()
    mask = mask_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img, cmap='gray', vmin=-1, vmax=1)
    axs[0].set_title(f"{title} - Image")
    axs[0].axis('off')

    # Force mask contrast — anything >0 becomes white
    axs[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title(f"{title} - Mask (contrast fixed)")
    axs[1].axis('off')

    plt.show()


def compute_mean_std(loader, n_batches=20):
    """Estimate mean and std of dataset from a few batches."""
    mean = 0.0
    std = 0.0
    nb_samples = 0

    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(1).sum()
        std += images.std(1).sum()
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean.item(), std.item()

def sanity_check_tensors(images, masks, name=""):
    """Check shapes, dtype, and range."""
    print(f"--- Sanity Check: {name} ---")
    print(f"Images shape: {images.shape}, dtype: {images.dtype}, range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Masks shape:  {masks.shape}, dtype: {masks.dtype}, unique values: {torch.unique(masks)}")
    if torch.isnan(images).any():
        print("⚠️ NaNs detected in images!")
    if torch.isnan(masks).any():
        print("⚠️ NaNs detected in masks!")
    print("--------------------------------------------\n")

# ======================================================================
# LOADERS
# ======================================================================
print("\n=== Visual Validation for Curriculum DataLoaders ===\n")

basic_loader = create_basic_loader(SPLIT_CSV, DICOM_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
standard_loader = create_standard_loader(SPLIT_CSV, DICOM_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
advanced_loader = create_advanced_loader(SPLIT_CSV, DICOM_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# ======================================================================
# 1️⃣ VISUALIZE: BASIC → STANDARD → ADVANCED
# ======================================================================

# ======================================================================
# 1️⃣ VISUALIZE: BASIC → STANDARD → ADVANCED (FIXED VERSION)
# ======================================================================

print("\n--- VISUAL CHECK (FIXED) ---")

# Create iterators for each loader
basic_iter = iter(basic_loader)
standard_iter = iter(standard_loader) 
advanced_iter = iter(advanced_loader)

for i in range(3):  # Check first 3 samples
    print(f"\n🔍 Processing sample {i}")
    
    try:
        # Get batches from each loader
        b_img, b_mask = next(basic_iter)
        s_img, s_mask = next(standard_iter)
        a_img, a_mask = next(advanced_iter)
        
        # Print debug info for the actual samples being processed
        print(f"🔍 [REAL DEBUG] Basic loader - Image range: [{b_img.min():.3f}, {b_img.max():.3f}], Mask sum: {b_mask.sum().item():.1f}")
        print(f"🔍 [REAL DEBUG] Standard loader - Image range: [{s_img.min():.3f}, {s_img.max():.3f}], Mask sum: {s_mask.sum().item():.1f}")
        print(f"🔍 [REAL DEBUG] Advanced loader - Image range: [{a_img.min():.3f}, {a_img.max():.3f}], Mask sum: {a_mask.sum().item():.1f}")
        
        # Check if we have any non-zero masks
        if b_mask.sum().item() == 0 and s_mask.sum().item() == 0 and a_mask.sum().item() == 0:
            print("❌ ALL MASKS ARE ZERO - This might be a negative case or sampling issue")
        else:
            print("✅ Found non-zero masks!")
        
        # Process first sample in each batch
        sanity_check_tensors(b_img, b_mask, "Level 1 - Basic")
        show_image_mask(b_img[0], b_mask[0], title="Level 1: Basic (Original)")
        
        sanity_check_tensors(s_img, s_mask, "Level 2 - Standard") 
        show_image_mask(s_img[0], s_mask[0], title="Level 2: Standard (Augmented)")
        
        sanity_check_tensors(a_img, a_mask, "Level 3 - Advanced")
        show_image_mask(a_img[0], a_mask[0], title="Level 3: Advanced (Artifacts)")
        
    except StopIteration:
        print("⚠️ Reached end of one dataset")
        break

# ======================================================================
# 2️⃣ COMPUTE MEAN & STD (Check Normalization)
# ======================================================================
print("\n--- COMPUTE DATASET MEAN & STD ---")
basic_mean, basic_std = compute_mean_std(basic_loader)
std_mean, std_std = compute_mean_std(standard_loader)
adv_mean, adv_std = compute_mean_std(advanced_loader)

print(f"Level 1 - Basic:    mean={basic_mean:.4f}, std={basic_std:.4f}")
print(f"Level 2 - Standard: mean={std_mean:.4f}, std={std_std:.4f}")
print(f"Level 3 - Advanced: mean={adv_mean:.4f}, std={adv_std:.4f}")

# ======================================================================
# 3️⃣ CHECK ALIGNMENT: IMAGE vs MASK
# ======================================================================
print("\n--- ALIGNMENT CHECK ---")

def overlay_mask(image_tensor, mask_tensor, alpha=0.4):
    """Overlay mask on image to verify alignment visually."""
    img = image_tensor.squeeze().cpu().numpy()
    mask = mask_tensor.squeeze().cpu().numpy()
    overlay = np.stack([img, img, img], axis=-1)
    overlay[..., 1] += mask * alpha  # green overlay for mask
    overlay = np.clip(overlay, 0, 1)
    plt.figure(figsize=(4,4))
    plt.imshow(overlay)
    plt.title("Overlay Check (Green = Mask)")
    plt.axis("off")
    plt.show()

for i, (images, masks) in enumerate(basic_loader):
    overlay_mask(images[0], masks[0])
    if i == 2:
        break

