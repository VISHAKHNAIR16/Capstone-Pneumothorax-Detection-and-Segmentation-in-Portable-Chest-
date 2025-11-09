import sys
sys.path.append('.')
from data_loader import create_production_loader

def test_fixed_loader():
    print("🧪 Testing fixed data loader...")
    
    loader = create_production_loader(
        split_csv="../../Data/splits/train_split.csv",
        dicom_dir="../../Data/siim-original/dicom-images-train",
        level=1,
        batch_size=4,
        preload_ram=True
    )
    
    # Check first few batches
    for batch_idx, (images, masks) in enumerate(loader):
        print(f"\n=== BATCH {batch_idx} ===")
        print(f"Images: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Masks: {masks.shape}, unique: {torch.unique(masks)}")
        print(f"Mask sum: {masks.sum().item()}")
        
        if batch_idx >= 2:
            break

if __name__ == "__main__":
    test_fixed_loader()