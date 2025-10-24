"""
Data Exploration Script for SIIM-ACR Pneumothorax Dataset
FIXED: Uses official SIIM-ACR RLE decoder
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from tqdm import tqdm

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# =====================================================
# SECTION 1: OFFICIAL RLE Decoding Function
# =====================================================

def rle2mask(rle, width, height):
    """
    Official SIIM-ACR RLE decoder
    Uses relative positioning (column-major order)
    """
    mask = np.zeros(width * height)
    
    # Handle no-pneumothorax case
    if rle == '-1' or pd.isna(rle):
        return mask.reshape(width, height)
    
    # Parse RLE: alternating offset and length
    array = np.asarray([int(x) for x in str(rle).split()])
    starts = array[0::2]      # Offset positions
    lengths = array[1::2]     # Run lengths
    
    # Reconstruct mask with relative positioning
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start  # KEY: Relative positioning!
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
    
    return mask.reshape(width, height)


# =====================================================
# SECTION 2: Load and Parse Dataset
# =====================================================

def load_dataset_info(csv_path):
    """Load CSV and compute basic statistics"""
    print("Loading dataset CSV...")
    df = pd.read_csv(csv_path)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Strip EncodedPixels values
    if 'EncodedPixels' in df.columns:
        df['EncodedPixels'] = df['EncodedPixels'].astype(str).str.strip()
    else:
        raise ValueError(f"EncodedPixels column not found! Available columns: {df.columns.tolist()}")
    
    print(f"Total images: {len(df)}")
    
    # Separate positive and negative cases
    df['has_pneumothorax'] = df['EncodedPixels'] != '-1'
    positive_cases = df[df['has_pneumothorax']]
    negative_cases = df[~df['has_pneumothorax']]
    
    print(f"Images with pneumothorax: {len(positive_cases)} ({len(positive_cases)/len(df)*100:.2f}%)")
    print(f"Images without pneumothorax: {len(negative_cases)} ({len(negative_cases)/len(df)*100:.2f}%)")
    
    return df, positive_cases, negative_cases


# =====================================================
# SECTION 3: Analyze Pneumothorax Sizes
# =====================================================

def analyze_pneumothorax_sizes(positive_df, sample_size=None):
    """
    Analyze size distribution of pneumothoraces
    NOW WITH CORRECT RLE DECODER
    """
    print("\nAnalyzing pneumothorax sizes...")
    
    if sample_size:
        positive_df = positive_df.sample(min(sample_size, len(positive_df)))
    
    size_stats = []
    
    for idx, row in tqdm(positive_df.iterrows(), total=len(positive_df)):
        rle = row['EncodedPixels']
        
        # Decode mask with CORRECT decoder
        mask = rle2mask(rle, 1024, 1024)
        
        # Calculate statistics
        ptx_pixels = (mask == 255).sum()  # Count white pixels (255)
        total_pixels = mask.size
        percentage = (ptx_pixels / total_pixels) * 100
        
        # Calculate bounding box
        mask_binary = mask == 255
        rows = np.any(mask_binary, axis=1)
        cols = np.any(mask_binary, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_height = rmax - rmin
            bbox_width = cmax - cmin
            bbox_area = bbox_height * bbox_width
        else:
            bbox_height, bbox_width, bbox_area = 0, 0, 0
        
        # Count number of runs
        rle_cleaned = str(rle).strip()
        num_runs = len(rle_cleaned.split()) // 2
        avg_run_length = ptx_pixels / num_runs if num_runs > 0 else 0
        
        # Classify size
        if percentage < 0.5:
            size_category = 'Very Small'
        elif percentage < 1.5:
            size_category = 'Small'
        elif percentage < 3.0:
            size_category = 'Medium'
        elif percentage < 6.0:
            size_category = 'Large'
        else:
            size_category = 'Very Large'
        
        size_stats.append({
            'ImageId': row['ImageId'],
            'ptx_pixels': ptx_pixels,
            'percentage': percentage,
            'num_runs': num_runs,
            'avg_run_length': avg_run_length,
            'bbox_height': bbox_height,
            'bbox_width': bbox_width,
            'bbox_area': bbox_area,
            'size_category': size_category
        })
    
    return pd.DataFrame(size_stats)


# =====================================================
# SECTION 4: Visualization Functions
# =====================================================

def visualize_size_distribution(size_df):
    """Create comprehensive visualization of size distribution"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Pneumothorax Size Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram of percentage
    axes[0, 0].hist(size_df['percentage'], bins=50, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Pneumothorax Size (% of image)')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].set_title('Size Distribution (Percentage)')
    axes[0, 0].axvline(size_df['percentage'].median(), color='red', linestyle='--', 
                       label=f'Median: {size_df["percentage"].median():.2f}%')
    axes[0, 0].legend()
    
    # 2. Category distribution
    category_counts = size_df['size_category'].value_counts()
    category_order = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    category_counts = category_counts.reindex(category_order, fill_value=0)
    
    axes[0, 1].bar(range(len(category_counts)), category_counts.values, 
                   color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
    axes[0, 1].set_xticks(range(len(category_counts)))
    axes[0, 1].set_xticklabels(category_counts.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Pneumothorax Size Categories')
    for i, v in enumerate(category_counts.values):
        axes[0, 1].text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # 3. Cumulative distribution
    sorted_percentages = np.sort(size_df['percentage'])
    cumulative = np.arange(1, len(sorted_percentages) + 1) / len(sorted_percentages) * 100
    axes[0, 2].plot(sorted_percentages, cumulative, linewidth=2, color='darkgreen')
    axes[0, 2].set_xlabel('Pneumothorax Size (% of image)')
    axes[0, 2].set_ylabel('Cumulative Percentage of Cases')
    axes[0, 2].set_title('Cumulative Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(50, color='red', linestyle='--', alpha=0.5)
    
    # 4. Number of runs (shape complexity)
    axes[1, 0].hist(size_df['num_runs'], bins=50, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Runs (Shape Complexity)')
    axes[1, 0].set_ylabel('Number of Images')
    axes[1, 0].set_title('Shape Complexity Distribution')
    axes[1, 0].axvline(size_df['num_runs'].median(), color='red', linestyle='--',
                       label=f'Median: {size_df["num_runs"].median():.0f}')
    axes[1, 0].legend()
    
    # 5. Bounding box aspect ratio
    aspect_ratios = size_df['bbox_width'] / (size_df['bbox_height'] + 1e-6)
    axes[1, 1].hist(aspect_ratios, bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Bounding Box Aspect Ratio (width/height)')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].set_title('Pneumothorax Shape Orientation')
    axes[1, 1].axvline(1.0, color='red', linestyle='--', label='Square shape')
    axes[1, 1].legend()
    
    # 6. Size vs Complexity scatter
    axes[1, 2].scatter(size_df['percentage'], size_df['num_runs'], 
                      alpha=0.5, s=20, color='teal')
    axes[1, 2].set_xlabel('Pneumothorax Size (% of image)')
    axes[1, 2].set_ylabel('Number of Runs (Complexity)')
    axes[1, 2].set_title('Size vs Shape Complexity')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pneumothorax_size_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'pneumothorax_size_analysis.png'")
    plt.show()


# =====================================================
# SECTION 5: Statistical Summary
# =====================================================

def print_statistical_summary(size_df):
    """Print comprehensive statistical summary"""
    
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY OF PNEUMOTHORAX DATASET")
    print("="*60)
    
    print("\n--- SIZE STATISTICS ---")
    print(f"Mean size: {size_df['percentage'].mean():.3f}% of image")
    print(f"Median size: {size_df['percentage'].median():.3f}% of image")
    print(f"Std deviation: {size_df['percentage'].std():.3f}%")
    print(f"Min size: {size_df['percentage'].min():.3f}%")
    print(f"Max size: {size_df['percentage'].max():.3f}%")
    
    print("\n--- SIZE CATEGORIES ---")
    category_counts = size_df['size_category'].value_counts()
    for category in ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']:
        count = category_counts.get(category, 0)
        percentage = (count / len(size_df)) * 100
        print(f"{category:12s}: {count:4d} images ({percentage:5.2f}%)")
    
    print("\n--- SHAPE COMPLEXITY ---")
    print(f"Mean number of runs: {size_df['num_runs'].mean():.1f}")
    print(f"Median number of runs: {size_df['num_runs'].median():.1f}")
    print(f"Max number of runs: {size_df['num_runs'].max():.0f}")
    
    print("\n--- BOUNDING BOX STATISTICS ---")
    print(f"Mean height: {size_df['bbox_height'].mean():.1f} pixels")
    print(f"Mean width: {size_df['bbox_width'].mean():.1f} pixels")
    
    print("\n--- KEY INSIGHTS FOR YOUR PROJECT ---")
    small_fraction = len(size_df[size_df['percentage'] < 1.5]) / len(size_df) * 100
    print(f"• {small_fraction:.1f}% of pneumothoraces are SMALL (<1.5% of image)")
    print(f"  → These are clinically critical but easy to miss!")
    print(f"  → Your augmentation should oversample these cases")
    
    complex_shapes = len(size_df[size_df['num_runs'] > 100]) / len(size_df) * 100
    print(f"\n• {complex_shapes:.1f}% have COMPLEX shapes (>100 runs)")
    print(f"  → Irregular boundaries are common")
    print(f"  → Model needs to handle non-circular shapes")
    
    print("\n" + "="*60)


# =====================================================
# SECTION 6: Main Execution
# =====================================================

def main():
    """Main execution function"""
    
    # Configuration
    CSV_PATH = '../../Data/train-rle.csv'
    SAMPLE_SIZE = None  # Analyze ALL positive cases
    
    print("="*60)
    print("SIIM-ACR PNEUMOTHORAX DATASET EXPLORATION")
    print("="*60)
    
    # Load dataset
    df, positive_df, negative_df = load_dataset_info(CSV_PATH)
    
    # Analyze sizes with CORRECT decoder
    size_df = analyze_pneumothorax_sizes(positive_df, sample_size=SAMPLE_SIZE)
    
    # Print statistics
    print_statistical_summary(size_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_size_distribution(size_df)
    
    # Save results
    size_df.to_csv('pneumothorax_size_analysis.csv', index=False)
    print("\nDetailed results saved to 'pneumothorax_size_analysis.csv'")
    
    print("\n✓ Data exploration complete!")


if __name__ == "__main__":
    main()
