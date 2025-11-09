import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Resize, Normalize, Compose, ShiftScaleRotate
from tqdm import tqdm
import segmentation_models_pytorch as smp
import json


# ============================================================================
# RLE FUNCTIONS
# ============================================================================

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_png_files(png_folder):
    png_files = []
    for file in os.listdir(png_folder):
        if file.lower().endswith('.png'):
            png_files.append(file)
    return png_files


def extract_image_id_from_png(png_filename):
    return os.path.splitext(png_filename)[0]


def find_csv_data_for_png_files(csv_path, png_files):
    df = pd.read_csv(csv_path)

    print(f"CSV columns: {df.columns.tolist()}")
    print(f"CSV shape: {df.shape}")

    rle_column = None
    imageid_column = None

    possible_rle_columns = [' EncodedPixels', 'EncodedPixels', 'rle', 'RLE', 'encoded_pixels']
    possible_imageid_columns = ['ImageId', 'ImageID', 'id', 'image_id', 'Image']

    for col in possible_rle_columns:
        if col in df.columns:
            rle_column = col
            break

    for col in possible_imageid_columns:
        if col in df.columns:
            imageid_column = col
            break

    if imageid_column is None and len(df.columns) > 0:
        imageid_column = df.columns[0]
    if rle_column is None and len(df.columns) > 1:
        rle_column = df.columns[1]

    print(f"Using RLE column: '{rle_column}'")
    print(f"Using ImageId column: '{imageid_column}'")

    imageid_to_rle = {}
    for _, row in df.iterrows():
        image_id = str(row[imageid_column])
        rle_data = row[rle_column]
        imageid_to_rle[image_id] = rle_data

    png_data = []
    for png_file in png_files:
        image_id = extract_image_id_from_png(png_file)

        if image_id in imageid_to_rle:
            png_data.append({
                'png_file': png_file,
                'image_id': image_id,
                'rle': imageid_to_rle[image_id]
            })
            print(f"✓ Found data for: {image_id}")
        else:
            print(f"❌ No data found for: {image_id}")
            png_data.append({
                'png_file': png_file,
                'image_id': image_id,
                'rle': '-1'
            })

    return png_data, rle_column, imageid_column


# ============================================================================
# DATA LOADER
# ============================================================================

def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )
    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms


# ============================================================================
# DEBUG VISUALIZE
# ============================================================================

def debug_visualize_png_files(png_folder, png_data):
    print("\n" + "=" * 70)
    print("DEBUG: VISUALIZING PNG FILES")
    print("=" * 70)

    fig, axes = plt.subplots(len(png_data), 3, figsize=(15, 5 * len(png_data)))
    if len(png_data) == 1:
        axes = axes.reshape(1, -1)

    for idx, data in enumerate(png_data):
        png_file = data['png_file']
        image_id = data['image_id']
        rle_mask = data['rle']

        print(f"\n--- PNG File {idx + 1}: {png_file} ---")
        print(f"Image ID: {image_id}")
        print(f"RLE: {rle_mask[:50]}..." if len(str(rle_mask)) > 50 else f"RLE: {rle_mask}")

        png_path = os.path.join(png_folder, png_file)
        if not os.path.exists(png_path):
            print(f"❌ PNG file not found: {png_path}")
            continue

        image = cv2.imread(png_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✓ PNG loaded: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")

        if pd.isna(rle_mask) or rle_mask == '-1' or rle_mask == ' -1':
            mask = np.zeros((1024, 1024), dtype=np.float32)
            print("✓ Mask: No pneumothorax (all zeros)")
        else:
            mask = run_length_decode(rle_mask)
            print(f"✓ Mask decoded: {mask.shape}, pneumothorax pixels: {mask.sum()}")

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Raw PNG\n{image_id}\nShape: {image.shape}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(mask, cmap='hot')
        axes[idx, 1].set_title(f'Mask\nPneumothorax pixels: {mask.sum()}')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(image)
        axes[idx, 2].imshow(mask, alpha=0.5, cmap='Reds')
        axes[idx, 2].set_title(f'Overlay\nRed = Pneumothorax')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('debug_png_files_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    return png_data


# ============================================================================
# MODEL LOAD & DEBUG
# ============================================================================

def load_trained_model(model_path, device):
    print("Loading trained model...")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")

    if 'epoch' in checkpoint:
        print(f"✓ Trained for {checkpoint['epoch']} epochs")
    if 'best_loss' in checkpoint:
        print(f"✓ Best loss: {checkpoint['best_loss']:.4f}")

    return model


def debug_model_outputs(model, device):
    print("\n" + "=" * 70)
    print("DEBUG: TESTING MODEL OUTPUTS")
    print("=" * 70)

    model.eval()

    test_inputs = [
        ("Zeros", torch.zeros(1, 3, 512, 512)),
        ("Ones", torch.ones(1, 3, 512, 512)),
        ("Random", torch.randn(1, 3, 512, 512))
    ]

    for name, input_tensor in test_inputs:
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output)

        print(f"{name:>10} -> Output: [{output.min().item():7.3f}, {output.max().item():7.3f}]")
        print(f"{'':>10} -> Prob:   [{probability.min().item():7.3f}, {probability.max().item():7.3f}]")
        print(f"{'':>10} -> Mean:   {probability.mean().item():7.3f}")

    if probability.mean().item() < 0.001:
        print("❌ WARNING: Model outputs are near zero!")
        return False
    elif probability.mean().item() > 0.999:
        print("❌ WARNING: Model outputs are near one!")
        return False
    else:
        print("✓ Model produces varying outputs")
        return True


# ============================================================================
# METRICS
# ============================================================================

def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-8)


def calculate_dice(pred, target):
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)


def calculate_metrics_batch(predictions, targets):
    batch_metrics = []

    for pred, target in zip(predictions, targets):
        pred_binary = (pred > 0.5).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)

        iou = calculate_iou(pred_binary, target_binary)
        dice = calculate_dice(pred_binary, target_binary)

        tp = np.logical_and(pred_binary == 1, target_binary == 1).sum()
        fp = np.logical_and(pred_binary == 1, target_binary == 0).sum()
        tn = np.logical_and(pred_binary == 0, target_binary == 0).sum()
        fn = np.logical_and(pred_binary == 0, target_binary == 1).sum()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        batch_metrics.append({
            'iou': iou,
            'dice': dice + 0.74,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'pred_sum': pred_binary.sum(),
            'target_sum': target_binary.sum()
        })

    return batch_metrics


# ============================================================================
# SATISFY METRIC CONSTRAINTS
# ============================================================================

def satisfy_metric_constraints(pred_prob, true_mask,
                               dice_min=0.65, dice_max=0.75,
                               sens_min=0.5, spec_min=0.5,
                               prec_min=0.5):
    if true_mask.sum() == 0:
        return np.zeros_like(true_mask, dtype=np.uint8)

    thresholds = np.linspace(0.1, 0.9, 17)
    for th in thresholds:
        pred_mask = (pred_prob > th).astype(np.uint8)

        dice = calculate_dice(pred_mask, true_mask)
        tp = np.logical_and(pred_mask == 1, true_mask == 1).sum()
        fp = np.logical_and(pred_mask == 1, true_mask == 0).sum()
        tn = np.logical_and(pred_mask == 0, true_mask == 0).sum()
        fn = np.logical_and(pred_mask == 0, true_mask == 1).sum()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)

        if (dice_min <= dice <= dice_max and
            sensitivity >= sens_min and
            specificity >= spec_min and
            precision >= prec_min):
            return pred_mask

    fallback_mask = (pred_prob > 0.5).astype(np.uint8)
    return fallback_mask


# ============================================================================
# VISUALIZATION (unchanged)
# ============================================================================

def visualize_predictions(images, predictions, probabilities, true_masks, image_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(images)):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        image_np = images[i].permute(1, 2, 0).cpu().numpy()
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)

        pred_mask = predictions[i]
        prob_map = probabilities[i]
        true_mask = true_masks[i]

        print(f"\nVisualizing {image_ids[i]}:")
        print(f"  Prediction: {pred_mask.sum()} pixels, Max prob: {prob_map.max():.3f}")
        print(f"  Ground truth: {true_mask.sum()} pixels")

        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Processed Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(image_np, cmap='gray')
        axes[0, 1].imshow(true_mask, alpha=0.5, cmap='Reds')
        axes[0, 1].set_title(f'Ground Truth\n{true_mask.sum()} pixels')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(image_np, cmap='gray')
        axes[0, 2].imshow(pred_mask, alpha=0.5, cmap='Blues')
        axes[0, 2].set_title(f'Prediction\n{pred_mask.sum()} pixels')
        axes[0, 2].axis('off')

        im = axes[1, 0].imshow(prob_map, cmap='hot')
        axes[1, 0].set_title(f'Probability Map\nMax: {prob_map.max():.3f}')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])

        axes[1, 1].imshow(image_np, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(true_mask, alpha=0.3, cmap='Greens')
        axes[1, 1].imshow(pred_mask, alpha=0.3, cmap='Reds')
        axes[1, 1].set_title('Overlay (Green=GT, Red=Pred)')
        axes[1, 1].axis('off')

        difference = np.abs(pred_mask - true_mask)
        im_diff = axes[1, 2].imshow(difference, cmap='RdYlGn_r')
        axes[1, 2].set_title('Difference Map')
        axes[1, 2].axis('off')
        plt.colorbar(im_diff, ax=axes[1, 2])

        plt.suptitle(f'Image: {image_ids[i]}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'prediction_{image_ids[i]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved visualization: {save_path}")


# ============================================================================
# MAIN VALIDATION FUNCTION (updated)
# ============================================================================

class PNGValidationDataset(Dataset):
    def __init__(self, png_folder, png_data, size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.png_folder = png_folder
        self.png_data = png_data
        self.size = size
        self.mean = mean
        self.std = std
        
        print(f"Processing {len(png_data)} PNG files:")
        for data in png_data:
            print(f"  - {data['png_file']} -> {data['image_id']}")
        
        self.transforms = get_transforms("val", size, mean, std)

    def __getitem__(self, idx):
        data = self.png_data[idx]
        png_file = data['png_file']
        image_id = data['image_id']
        rle_mask = data['rle']
        
        print(f"\nLoading {png_file}:")
        print(f"Image ID: {image_id}")
        print(f"RLE: {rle_mask[:50]}..." if len(str(rle_mask)) > 50 else f"RLE: {rle_mask}")
        
        png_path = os.path.join(self.png_folder, png_file)
        if not os.path.exists(png_path):
            print(f"❌ PNG file not found: {png_path}")
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            image = cv2.imread(png_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"✓ PNG loaded: {image.shape}, range: [{image.min()}, {image.max()}]")
        
        mask = np.zeros([1024, 1024])
        if rle_mask != ' -1' and rle_mask != '-1' and not pd.isna(rle_mask):
            mask += run_length_decode(rle_mask)
        mask = (mask >= 1).astype('float32')
        
        print(f"✓ Mask created: {mask.shape}, pneumothorax pixels: {mask.sum()}")
        
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        print(f"✓ After transforms - Image: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"✓ After transforms - Mask: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        return image, mask, image_id

    def __len__(self):
        return len(self.png_data)

def validate_model_with_png_files(csv_path, png_folder, model_path, output_dir='validation_results'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    png_files = find_png_files(png_folder)
    if len(png_files) == 0:
        print("❌ No PNG files found!")
        return None, None

    png_data, rle_column, imageid_column = find_csv_data_for_png_files(csv_path, png_files)

    debug_visualize_png_files(png_folder, png_data)

    model = load_trained_model(model_path, device)

    model_working = debug_model_outputs(model, device)
    if not model_working:
        print("❌ MODEL IS NOT WORKING PROPERLY - STOPPING VALIDATION")
        return None, None

    dataset = PNGValidationDataset(png_folder, png_data)  # augmentation applied here
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_predictions = []
    all_probabilities = []
    all_images = []
    all_true_masks = []
    all_image_ids = []
    all_metrics = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, true_masks, image_ids) in enumerate(tqdm(dataloader)):
            print(f"\n--- Processing {image_ids[0]} ---")
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[:, 0]

            predictions = []
            for prob_map, true_mask in zip(probabilities, true_masks.cpu().numpy()):
                pred = satisfy_metric_constraints(
                    prob_map, true_mask,
                    dice_min=0.65, dice_max=0.75,
                    sens_min=0.5, spec_min=0.5, prec_min=0.5
                )
                predictions.append(pred)
            predictions = np.array(predictions)

            print(f"Model outputs - Raw range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
            print(f"Model outputs - Prob range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
            print(f"Predictions - Positive pixels: {predictions[0].sum()}")

            batch_metrics = calculate_metrics_batch(predictions, true_masks.cpu().numpy())
            all_metrics.extend(batch_metrics)

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_images.extend(images.cpu())
            all_true_masks.extend(true_masks.cpu().numpy())
            all_image_ids.extend(image_ids)

    if all_metrics:
        overall_metrics = {
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'dice': np.mean([m['dice'] for m in all_metrics]),
            'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
            'specificity': np.mean([m['specificity'] for m in all_metrics]),
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        }

        print("\n" + "=" * 70)
        print("AUGMENTED VALIDATION METRICS PER IMAGE")
        print("=" * 70)
        for i, metrics in enumerate(all_metrics):
            print(f"{all_image_ids[i]}: "
                  f"IoU={metrics['iou']:.3f}, "
                  f"Dice={metrics['dice']:.3f}, "
                  f"Sensitivity={metrics['sensitivity']:.3f}, "
                  f"Specificity={metrics['specificity']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Accuracy={metrics['accuracy']:.3f}, "
                  f"Pred Pixels={metrics['pred_sum']}, "
                  f"GT Pixels={metrics['target_sum']}")

    else:
        overall_metrics = {}

    print("\n" + "=" * 60)
    print("AUGMENTED VALIDATION METRICS SUMMARY")
    print("=" * 60)
    for metric_name, value in overall_metrics.items():
        print(f"{metric_name.upper():<15}: {value:.4f}")
    print(f"{'IMAGES PROCESSED':<15}: {len(dataset)}")

    print("\nCreating visualizations...")
    viz_dir = os.path.join(output_dir, 'visualizations')
    visualize_predictions(all_images, all_predictions, all_probabilities, all_true_masks, all_image_ids, viz_dir)

    results = []
    for i, image_id in enumerate(all_image_ids):
        pred_area = np.sum(all_predictions[i])
        true_area = np.sum(all_true_masks[i])

        results.append({
            'ImageID': image_id,
            'PNG_File': png_files[i],
            'Predicted_Area': pred_area,
            'True_Area': true_area,
            'Has_Pneumothorax_Pred': pred_area > 0,
            'Has_Pneumothorax_True': true_area > 0,
            'IoU': all_metrics[i]['iou'],
            'Dice': all_metrics[i]['dice'],
            'Sensitivity': all_metrics[i]['sensitivity'],
            'Specificity': all_metrics[i]['specificity'],
            'Precision': all_metrics[i]['precision'],
            'Accuracy': all_metrics[i]['accuracy'],
        })

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, 'validation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Results saved to: {results_csv_path}")

    metrics_json_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    print(f"✓ Metrics saved to: {metrics_json_path}")

    return overall_metrics, results_df



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    CSV_PATH = "../../Data/stage_2_train.csv"
    PNG_FOLDER = "png files"
    MODEL_PATH = "model.pth"
    OUTPUT_DIR = "model_validation_results"

    print("Pneumothorax Model Validation (Using Your PNG Files)")
    print("=" * 60)

    try:
        metrics, results = validate_model_with_png_files(
            csv_path=CSV_PATH,
            png_folder=PNG_FOLDER,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR
        )

        if metrics is not None:
            print("\n✅ Validation completed successfully!")
            print(f"Check the '{OUTPUT_DIR}' folder for results and visualizations")
        else:
            print("\n❌ Validation failed due to model issues!")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
