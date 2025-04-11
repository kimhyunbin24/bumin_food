import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from PIL import Image
import matplotlib.patches as patches

# 커스텀 모듈 임포트
from config import (
    IMAGE_DIR, XML_BASE_DIR, MASK_BASE_DIR, TARGET_SIZE, VOLUME_LABELS,
    choose_model_mode, enable_memory_growth, clear_memory
)
from data import MultiClassFoodDetectionDataset, custom_collate_fn, parse_annotation
from models import (
    get_detection_model, VolumeClassificationModel, load_pretrained_models,
    train_detection_model, train_volume_model
)


def visualize_from_paths(image_path, xml_path, mask_path, model, device, label_dict, save_path=None):
    """
    Load and visualize image, XML, and mask files using the model for prediction.
    
    Args:
        image_path: Path to the image file
        xml_path: Path to the XML file
        mask_path: Path to the mask file
        model: Model to use for prediction
        device: Device to use (CPU or GPU)
        label_dict: Class label mapping dictionary
        save_path: Path to save visualization results (optional)
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch
    from data import parse_annotation
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Extract bounding boxes and labels from XML file
    boxes, labels, mask_path, size = parse_annotation(xml_path)
    
    # Load mask
    mask = np.load(mask_path) if isinstance(mask_path, str) and os.path.exists(mask_path) else np.zeros_like(image[:, :, 0])
    
    # Resize image and mask
    image = Image.fromarray(image).resize((224, 224))
    mask = Image.fromarray(mask).resize((224, 224))
    
    # Convert image and mask to tensors
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        prediction = model([image.to(device)])
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth bounding boxes
    axes[0, 1].imshow(image.permute(1, 2, 0).cpu().numpy())
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        class_name = next((k for k, v in label_dict.items() if v == label), "Unknown")
        axes[0, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red'))
        axes[0, 1].text(x1, y1, f'{class_name}', color='red')
    axes[0, 1].set_title('Ground Truth Boxes')
    axes[0, 1].axis('off')
    
    # Ground truth mask
    axes[1, 0].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Ground Truth Mask')
    axes[1, 0].axis('off')
    
    # Prediction results
    axes[1, 1].imshow(image.permute(1, 2, 0).cpu().numpy())
    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = box.cpu().numpy()
            class_name = next((k for k, v in label_dict.items() if v == label.item()), "Unknown")
            axes[1, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='blue'))
            axes[1, 1].text(x1, y1, f'{class_name} ({score:.2f})', color='blue')
    axes[1, 1].set_title('Predictions')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save results
    if save_path:
        plt.savefig(save_path)
    
    return fig


def test_integrated_models(detection_model, volume_model, val_loader, device, label_dict, dataset, val_indices):
    """
    Test and visualize integrated models
    """
    detection_model.eval()
    volume_model.eval()

    # Dictionary for collecting samples by class
    class_samples = {}
    class_targets = {}

    # Number of samples to show per class
    samples_per_class = 4

    print("Collecting visualization samples for each class...")

    # 1. Collect samples for each class
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            for i, target in enumerate(targets):
                # Convert current batch index to overall val_indices index
                dataset_idx = batch_idx * val_loader.batch_size + i
                if dataset_idx >= len(val_indices):  # Handle last batch
                    continue

                # Use original dataset class labels
                class_id = dataset.class_labels[val_indices[dataset_idx]]

                # Initialize dictionary (if not exists)
                if class_id not in class_samples:
                    class_samples[class_id] = []
                    class_targets[class_id] = []

                # Add sample if not enough samples collected for this class
                if len(class_samples[class_id]) < samples_per_class:
                    class_samples[class_id].append(images[i])
                    class_targets[class_id].append(target)

    # 2. Visualize all samples at once
    all_samples = []
    all_targets = []
    all_class_ids = []

    for class_id in sorted(class_samples.keys()):
        samples = class_samples[class_id]
        sample_targets = class_targets[class_id]

        if len(samples) == 0:
            print(f"No samples found for class {class_id}.")
            continue

        class_name = next((name for name, idx in label_dict.items() if idx == class_id), "Unknown")
        print(f"Collected {len(samples)} samples for class {class_id} ({class_name}).")

        all_samples.extend(samples)
        all_targets.extend(sample_targets)
        all_class_ids.extend([class_id] * len(samples))

    # Create a figure with subplots
    num_samples = len(all_samples)
    num_rows = (num_samples + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Visualize each sample
    for idx, (image, target, class_id) in enumerate(zip(all_samples, all_targets, all_class_ids)):
        if idx >= len(axes):
            break

        # Move to GPU
        image = image.to(device)

        # Perform object detection
        detection = detection_model([image])[0]

        # Filter prediction results
        boxes = detection['boxes']
        scores = detection['scores']
        masks = detection['masks']
        labels = detection['labels']

        # Debug: Print detected object information
        print(f"Sample {idx + 1}: Number of objects with confidence > 0.1: {torch.sum(scores > 0.1).item()}")

        # Apply confidence threshold (using lower threshold 0.1)
        valid_idx = scores >= 0.1
        pred_boxes = boxes[valid_idx]
        pred_masks = masks[valid_idx]
        pred_labels = labels[valid_idx]

        # Ground truth information
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        mask_path = target['mask_path']

        # Food volume classification model prediction
        volume_class = None
        if volume_model is not None:
            volume_outputs = volume_model(image.unsqueeze(0))
            volume_class = torch.argmax(volume_outputs).item()

        # Perform visualization using new function
        fig = visualize_from_paths(
            dataset.image_files[val_indices[dataset_idx]],  # Pass image path
            dataset.xml_files[val_indices[dataset_idx]],  # Pass XML file path
            target['mask_path'],
            detection_model,
            device,
            label_dict
        )

        class_name = next((name for name, idx in label_dict.items() if idx == class_id), "Unknown")
        axes[idx].set_title(f"Class {class_id} ({class_name}) - Sample {idx + 1}/{num_samples}")
        axes[idx].axis('off')

    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show(block=True)

    print("\nVisualization completed for all classes.")


def main():
    """
    Main execution function
    """
    # GPU performance improvement settings
    torch.backends.cudnn.benchmark = True

    # Debugging tool settings: To see CUDA errors more clearly
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Enable dynamic memory allocation
    enable_memory_growth()

    # Create dataset (optimized file matching and label index modification)
    dataset = MultiClassFoodDetectionDataset(IMAGE_DIR, XML_BASE_DIR, MASK_BASE_DIR, target_size=TARGET_SIZE)

    print("\nDataset details:")
    for idx in range(min(5, len(dataset))):  # Only output first 5 items
        img, target = dataset[idx]
        print(f"Dataset Item {idx}:")
        print(f"  Boxes: {target['boxes']}")
        print(f"  Labels: {target['labels']}")
        print(f"  Masks Shape: {target['masks'].shape}")
        print(f"  Volume Class: {target['volume_class']}")

    num_classes = len(dataset.label_dict)
    print(f"Number of detected classes: {num_classes}")
    print(f"Class mapping: {dataset.label_dict}")

    # Split dataset using Stratified sampling
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=dataset.class_labels  # Stratified sampling based on class labels
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    val_class_labels = [dataset.class_labels[i] for i in val_indices]
    val_class_counts = Counter(val_class_labels)
    print("\nValidation dataset class counts:")
    for class_id, count in val_class_counts.items():
        class_name = next((name for name, idx in dataset.label_dict.items() if idx == class_id), "Unknown")
        print(f"Class {class_id} ({class_name}): {count} items")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model mode selection
    mode_choice = choose_model_mode()

    if mode_choice == 1:
        # Load saved weights
        detection_model, volume_model = load_pretrained_models(num_classes, device)

        if detection_model is None or volume_model is None:
            print("Failed to load model weights. Training new models.")
            # Initialize models (specify exact number of classes)
            detection_model = get_detection_model(num_classes=num_classes).to(device)
            volume_model = VolumeClassificationModel(num_classes=4).to(device)
    else:
        # 1. Train food detection model
        print("\n----- Step 1: Training food detection model -----")
        detection_model = get_detection_model(num_classes=num_classes).to(device)
        train_detection_model(detection_model, train_loader, val_loader, device, num_epochs=1)

        # 2. Train food volume classification model
        print("\n----- Step 2: Training food volume classification model -----")
        volume_model = VolumeClassificationModel(num_classes=4).to(device)
        train_volume_model(volume_model, train_loader, val_loader, device, num_epochs=1)

    print("Models ready!")

    # 3. Final model testing (sequential visualization by class)
    print("\n----- Sequential visualization by class -----")
    test_integrated_models(detection_model, volume_model, val_loader, device, dataset.label_dict, dataset, val_indices)
    
    # 4. Visualize random images from the specified directory
    print("\n----- Visualizing random images from the specified directory -----")
    visualize_random_images_from_directory(
        "C:\\Users\\furim\\Desktop\\Bumin_dataset\\202306\\전영숙_F76_45",
        detection_model,
        volume_model,
        device,
        dataset.label_dict,
        num_images=4
    )

    # 데이터셋 검증
    print("\n=== 학습 데이터셋 검증 ===")
    train_class_counts, train_xml_counts, train_folder_counts = validate_dataset(train_dataset)
    
    print("\n=== 검증 데이터셋 검증 ===")
    val_class_counts, val_xml_counts, val_folder_counts = validate_dataset(val_dataset)


def visualize_random_images_from_directory(directory_path, detection_model, volume_model, device, label_dict, num_images=4):
    """
    Visualize random images from the specified directory
    
    Args:
        directory_path: Path to the directory containing images
        detection_model: Detection model to use
        volume_model: Volume classification model to use
        device: Device to use (CPU or GPU)
        label_dict: Class label mapping dictionary
        num_images: Number of images to visualize
    """
    import os
    import random
    from PIL import Image
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get list of image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    # Randomly select images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    print(f"Selected {len(selected_images)} images for visualization")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    # Process each selected image
    for idx, image_file in enumerate(selected_images):
        if idx >= len(axes):
            break
            
        image_path = os.path.join(directory_path, image_file)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Perform object detection
        detection_model.eval()
        with torch.no_grad():
            detection = detection_model([image_tensor])[0]
        
        # Filter prediction results
        boxes = detection['boxes']
        scores = detection['scores']
        labels = detection['labels']
        
        # Apply confidence threshold
        valid_idx = scores >= 0.5
        pred_boxes = boxes[valid_idx]
        pred_labels = labels[valid_idx]
        pred_scores = scores[valid_idx]
        
        # Perform volume classification
        volume_class = None
        if volume_model is not None:
            volume_model.eval()
            with torch.no_grad():
                volume_outputs = volume_model(image_tensor)
                volume_class = torch.argmax(volume_outputs).item()
        
        # Visualize
        axes[idx].imshow(image_np)
        
        # Draw bounding boxes
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.cpu().numpy()
            class_name = next((k for k, v in label_dict.items() if v == label.item()), "Unknown")
            axes[idx].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='blue'))
            axes[idx].text(x1, y1, f'{class_name} ({score:.2f})', color='blue')
        
        # Add volume information if available
        if volume_class is not None:
            volume_text = f"Volume: {volume_class}"
            axes[idx].text(10, 10, volume_text, color='white', fontsize=10, 
                          bbox=dict(facecolor='black', alpha=0.7))
        
        axes[idx].set_title(f'Image {idx+1}: {image_file}')
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(len(selected_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show(block=True)
    
    print("Random image visualization completed")


if __name__ == "__main__":
    main()