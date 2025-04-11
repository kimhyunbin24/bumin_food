import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import xml.etree.ElementTree as ET
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Global configuration variables
CONFIG = {
    'image_dir': r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\food data",
    'xml_base_dir': r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\food data\라벨 일부\xml",
    'detection_model_path': 'food_detection_model_with_masks.pth',
    'volume_model_path': 'food_volume_classification_model.pth',
    'target_size': (224, 224)
}

# Volume class labels
VOLUME_LABELS = ['0-25%', '25-50%', '50-75%', '75-100%']


def create_label_mapping(image_base_dir):
    """
    Creates a label mapping based on class folders in the given directory.
    """
    # Collect folders that start with a number
    class_folders = [
        d for d in os.listdir(image_base_dir)
        if os.path.isdir(os.path.join(image_base_dir, d)) and d[0].isdigit()
    ]

    # Sort by folder name
    class_folders.sort(key=lambda x: int(x.split('.')[0]))

    # Create label mapping (starting from 2)
    label_dict = {name: idx + 2 for idx, name in enumerate(class_folders)}
    label_dict['dish'] = 1  # Add 'dish' label

    return label_dict


class VolumeClassificationModel(nn.Module):
    """
    Model for food volume classification
    """

    def __init__(self, num_classes=4):
        super().__init__()
        # Use ResNet50 backbone
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def get_detection_model(num_classes):
    """
    Initialize a Mask R-CNN model and adjust for the number of classes.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Update box predictor (number of classes + background)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes + 1)

    # Update mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 128
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes + 1)

    return model


def load_models(num_classes, device, config):
    """
    Load both detection and volume models.
    """
    # 1. Load detection model
    detection_model = get_detection_model(num_classes=num_classes)

    try:
        detection_model.load_state_dict(torch.load(config['detection_model_path']))
        print("Successfully loaded food detection model weights.")
    except Exception as e:
        print(f"Failed to load food detection model weights: {e}")
        return None, None

    detection_model = detection_model.to(device)
    detection_model.eval()

    # 2. Load volume classification model
    volume_model = VolumeClassificationModel(num_classes=4)

    try:
        volume_model.load_state_dict(torch.load(config['volume_model_path']))
        print("Successfully loaded food volume classification model weights.")
    except Exception as e:
        print(f"Failed to load food volume classification model weights: {e}")
        return detection_model, None

    volume_model = volume_model.to(device)
    volume_model.eval()

    return detection_model, volume_model


def preprocess_image(image_path, target_size=None):
    """
    Load and preprocess the image.
    """
    # Load image
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load the image: {image_path}")

    # Limit image size
    max_size = 640
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    if target_size:
        image_rgb_resized = cv2.resize(image_rgb, target_size)
    else:
        image_rgb_resized = image_rgb

    return image_rgb, image_rgb_resized


def detect_and_classify(image_path, detection_model, volume_model, device, label_dict, confidence_threshold=0.5):
    """
    Detect food and classify volume in an image.
    """
    # Load and preprocess image
    original_image, resized_image = preprocess_image(image_path, target_size=(224, 224))

    # Convert image to tensor - for detection model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(resized_image).unsqueeze(0).to(device)

    # Detection model prediction
    with torch.no_grad():
        detection_results = detection_model(image_tensor)

    # Extract results
    pred_boxes = detection_results[0]['boxes'].cpu().numpy()
    pred_scores = detection_results[0]['scores'].cpu().numpy()
    pred_labels = detection_results[0]['labels'].cpu().numpy()
    pred_masks = detection_results[0]['masks'].squeeze().cpu().numpy()

    # Apply confidence threshold
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    pred_labels = pred_labels[mask]

    # Food volume classification
    volume_class = None
    if volume_model is not None:
        with torch.no_grad():
            volume_outputs = volume_model(image_tensor)
            volume_class = torch.argmax(volume_outputs, dim=1).item()
            volume_probs = torch.softmax(volume_outputs, dim=1).cpu().numpy()

    return original_image, pred_boxes, pred_scores, pred_labels, pred_masks, volume_class, volume_probs


def visualize_results(image, boxes, scores, labels, label_dict, volume_class=None, volume_probs=None, save_path=None):
    """
    Visualize detection and classification results.
    """
    plt.figure(figsize=(14, 10))
    plt.imshow(image)

    # Display total food volume in the image title
    if volume_class is not None:
        volume_text = VOLUME_LABELS[volume_class]
        volume_confidence = volume_probs[0][volume_class] if volume_probs is not None else None

        if volume_confidence is not None:
            plt.title(f"Food Detection Results - Total Food Volume: {volume_text} (Probability: {volume_confidence:.2f})", fontsize=16)
        else:
            plt.title(f"Food Detection Results - Total Food Volume: {volume_text}", fontsize=16)
    else:
        plt.title("Food Detection Results", fontsize=16)

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box

        # Find label name (reverse mapping)
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')

        # Use different color for dish
        if label == 1:  # dish
            color = 'blue'
            text = f'Dish: {score:.2f}'
        else:
            color = 'red'
            text = f'{label_name}: {score:.2f}'

            # Add volume information for food
            if volume_class is not None:
                text += f' | Volume: {VOLUME_LABELS[volume_class]}'

        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          fill=False, edgecolor=color, linewidth=2))

        plt.text(xmin, ymin - 10, text,
                 color=color, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def process_folder(folder_path, detection_model, volume_model, device, label_dict, save_results=True):
    """
    Process all images in the given folder.
    """
    # Create folder for saving results
    if save_results:
        results_folder = os.path.join(os.path.dirname(folder_path), 'detection_results')
        os.makedirs(results_folder, exist_ok=True)

    # List of image extensions
    image_extensions = ('.jpg', '.jpeg', '.png')

    # Process all images in folder
    total_images = sum(1 for filename in os.listdir(folder_path) if filename.lower().endswith(image_extensions))
    processed_images = 0

    print(f"\nProcessing {total_images} images in total.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            processed_images += 1
            image_path = os.path.join(folder_path, filename)
            print(f"\n[{processed_images}/{total_images}] Processing: {filename}")

            try:
                # Process image
                original_image, pred_boxes, pred_scores, pred_labels, pred_masks, volume_class, volume_probs = \
                    detect_and_classify(image_path, detection_model, volume_model, device, label_dict)

                print(f"Number of objects detected: {len(pred_boxes)}")
                if volume_class is not None:
                    print(f"Predicted food volume: {VOLUME_LABELS[volume_class]}")

                # Save or display results
                if save_results:
                    save_path = os.path.join(results_folder, f'result_{filename}')
                    visualize_results(original_image, pred_boxes, pred_scores, pred_labels,
                                      label_dict, volume_class, volume_probs, save_path)
                    print(f"Result saved: {save_path}")
                else:
                    visualize_results(original_image, pred_boxes, pred_scores, pred_labels,
                                      label_dict, volume_class, volume_probs)
                    input("Press Enter to see the next image...")

            except Exception as e:
                print(f"Error processing image: {e}")
                import traceback
                traceback.print_exc()
                continue


def process_single_image(image_path, detection_model, volume_model, device, label_dict, save_path=None):
    """
    Process a single image.
    """
    try:
        # Process image
        original_image, pred_boxes, pred_scores, pred_labels, pred_masks, volume_class, volume_probs = \
            detect_and_classify(image_path, detection_model, volume_model, device, label_dict)

        print(f"Number of objects detected: {len(pred_boxes)}")

        # Print class and confidence for each object
        for label, score in zip(pred_labels, pred_scores):
            label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')
            print(f"  - {label_name}: {score:.2f}")

        if volume_class is not None:
            print(f"Predicted food volume: {VOLUME_LABELS[volume_class]}")

        # Visualize results
        visualize_results(original_image, pred_boxes, pred_scores, pred_labels,
                          label_dict, volume_class, volume_probs, save_path)

        if save_path:
            print(f"Result saved: {save_path}")

        return original_image, pred_boxes, pred_scores, pred_labels, volume_class

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def main(config=None):
    """
    Main execution function that can optionally receive configuration values.
    """
    # Use default or provided configuration
    cfg = config or CONFIG

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create label mapping
    label_dict = create_label_mapping(cfg['image_dir'])

    # Check number of classes
    num_classes = len(label_dict)
    print(f"Number of classes: {num_classes}")
    print("Class mapping:", label_dict)

    # Load models
    detection_model, volume_model = load_models(num_classes, device, cfg)

    if detection_model is None:
        print("Failed to load models. Exiting program.")
        return

    # Test folder path (can be changed here if needed)
    folder_path = r"C:\Users\furim\Desktop\Bumin_dataset\202306\전영숙_F76_45"

    # Select processing method
    print("\nSelect processing method:")
    print("1. Process all images in folder")
    print("2. Process single image")

    choice = input("Enter number (1/2): ").strip()

    if choice == '1':
        # Process all images in folder
        process_folder(folder_path, detection_model, volume_model, device, label_dict, save_results=True)
    else:
        # Select single image
        print("\nYou've selected single image processing.")
        image_path = input("Enter the image file path: ").strip()

        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return

        # Result save path
        save_path = os.path.join(os.path.dirname(image_path), f'result_{os.path.basename(image_path)}')

        # Process single image
        process_single_image(image_path, detection_model, volume_model, device, label_dict, save_path)


if __name__ == "__main__":
    # Run with default configuration
    main()

    # Example of running with custom configuration
    # custom_config = {
    #     'image_dir': r"alternative/path/to/image/dir",
    #     'xml_base_dir': r"alternative/path/to/xml/dir",
    #     'detection_model_path': 'food_detection_model_with_masks.pth',
    #     'volume_model_path': 'food_volume_classification_model.pth',
    #     'target_size': (256, 256)
    # }
    # main(custom_config)