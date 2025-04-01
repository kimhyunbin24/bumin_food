import os
import cv2
import torch
import torchvision
import torchvision.models as models
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F, Resize, ToTensor, Compose, Normalize
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn.functional as F_nn
from PIL import Image
import glob
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 사용 가능한 모든 GPU 사용
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.3, device=0)

def enable_memory_growth():
    # GPU 메모리 동적 할당 활성화
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

# 모델 학습 전에 호출
enable_memory_growth()

def clear_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 추가: 메모리 사용량 출력
        print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB Allocated")


def create_label_mapping(image_base_dir):
    class_folders = [
        d for d in os.listdir(image_base_dir)
        if os.path.isdir(os.path.join(image_base_dir, d)) and d[0].isdigit()
    ]

    # 폴더명 기준으로 정렬
    class_folders.sort(key=lambda x: int(x.split('.')[0]))

    # 클래스 라벨을 1부터 시작하도록 변경 (0은 배경용)
    label_dict = {name: idx + 1 for idx, name in enumerate(class_folders)}

    return label_dict


def parse_annotation(xml_path, target_size=None):
    """
    XML 파일에서 어노테이션 정보를 추출하고 마스크를 생성합니다.

    Parameters:
    - xml_path: XML 파일 경로
    - target_size: 리사이징할 목표 크기 (선택적)

    Returns:
    - boxes: 바운딩 박스 텐서
    - labels: 클래스 라벨 텐서
    - masks: 마스크 텐서
    - size: 원본 또는 리사이즈된 이미지 크기
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 원본 이미지 크기 정보 추출 (안전하게)
        size_elem = root.find('size')
        if size_elem is None:
            # 크기 정보가 없다면 기본값 사용
            orig_width = target_size[0] if target_size else 224
            orig_height = target_size[1] if target_size else 224
        else:
            width_elem = size_elem.find('width')
            height_elem = size_elem.find('height')

            orig_width = int(width_elem.text) if width_elem is not None else (target_size[0] if target_size else 224)
            orig_height = int(height_elem.text) if height_elem is not None else (target_size[1] if target_size else 224)

        # 스케일링 계수 계산 (리사이즈가 필요한 경우)
        width_scale = 1.0
        height_scale = 1.0
        if target_size:
            width_scale = target_size[0] / orig_width
            height_scale = target_size[1] / orig_height

        boxes = []
        masks = []

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')

            # XML에 바운딩 박스 정보가 없는 경우 스킵
            if bndbox is None:
                continue

            # 각 좌표 요소를 안전하게 추출
            xmin_elem = bndbox.find('xmin')
            ymin_elem = bndbox.find('ymin')
            xmax_elem = bndbox.find('xmax')
            ymax_elem = bndbox.find('ymax')

            # 요소가 없으면 스킵
            if not all([xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                continue

            xmin = int(float(xmin_elem.text) * width_scale)
            ymin = int(float(ymin_elem.text) * height_scale)
            xmax = int(float(xmax_elem.text) * width_scale)
            ymax = int(float(ymax_elem.text) * height_scale)

            # 좌표 검증
            if target_size:
                xmin = max(0, min(xmin, target_size[0] - 1))
                ymin = max(0, min(ymin, target_size[1] - 1))
                xmax = max(0, min(xmax, target_size[0] - 1))
                ymax = max(0, min(ymax, target_size[1] - 1))

            boxes.append([xmin, ymin, xmax, ymax])

            # 마스크 생성
            height = target_size[1] if target_size else orig_height
            width = target_size[0] if target_size else orig_width
            mask = np.zeros((height, width), dtype=np.uint8)

            # 세그멘테이션 데이터가 없는 경우 바운딩 박스로 대체
            mask[ymin:ymax, xmin:xmax] = 1
            masks.append(mask)

        # Tensor 변환
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        else:
            # 객체가 없는 경우 빈 텐서 생성
            height = target_size[1] if target_size else orig_height
            width = target_size[0] if target_size else orig_width
            boxes = torch.empty((0, 4), dtype=torch.float32)
            masks = torch.empty((0, height, width), dtype=torch.uint8)

        # 빈 라벨 텐서 반환 (실제 라벨은 데이터셋에서 직접 지정)
        labels = torch.empty((0,), dtype=torch.int64)

        size = target_size if target_size else (orig_width, orig_height)
        return boxes, labels, masks, size

    except ET.ParseError:
        print(f"XML 파일 파싱 오류: {xml_path}")
        # 파싱 오류 시 빈 텐서 반환
        height = target_size[1] if target_size else 224
        width = target_size[0] if target_size else 224
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0, height, width), dtype=torch.uint8),
            target_size or (width, height)
        )


def calculate_volume_class(mask_array):
    """
    마스크 배열을 기반으로 음식 양 클래스를 계산합니다.
    클래스 값:
    0: 0-25%
    1: 25-50%
    2: 50-75%
    3: 75-100%
    """
    # 마스크가 3차원이면 첫 번째 차원 제거
    if mask_array.ndim > 2:
        mask_array = mask_array[0]  # 첫 번째 채널만 사용

    # 01011001 음식(라벨 2)에 해당하는 픽셀만 선택
    food_mask = (mask_array == 2).astype(np.uint8)
    dish_mask = (mask_array == 1).astype(np.uint8)

    # 음식과 접시가 모두 존재하는 경우만 계산
    if np.sum(food_mask) > 0 and np.sum(dish_mask) > 0:
        # 음식 픽셀 수 / 접시 픽셀 수 = 비율
        food_pixel_count = np.sum(food_mask)
        dish_pixel_count = np.sum(dish_mask)
        ratio = food_pixel_count / dish_pixel_count

        # 비율에 따른 클래스 결정
        if ratio <= 0.25:
            return 0  # 0-25%
        elif ratio <= 0.50:
            return 1  # 25-50%
        elif ratio <= 0.75:
            return 2  # 50-75%
        else:
            return 3  # 75-100%
    else:
        # 단일 마스크인 경우 마스크 영역 비율로 판단
        if mask_array.ndim == 2:
            # 마스크 픽셀 비율 계산
            pixel_ratio = np.sum(mask_array) / (mask_array.shape[0] * mask_array.shape[1])

            # 음식 양 클래스 결정
            if pixel_ratio <= 0.25:
                return 0  # 0-25%
            elif pixel_ratio <= 0.50:
                return 1  # 25-50%
            elif pixel_ratio <= 0.75:
                return 2  # 50-75%
            else:
                return 3  # 75-100%
        # 음식이나 접시가 없는 경우 기본값
        return 0


class MultiClassFoodDetectionDataset(Dataset):
    def __init__(self, image_base_dir, xml_base_dir, mask_base_dir, transform=None, target_size=(224, 224)):
        # 클래스 라벨 매핑
        self.label_dict = create_label_mapping(image_base_dir)

        self.image_files = []
        self.xml_files = []
        self.mask_files = []
        self.class_labels = []  # 각 데이터의 클래스 라벨 저장
        self.transform = transform or self._get_transform(target_size)
        self.target_size = target_size

        # 클래스별 파일 세트 카운트
        class_counts = {class_name: 0 for class_name in self.label_dict.keys()}

        print("데이터 매칭 시작...")

        # 1. 먼저 XML 파일 목록 구축 (인덱싱)
        print("XML 파일 인덱싱 중...")
        xml_files_dict = {}
        for root, _, files in os.walk(xml_base_dir):
            for file in files:
                if file.endswith('.xml'):
                    basename = os.path.splitext(file)[0]
                    xml_files_dict[basename] = os.path.join(root, file)

        print(f"XML 파일 {len(xml_files_dict)}개 인덱싱 완료")

        # 2. 마스크 파일 목록 구축 (인덱싱)
        print("마스크 파일 인덱싱 중...")
        mask_files_dict = {}
        for root, _, files in os.walk(mask_base_dir):
            for file in files:
                if file.endswith('.npy'):
                    basename = os.path.splitext(file)[0]
                    mask_files_dict[basename] = os.path.join(root, file)

        print(f"마스크 파일 {len(mask_files_dict)}개 인덱싱 완료")

        # 3. 이미지 파일 검색 및 매칭
        print("이미지 파일 수집 및 매칭 중...")
        for class_name in self.label_dict.keys():
            class_dir = os.path.join(image_base_dir, class_name)

            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        basename = os.path.splitext(file)[0]
                        img_path = os.path.join(root, file)

                        # 인덱싱된 사전에서 해당 파일명으로 XML과 마스크 찾기
                        xml_path = xml_files_dict.get(basename)
                        mask_path = mask_files_dict.get(basename)

                        # 모든 파일이 존재하는 경우 데이터셋에 추가
                        if xml_path and mask_path:
                            self.image_files.append(img_path)
                            self.xml_files.append(xml_path)
                            self.mask_files.append(mask_path)

                            # 클래스 라벨 저장
                            self.class_labels.append(self.label_dict[class_name])

                            # 클래스별 카운트 증가
                            class_counts[class_name] += 1

        # 클래스별 데이터 수 출력
        print("\n--- 클래스별 데이터 세트(이미지+XML+마스크) 통계 ---")
        for class_name, count in class_counts.items():
            print(f"클래스 '{class_name}' (라벨: {self.label_dict[class_name]}): {count}개 세트")

        total_sets = sum(class_counts.values())
        print(f"\n총 데이터 세트 수: {total_sets}")
        print(f"클래스 수: {len(self.label_dict)}")
        print(f"클래스 라벨 범위: 0 ~ {len(self.label_dict) - 1}")

    def _get_transform(self, target_size):
        return Compose([
            Resize(target_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        xml_path = self.xml_files[idx]
        mask_path = self.mask_files[idx]
        class_label = self.class_labels[idx]

        # 이미지를 로드하고 변환 적용
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        # XML 파싱에 클래스 라벨 매핑이 사용되지 않도록 수정
        boxes, _, _, _ = parse_annotation(
            xml_path,
            self.target_size
        )

        # 라벨은 저장된 클래스 라벨 사용
        labels = torch.tensor([class_label], dtype=torch.int64)

        # 마스크 파일 로드 (.npy 파일)
        mask_array = np.load(mask_path)

        # 마스크 리사이즈
        if mask_array.shape[:2] != self.target_size[::-1]:  # numpy는 (height, width) 순서
            mask_resized = cv2.resize(mask_array, self.target_size, interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_array

        # 마스크를 텐서로 변환
        food_mask = (mask_resized > 0).astype(np.uint8)
        food_mask_tensor = torch.tensor(food_mask, dtype=torch.uint8).unsqueeze(0)

        # 음식 양 클래스 계산
        volume_class = calculate_volume_class(food_mask_tensor.numpy())

        # 객체당 하나의 마스크 생성
        if len(boxes) > 0:
            masks = torch.zeros((len(boxes), *self.target_size), dtype=torch.uint8)
            for i in range(len(boxes)):
                masks[i] = torch.tensor(food_mask, dtype=torch.uint8)
        else:
            masks = torch.zeros((0, *self.target_size), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }
        return image_tensor, target


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 이미지가 동일한 크기라고 가정 (모두 리사이즈됨)
    images_tensor = torch.stack(images)

    return images_tensor, targets


def get_detection_model(num_classes):
    # Mask R-CNN 모델 초기화
    model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Box predictor 업데이트 (클래스 수 + 배경)
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes + 1)

    # Mask predictor 업데이트
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 128
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes + 1)

    return model


class VolumeClassificationModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # ResNet50 백본 사용
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 마지막 fully connected 레이어 수정
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# def train_detection_model(model, train_loader, val_loader, device, num_epochs=10):
#     # 옵티마이저와 학습률 스케줄러 설정
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#
#     # 학습 루프
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch}/{num_epochs - 1}")
#
#         # 학습 모드
#         model.train()
#         train_loss = 0
#         for i, (images, targets) in enumerate(train_loader):
#             images = images.to(device)
#             targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
#
#             # 학습에 필요한 필드만 유지
#             targets_clean = []
#             for t in targets:
#                 targets_clean.append({
#                     'boxes': t['boxes'],
#                     'labels': t['labels'],
#                     'masks': t['masks'],
#                     'image_id': t['image_id']
#                 })
#
#             loss_dict = model(images, targets_clean)
#             losses = sum(loss for loss in loss_dict.values())
#
#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()
#
#             train_loss += losses.item()
#
#             if i % 10 == 0:
#                 print(f"Batch [{i}/{len(train_loader)}], Loss: {losses.item():.4f}")
#
#         avg_train_loss = train_loss / len(train_loader)
#         print(f"Average Training Loss: {avg_train_loss:.4f}")
#
#         lr_scheduler.step()
#
#         # 검증 모드
#         model.eval()
#         val_loss = 0
#         print("\n검증 세트 평가 중...")
#
#         with torch.no_grad():
#             for i, (images, targets) in enumerate(val_loader):
#                 images = images.to(device)
#
#                 # 검증 시에는 모델이 예측만 수행
#                 outputs = model(images)
#                 for output in outputs:
#                     num_detections = len(output['boxes'])
#                     print(f"Found {num_detections} objects in image")
#
#             print("Validation completed")
#
#         clear_memory()
#
#     # 모델 저장
#     torch.save(model.state_dict(), "food_detection_model_with_masks.pth")
#     print('Detection model saved')


# *******************
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

def train_detection_model(model, train_loader, val_loader, device, num_epochs=10):
    # 옵티마이저와 학습률 스케줄러 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # GradScaler 초기화
    scaler = GradScaler()

    # 학습 루프
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        # 학습 모드
        model.train()
        train_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # 학습에 필요한 필드만 유지
            targets_clean = []
            for t in targets:
                targets_clean.append({
                    'boxes': t['boxes'],
                    'labels': t['labels'],
                    'masks': t['masks'],
                    'image_id': t['image_id']
                })

            # Mixed precision 적용
            with autocast():  # Autocast가 활성화된 영역
                loss_dict = model(images, targets_clean)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()

            # Gradient scaling (자동적으로 gradient가 너무 작아지는 문제를 방지)
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()  # 스케일러 상태 업데이트

            train_loss += losses.item()

            if i % 10 == 0:
                print(f"Batch [{i}/{len(train_loader)}], Loss: {losses.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        lr_scheduler.step()

        # 검증 모드
        model.eval()
        val_loss = 0
        print("\n검증 세트 평가 중...")

        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                images = images.to(device)

                # 검증 시에는 모델이 예측만 수행
                outputs = model(images)
                for output in outputs:
                    num_detections = len(output['boxes'])
                    print(f"Found {num_detections} objects in image")

            print("Validation completed")

        clear_memory()

    # 모델 저장
    torch.save(model.state_dict(), "food_detection_model_with_masks.pth")
    print('Detection model saved')




def train_volume_model(model, train_loader, val_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)

            # 각 타겟에서 volume_class 추출
            volume_classes = torch.stack([t['volume_class'] for t in targets]).squeeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, volume_classes)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += volume_classes.size(0)
            correct += predicted.eq(volume_classes).sum().item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Volume Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Accuracy: {100. * correct / total:.2f}%')

        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                volume_classes = torch.stack([t['volume_class'] for t in targets]).squeeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, volume_classes)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += volume_classes.size(0)
                correct += predicted.eq(volume_classes).sum().item()

            print(f'Volume Validation Loss: {val_loss / len(val_loader):.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')

    # 모델 저장
    torch.save(model.state_dict(), "food_volume_classification_model.pth")
    print('Volume classification model saved')


def visualize_comparison(image, gt_boxes, gt_labels, mask_path, pred_boxes, pred_labels, pred_masks, volume_labels,
                         label_dict, idx=0):
    """
    원본, 그라운드 트루스 박스, 실제 마스크 파일, 예측 결과를 서브플롯으로 시각화

    Parameters:
    - image: 원본 이미지 텐서
    - gt_boxes: 그라운드 트루스 바운딩 박스
    - gt_labels: 그라운드 트루스 라벨
    - mask_path: 마스크 파일 경로
    - pred_boxes: 예측된 바운딩 박스
    - pred_labels: 예측된 라벨
    - pred_masks: 예측된 마스크
    - volume_labels: 음식 양 라벨
    - label_dict: 클래스 라벨 매핑
    - idx: 이미지 인덱스
    """
    # 4개의 서브플롯 생성
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 이미지 정규화 해제를 위한 전처리
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    # 1. 원본 이미지
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. 원본 이미지 + 그라운드 트루스 박스
    axes[1].imshow(img_np)

    # 그라운드 트루스 박스 그리기 (녹색)
    for box, label in zip(gt_boxes.cpu().numpy(), gt_labels.cpu().numpy()):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='g', linewidth=2)
        axes[1].add_patch(rect)

        # 라벨명 찾기 (역매핑)
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')
        axes[1].text(xmin, ymin - 10, label_name,
                     color='green', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7))

    axes[1].set_title("Ground Truth Boxes")
    axes[1].axis('off')

    # 3. 원본 이미지 + 실제 마스크 파일
    axes[2].imshow(img_np)

    # 실제 마스크 파일 로드
    mask_array = np.load(mask_path)

    # 마스크가 이미지 크기와 다른 경우 리사이즈
    if mask_array.shape[:2] != (img_np.shape[0], img_np.shape[1]):
        mask_array = cv2.resize(mask_array, (img_np.shape[1], img_np.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    # 각 클래스별로 다른 색상 적용
    mask_viz = np.zeros_like(img_np)
    unique_labels = np.unique(mask_array)

    color_map = {
        1: [0, 1, 0],  # 첫 번째 클래스 (녹색)
        2: [0, 0, 1],  # 두 번째 클래스 (파란색)
        3: [1, 0, 0],  # 세 번째 클래스 (빨간색)
    }

    for label in unique_labels:
        if label > 0 and label in color_map:
            mask = (mask_array == label)
            for i in range(3):
                mask_viz[:, :, i] += mask * color_map[label][i] * 0.7

    axes[2].imshow(mask_viz, alpha=0.5)
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis('off')

    # 4. 원본 이미지 + 예측 결과
    axes[3].imshow(img_np)

    # 예측 박스 그리기 (빨간색)
    for j, (box, label) in enumerate(zip(pred_boxes.cpu().numpy(), pred_labels.cpu().numpy())):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='r', linewidth=2)
        axes[3].add_patch(rect)

        # 라벨명 찾기
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')

        # 마스크 그리기
        mask = pred_masks[j]
        mask_np = mask.squeeze().cpu().numpy() > 0.5

        # 마스크 오버레이 (보라색)
        mask_viz = np.zeros_like(img_np)
        mask_viz[:, :, 0] = mask_np * 0.5  # 빨간색 채널
        mask_viz[:, :, 2] = mask_np * 0.5  # 파란색 채널
        axes[3].imshow(mask_viz, alpha=0.6)

        # 음식 양 예측
        volume_class = calculate_volume_class(mask_np)
        vol_label = volume_labels[volume_class]

        # 박스 위에 텍스트 표시
        axes[3].text(xmin, ymin - 10, f"{label_name}\nVol: {vol_label}",
                     color='white', fontsize=10,
                     bbox=dict(facecolor='red', alpha=0.5))

    axes[3].set_title("Predictions")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def test_integrated_models(detection_model, volume_model, val_loader, device, label_dict):
    detection_model.eval()
    volume_model.eval()

    volume_labels = ['0-25%', '25-50%', '50-75%', '75-100%']

    # 클래스별 샘플 수집을 위한 딕셔너리
    class_samples = {class_id: [] for class_id in label_dict.values()}
    class_targets = {class_id: [] for class_id in label_dict.values()}

    # 클래스별로 보여줄 샘플 수
    samples_per_class = 4

    print("각 클래스별 시각화 샘플 수집 중...")

    # 1. 각 클래스별 샘플 수집
    with torch.no_grad():
        for images, targets in val_loader:
            for i, target in enumerate(targets):
                if len(target['labels']) > 0:
                    class_id = target['labels'][0].item()

                    # 해당 클래스의 샘플이 아직 충분하지 않으면 추가
                    if len(class_samples[class_id]) < samples_per_class:
                        class_samples[class_id].append(images[i])
                        class_targets[class_id].append(target)

            # 모든 클래스에 대해 충분한 샘플을 수집했는지 확인
            all_collected = True
            for class_id in label_dict.values():
                if len(class_samples[class_id]) < samples_per_class:
                    all_collected = False
                    break

            if all_collected:
                break

    # 2. 클래스 ID 순서대로 정렬 (1번 클래스부터 차례대로)
    sorted_classes = sorted([(name, id) for name, id in label_dict.items()], key=lambda x: x[1])

    # 3. 클래스별로 순차적으로 시각화
    for class_name, class_id in sorted_classes:
        samples = class_samples[class_id]
        sample_targets = class_targets[class_id]

        if len(samples) == 0:
            print(f"클래스 {class_id} ({class_name})에 대한 샘플이 없습니다.")
            continue

        print(f"\n===== 클래스 {class_id} ({class_name}) 시각화 중... =====")
        print(f"수집된 샘플 수: {len(samples)}")

        # 각 샘플에 대해 시각화
        for idx, (image, target) in enumerate(zip(samples, sample_targets)):
            # GPU로 이동
            image = image.to(device)

            # 객체 탐지 수행
            detection = detection_model([image])[0]

            # 예측 결과 필터링
            boxes = detection['boxes']
            scores = detection['scores']
            masks = detection['masks']
            labels = detection['labels']

            # 신뢰도 임계값 적용 (0.5)
            valid_idx = scores >= 0.5
            pred_boxes = boxes[valid_idx]
            pred_masks = masks[valid_idx]
            pred_labels = labels[valid_idx]

            if len(pred_boxes) > 0:
                # 그라운드 트루스 정보
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                mask_path = target['mask_path']

                # 시각화
                plt.figure(figsize=(20, 5))
                fig = visualize_comparison(
                    image,
                    gt_boxes,
                    gt_labels,
                    mask_path,
                    pred_boxes,
                    pred_labels,
                    pred_masks,
                    volume_labels,
                    label_dict,
                    idx=idx
                )
                plt.suptitle(f"클래스 {class_id} ({class_name}) - 샘플 {idx + 1}/{len(samples)}", fontsize=16)
                plt.tight_layout()
                plt.show()

        # 다음 클래스로 진행하기 전에 사용자 입력 대기
        if class_id < max(label_dict.values()):  # 마지막 클래스가 아니면
            input(f"\n클래스 {class_id} ({class_name}) 시각화 완료. 다음 클래스로 진행하려면 Enter 키를 누르세요...")


def visualize_comparison(image, gt_boxes, gt_labels, mask_path, pred_boxes, pred_labels, pred_masks, volume_labels,
                         label_dict, idx=0):
    """
    원본, 그라운드 트루스 박스, 실제 마스크 파일, 예측 결과를 서브플롯으로 시각화

    Parameters:
    - image: 원본 이미지 텐서
    - gt_boxes: 그라운드 트루스 바운딩 박스
    - gt_labels: 그라운드 트루스 라벨
    - mask_path: 마스크 파일 경로
    - pred_boxes: 예측된 바운딩 박스
    - pred_labels: 예측된 라벨
    - pred_masks: 예측된 마스크
    - volume_labels: 음식 양 라벨
    - label_dict: 클래스 라벨 매핑
    - idx: 이미지 인덱스
    """
    # 4개의 서브플롯 생성
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 이미지 정규화 해제를 위한 전처리
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    # 1. 원본 이미지
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. 원본 이미지 + 그라운드 트루스 박스
    axes[1].imshow(img_np)

    # 그라운드 트루스 박스 그리기 (녹색)
    for box, label in zip(gt_boxes.cpu().numpy(), gt_labels.cpu().numpy()):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='g', linewidth=2)
        axes[1].add_patch(rect)

        # 라벨명 찾기 (역매핑)
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')
        axes[1].text(xmin, ymin - 10, label_name,
                     color='green', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.7))

    axes[1].set_title("Ground Truth Boxes")
    axes[1].axis('off')

    # 3. 원본 이미지 + 실제 마스크 파일
    axes[2].imshow(img_np)

    # 실제 마스크 파일 로드
    mask_array = np.load(mask_path)

    # 마스크가 이미지 크기와 다른 경우 리사이즈
    if mask_array.shape[:2] != (img_np.shape[0], img_np.shape[1]):
        mask_array = cv2.resize(mask_array, (img_np.shape[1], img_np.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    # 각 클래스별로 다른 색상 적용
    mask_viz = np.zeros_like(img_np)
    unique_labels = np.unique(mask_array)

    color_map = {
        1: [0, 1, 0],  # 첫 번째 클래스 (녹색)
        2: [0, 0, 1],  # 두 번째 클래스 (파란색)
        3: [1, 0, 0],  # 세 번째 클래스 (빨간색)
    }

    for label in unique_labels:
        if label > 0 and label in color_map:
            mask = (mask_array == label)
            for i in range(3):
                mask_viz[:, :, i] += mask * color_map[label][i] * 0.7

    axes[2].imshow(mask_viz, alpha=0.5)
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis('off')

    # 4. 원본 이미지 + 예측 결과
    axes[3].imshow(img_np)

    # 예측 박스 그리기 (빨간색)
    for j, (box, label) in enumerate(zip(pred_boxes.cpu().numpy(), pred_labels.cpu().numpy())):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='r', linewidth=2)
        axes[3].add_patch(rect)

        # 라벨명 찾기
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')

        # 마스크 그리기
        mask = pred_masks[j]
        mask_np = mask.squeeze().cpu().numpy() > 0.5

        # 마스크 오버레이 (보라색)
        mask_viz = np.zeros_like(img_np)
        mask_viz[:, :, 0] = mask_np * 0.5  # 빨간색 채널
        mask_viz[:, :, 2] = mask_np * 0.5  # 파란색 채널
        axes[3].imshow(mask_viz, alpha=0.6)

        # 음식 양 예측
        volume_class = calculate_volume_class(mask_np)
        vol_label = volume_labels[volume_class]

        # 박스 위에 텍스트 표시
        axes[3].text(xmin, ymin - 10, f"{label_name}\nVol: {vol_label}",
                     color='white', fontsize=10,
                     bbox=dict(facecolor='red', alpha=0.5))

    axes[3].set_title("Predictions")
    axes[3].axis('off')

    plt.tight_layout()
    return fig  # 상위 함수에서 plt.show() 호출


def main():
    # GPU 성능 개선을 위한 추가 설정
    torch.backends.cudnn.benchmark = True

    # 디버깅 도구 설정: CUDA 에러를 더 명확하게 보기 위함
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 디렉토리 설정
    image_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\Training"
    xml_base_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\Training\[라벨]음식분류_TRAIN\xml"
    mask_base_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\Training\[라벨]음식분류_TRAIN\mask"

    # 타겟 크기 설정
    target_size = (224, 224)

    # 데이터셋 생성 (최적화된 파일 매칭 및 라벨 인덱스 수정)
    dataset = MultiClassFoodDetectionDataset(image_dir, xml_base_dir, mask_base_dir, target_size=target_size)

    num_classes = len(dataset.label_dict)
    print(f"탐지된 클래스 수: {num_classes}")
    print(f"클래스 매핑: {dataset.label_dict}")

    # 분할 및 데이터로더 생성
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 음식 탐지 모델 훈련
    print("\n----- 1단계: 음식 탐지 모델 훈련 -----")

    # 클래스 수 확인 메시지
    print(f"모델 초기화 전 클래스 수 확인: {num_classes}")

    # 모델 초기화 (클래스 수를 정확히 지정)
    detection_model = get_detection_model(num_classes=12).to(device)

    # 트레이닝 실행 전 확인 단계
    print("첫 번째 배치 확인...")
    images, targets = next(iter(train_loader))
    print(f"배치 내 이미지 수: {len(images)}")
    for i, target in enumerate(targets):
        print(f"이미지 {i}, 라벨: {target['labels']}, 범위: 0-{num_classes - 1}")

    # 모델 훈련
    train_detection_model(detection_model, train_loader, val_loader, device, num_epochs=1)

    # 2. 음식 양 분류 모델 훈련
    print("\n----- 2단계: 음식 양 분류 모델 훈련 -----")
    volume_model = VolumeClassificationModel(num_classes=4).to(device)
    train_volume_model(volume_model, train_loader, val_loader, device, num_epochs=1)

    print("모델 학습 완료!")

    # 3. 최종 모델 테스트 (클래스별 순차적 시각화)
    print("\n----- 3단계: 클래스별 순차적 시각화 -----")
    test_integrated_models(detection_model, volume_model, val_loader, device, dataset.label_dict)

if __name__ == "__main__":
    main()