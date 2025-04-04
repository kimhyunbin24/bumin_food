import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from config import clear_memory


def get_detection_model(num_classes):
    """
    객체 탐지 및 세그멘테이션용 Mask R-CNN 모델 생성

    Args:
        num_classes: 분류할 클래스 수

    Returns:
        model: Mask R-CNN 모델
    """
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
    """
    음식 양 분류를 위한 모델
    """

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


def load_pretrained_models(num_classes, device):
    """
    저장된 모델 가중치 로드 함수

    Args:
        num_classes: 분류할 클래스 수
        device: 모델을 로드할 장치 (CPU/GPU)

    Returns:
        tuple: (detection_model, volume_model) 로드된 모델 두 개
    """
    # 음식 탐지 모델 로드
    detection_model = get_detection_model(num_classes=num_classes).to(device)
    try:
        detection_model.load_state_dict(torch.load('food_detection_model_with_masks.pth'))
        print("음식 탐지 모델 가중치 성공적으로 로드")
    except Exception as e:
        print(f"음식 탐지 모델 가중치 로드 실패: {e}")
        return None, None

    # 음식 양 분류 모델 로드
    volume_model = VolumeClassificationModel(num_classes=4).to(device)
    try:
        volume_model.load_state_dict(torch.load('food_volume_classification_model.pth'))
        print("음식 양 분류 모델 가중치 성공적으로 로드")
    except Exception as e:
        print(f"음식 양 분류 모델 가중치 로드 실패: {e}")
        return None, None

    return detection_model, volume_model


def train_detection_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    객체 탐지 모델 훈련 함수

    Args:
        model: 훈련할 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        device: 훈련에 사용할 장치 (CPU/GPU)
        num_epochs: 훈련 에폭 수
    """
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
    """
    음식 양 분류 모델 훈련 함수

    Args:
        model: 훈련할 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        device: 훈련에 사용할 장치 (CPU/GPU)
        num_epochs: 훈련 에폭 수
    """
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