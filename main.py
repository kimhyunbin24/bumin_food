import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2

# 커스텀 모듈 임포트
from config import (
    IMAGE_DIR, XML_BASE_DIR, MASK_BASE_DIR, TARGET_SIZE, VOLUME_LABELS,
    choose_model_mode, enable_memory_growth, clear_memory
)
from data import MultiClassFoodDetectionDataset, custom_collate_fn
from models import (
    get_detection_model, VolumeClassificationModel, load_pretrained_models,
    train_detection_model, train_volume_model
)


def visualize_comparison(image, gt_boxes, gt_labels, mask_path, pred_boxes, pred_labels, pred_masks, volume_labels,
                         label_dict, idx=0):
    """
    원본, 그라운드 트루스 박스, 실제 마스크 파일, 예측 결과를 서브플롯으로 시각화
    """
    # 4개의 서브플롯 생성
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 이미지 정규화 해제를 위한 전처리
    img_np = image.cpu().detach().numpy().transpose(1, 2, 0)
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    # 색상 매핑
    class_colors = {
        'dish': {'bbox': (255, 0, 0), 'mask': (0, 0, 255)},  # 파랑(마스크), 빨강(박스)
        'food': {'bbox': (0, 255, 0), 'mask': (0, 255, 0)},  # 초록(마스크&박스)
        'food_alt': {'bbox': (0, 200, 100), 'mask': (0, 200, 100)},  # 연두색(마스크&박스)
        'default': {'bbox': (255, 255, 0), 'mask': (128, 128, 128)}  # 노랑(박스), 회색(마스크)
    }

    # 1. 원본 이미지
    axes[0].imshow(img_np)
    axes[0].set_title("원본 이미지")
    axes[0].axis('off')

    # 2. 원본 이미지 + 바운딩 박스 (그라운드 트루스)
    axes[1].imshow(img_np)

    # 바운딩 박스 그리기
    for box, label in zip(gt_boxes.cpu().detach().numpy(), gt_labels.cpu().detach().numpy()):
        xmin, ymin, xmax, ymax = map(int, box)

        # 라벨명 찾기
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')

        # 박스 색상 선택
        color = class_colors.get(label_name, class_colors['default'])['bbox']
        color_normalized = tuple(c / 255 for c in color)

        # 박스 그리기
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor=color_normalized, linewidth=2)
        axes[1].add_patch(rect)

        # 라벨 텍스트 추가
        axes[1].text(xmin, ymin - 10, label_name,
                     color='white', fontsize=10,
                     bbox=dict(facecolor=color_normalized, alpha=0.7))

    axes[1].set_title("바운딩 박스 시각화")
    axes[1].axis('off')

    # 3. 마스크 시각화
    try:
        # 실제 마스크 파일 로드
        mask_array = np.load(mask_path)

        # 마스크가 이미지 크기와 다른 경우 리사이즈
        if mask_array.shape[:2] != (img_np.shape[0], img_np.shape[1]):
            mask_array = cv2.resize(mask_array, (img_np.shape[1], img_np.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        # 마스크 시각화를 위한 배열 준비
        mask_display = np.zeros_like(img_np)

        # 음식 라벨을 우선순위로 시각화
        unique_vals = np.unique(mask_array)
        unique_vals = unique_vals[unique_vals > 0]
        food_first = sorted(unique_vals, reverse=True)

        for class_id in food_first:
            if class_id > 0:
                label_name = next((k for k, v in label_dict.items() if v == class_id), 'Unknown')
                color = class_colors.get(label_name, class_colors['default'])['mask']
                color_normalized = color / 255.0
                mask_display[mask_array == class_id] = color_normalized

        axes[2].imshow(mask_display)
        axes[2].set_title("마스크만 시각화 (음식 우선)")
        axes[2].axis('off')

    except Exception as e:
        print(f"마스크 로드 중 오류 발생: {e}")
        axes[2].text(0.5, 0.5, "마스크 로드 실패",
                     ha='center', va='center', transform=axes[2].transAxes,
                     color='red', fontsize=12)

    # 4. 원본 이미지 + 예측 결과
    axes[3].imshow(img_np)

    # 예측 박스 그리기
    for j, (box, label) in enumerate(zip(pred_boxes.cpu().detach().numpy(), pred_labels.cpu().detach().numpy())):
        xmin, ymin, xmax, ymax = map(int, box)

        # 라벨명 찾기
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')

        # 박스 색상 선택
        color = class_colors.get(label_name, class_colors['default'])['bbox']
        color_normalized = tuple(c / 255 for c in color)

        # 박스 그리기
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor=color_normalized, linewidth=2)
        axes[3].add_patch(rect)

        # 마스크 그리기
        mask = pred_masks[j]
        mask_np = mask.squeeze().cpu().detach().numpy() > 0.5

        # 마스크 오버레이
        mask_viz = np.zeros_like(img_np)
        mask_viz[mask_np] = [0.5, 0, 0.5]  # 보라색 마스크
        axes[3].imshow(mask_viz, alpha=0.6)

        # 음식 양 예측
        from config import calculate_volume_class
        volume_class = calculate_volume_class(mask_np)
        vol_label = volume_labels[volume_class]

        # 박스 위에 텍스트 표시
        axes[3].text(xmin, ymin - 10, f"{label_name}\nVol: {vol_label}",
                     color='white', fontsize=10,
                     bbox=dict(facecolor=color_normalized, alpha=0.5))

    axes[3].set_title("Predictions")
    axes[3].axis('off')

    plt.tight_layout()
    return fig  # 상위 함수에서 plt.show() 호출


def test_integrated_models(detection_model, volume_model, val_loader, device, label_dict, dataset, val_indices):
    """
    통합 모델 테스트 및 시각화 함수

    Args:
        detection_model: 음식 탐지 모델
        volume_model: 음식 양 분류 모델
        val_loader: 검증 데이터 로더
        device: 테스트에 사용할 장치 (CPU/GPU)
        label_dict: 클래스 라벨 매핑 딕셔너리
        dataset: 전체 데이터셋
        val_indices: 검증 데이터셋 인덱스
    """
    import cv2

    detection_model.eval()
    volume_model.eval()

    # 클래스별 샘플 수집을 위한 딕셔너리
    class_samples = {}
    class_targets = {}

    # 클래스별로 보여줄 샘플 수
    samples_per_class = 4

    print("각 클래스별 시각화 샘플 수집 중...")

    # 1. 각 클래스별 샘플 수집
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            for i, target in enumerate(targets):
                # 현재 배치의 인덱스를 전체 val_indices 인덱스로 변환
                dataset_idx = batch_idx * val_loader.batch_size + i
                if dataset_idx >= len(val_indices):  # 마지막 배치 처리
                    continue

                # 원본 데이터셋의 클래스 라벨 사용
                class_id = dataset.class_labels[val_indices[dataset_idx]]

                # 딕셔너리 초기화 (없는 경우)
                if class_id not in class_samples:
                    class_samples[class_id] = []
                    class_targets[class_id] = []

                # 해당 클래스의 샘플이 아직 충분하지 않으면 추가
                if len(class_samples[class_id]) < samples_per_class:
                    class_samples[class_id].append(images[i])
                    class_targets[class_id].append(target)

    # 2. 클래스별로 순차적으로 시각화
    for class_id in sorted(class_samples.keys()):
        samples = class_samples[class_id]
        sample_targets = class_targets[class_id]

        if len(samples) == 0:
            print(f"클래스 {class_id}에 대한 샘플이 없습니다.")
            continue

        class_name = next((name for name, idx in label_dict.items() if idx == class_id), "Unknown")
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

            # 디버깅: 탐지된 객체 정보 출력
            print(f"샘플 {idx + 1}: 신뢰도 > 0.1인 객체 수: {torch.sum(scores > 0.1).item()}")

            # 신뢰도 임계값 적용 (더 낮은 임계값 0.1 사용)
            valid_idx = scores >= 0.1
            pred_boxes = boxes[valid_idx]
            pred_masks = masks[valid_idx]
            pred_labels = labels[valid_idx]

            # 그라운드 트루스 정보
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            mask_path = target['mask_path']

            # 시각화 수행
            fig = visualize_comparison(
                image,
                gt_boxes,
                gt_labels,
                mask_path,
                pred_boxes,
                pred_labels,
                pred_masks,
                VOLUME_LABELS,
                label_dict,
                idx=idx
            )

            plt.suptitle(f"클래스 {class_id} ({class_name}) - 샘플 {idx + 1}/{len(samples)}", fontsize=16)
            plt.tight_layout()

            # 사용자가 창을 닫을 때까지 대기
            print(f"창을 닫으면 다음 샘플로 넘어갑니다. (샘플 {idx + 1}/{len(samples)})")
            plt.show(block=True)  # block=True로 설정하여 사용자가 창을 닫을 때까지 대기

        print(f"클래스 {class_id} ({class_name}) 시각화 완료")

    print("\n모든 클래스 시각화 완료")


def main():
    """
    메인 실행 함수
    """
    # GPU 성능 개선을 위한 추가 설정
    torch.backends.cudnn.benchmark = True

    # 디버깅 도구 설정: CUDA 에러를 더 명확하게 보기 위함
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 메모리 동적 할당 활성화
    enable_memory_growth()

    # 데이터셋 생성 (최적화된 파일 매칭 및 라벨 인덱스 수정)
    dataset = MultiClassFoodDetectionDataset(IMAGE_DIR, XML_BASE_DIR, MASK_BASE_DIR, target_size=TARGET_SIZE)

    print("\n데이터셋 상세 정보:")
    for idx in range(min(5, len(dataset))):  # 처음 5개 항목만 출력
        img, target = dataset[idx]
        print(f"Dataset Item {idx}:")
        print(f"  Boxes: {target['boxes']}")
        print(f"  Labels: {target['labels']}")
        print(f"  Masks Shape: {target['masks'].shape}")
        print(f"  Volume Class: {target['volume_class']}")

    num_classes = len(dataset.label_dict)
    print(f"탐지된 클래스 수: {num_classes}")
    print(f"클래스 매핑: {dataset.label_dict}")

    # Stratified sampling을 사용하여 데이터셋 분할
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=dataset.class_labels  # 클래스 라벨을 기준으로 계층화 sampling
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    val_class_labels = [dataset.class_labels[i] for i in val_indices]
    val_class_counts = Counter(val_class_labels)
    print("\n검증 데이터셋 클래스별 개수:")
    for class_id, count in val_class_counts.items():
        class_name = next((name for name, idx in dataset.label_dict.items() if idx == class_id), "Unknown")
        print(f"클래스 {class_id} ({class_name}): {count}개")

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

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 모드 선택
    mode_choice = choose_model_mode()

    if mode_choice == 1:
        # 저장된 가중치 로드
        detection_model, volume_model = load_pretrained_models(num_classes, device)

        if detection_model is None or volume_model is None:
            print("모델 가중치 로드 실패. 새로운 모델을 학습합니다.")
            # 모델 초기화 (클래스 수를 정확히 지정)
            detection_model = get_detection_model(num_classes=num_classes).to(device)
            volume_model = VolumeClassificationModel(num_classes=4).to(device)
    else:
        # 1. 음식 탐지 모델 훈련
        print("\n----- 1단계: 음식 탐지 모델 훈련 -----")
        detection_model = get_detection_model(num_classes=num_classes).to(device)
        train_detection_model(detection_model, train_loader, val_loader, device, num_epochs=1)

        # 2. 음식 양 분류 모델 훈련
        print("\n----- 2단계: 음식 양 분류 모델 훈련 -----")
        volume_model = VolumeClassificationModel(num_classes=4).to(device)
        train_volume_model(volume_model, train_loader, val_loader, device, num_epochs=1)

    print("모델 준비 완료!")

    # 3. 최종 모델 테스트 (클래스별 순차적 시각화)
    print("\n----- 클래스별 순차적 시각화 -----")
    test_integrated_models(detection_model, volume_model, val_loader, device, dataset.label_dict, dataset, val_indices)


if __name__ == "__main__":
    main()