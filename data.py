import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose, Normalize

from config import create_label_mapping, calculate_volume_class


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 이미지가 동일한 크기라고 가정 (모두 리사이즈됨)
    images_tensor = torch.stack(images)

    return images_tensor, targets

# def parse_annotation(xml_path, target_size=None, label_dict=None, mask_base_dir=None):
#     """
#     XML 파일에서 어노테이션 정보를 추출하고 마스크를 로드합니다.
#
#     Parameters:
#     - xml_path: XML 파일 경로
#     - target_size: 리사이징할 목표 크기 (선택적)
#     - label_dict: 클래스 라벨 매핑 딕셔너리
#     - mask_base_dir: 마스크 파일 기본 디렉토리
#
#     Returns:
#     - boxes: 바운딩 박스 텐서
#     - labels: 클래스 라벨 텐서
#     - masks: 마스크 텐서
#     - size: 원본 또는 리사이즈된 이미지 크기
#     """
#
#     try:
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#
#         # 이미지 파일명 추출 (마스크 파일 로드에 사용)
#         filename = root.find('filename').text
#         basename = os.path.splitext(filename)[0]
#
#         # 원본 이미지 크기 정보 추출
#         size_elem = root.find('size')
#         if size_elem is None:
#             orig_width = target_size[0] if target_size else 224
#             orig_height = target_size[1] if target_size else 224
#         else:
#             width_elem = size_elem.find('width')
#             height_elem = size_elem.find('height')
#
#             orig_width = int(width_elem.text) if width_elem is not None else (target_size[0] if target_size else 224)
#             orig_height = int(height_elem.text) if height_elem is not None else (target_size[1] if target_size else 224)
#
#         # 스케일링 계수 계산
#         width_scale = target_size[0] / orig_width if target_size else 1.0
#         height_scale = target_size[1] / orig_height if target_size else 1.0
#
#         boxes = []
#         labels = []
#         masks = []
#
#         # 마스크 파일 로드
#         if mask_base_dir:
#             mask_path = os.path.join(mask_base_dir, f"{basename}.npy")
#             if os.path.exists(mask_path):
#                 mask_array = np.load(mask_path)
#
#                 # 마스크 리사이즈
#                 if target_size and mask_array.shape[:2] != target_size[::-1]:
#                     mask_array = cv2.resize(mask_array, target_size, interpolation=cv2.INTER_NEAREST)
#             else:
#                 # 마스크 파일이 없는 경우 빈 마스크 생성
#                 mask_array = np.zeros((orig_height, orig_width), dtype=np.uint8)
#         else:
#             mask_array = np.zeros((orig_height, orig_width), dtype=np.uint8)
#
#         # 현재 이미지/XML의 클래스명 추출 (폴더명)
#         current_class_name = os.path.basename(os.path.dirname(xml_path))
#
#         for obj in root.findall('object'):
#             bndbox = obj.find('bndbox')
#             name = obj.find('name').text
#
#             # XML에 바운딩 박스 정보가 없는 경우 스킵
#             if bndbox is None:
#                 continue
#
#             # 각 좌표 요소를 안전하게 추출
#             xmin_elem = bndbox.find('xmin')
#             ymin_elem = bndbox.find('ymin')
#             xmax_elem = bndbox.find('xmax')
#             ymax_elem = bndbox.find('ymax')
#
#             # 요소가 없으면 스킵
#             if not all([xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
#                 continue
#
#             # 좌표 스케일링 및 변환
#             xmin = int(float(xmin_elem.text) * width_scale)
#             ymin = int(float(ymin_elem.text) * height_scale)
#             xmax = int(float(xmax_elem.text) * width_scale)
#             ymax = int(float(ymax_elem.text) * height_scale)
#
#             # 좌표 검증
#             if target_size:
#                 xmin = max(0, min(xmin, target_size[0] - 1))
#                 ymin = max(0, min(ymin, target_size[1] - 1))
#                 xmax = max(0, min(xmax, target_size[0] - 1))
#                 ymax = max(0, min(ymax, target_size[1] - 1))
#
#             # 바운딩 박스 추가
#             boxes.append([xmin, ymin, xmax, ymax])
#
#             # 라벨 매핑 로직
#             if label_dict:
#                 # 현재 클래스의 기본 라벨 찾기
#                 base_label = label_dict[current_class_name]
#
#                 # 현재 클래스의 마스크 정보 확인
#                 # 마스크에서 2와 3을 해당 클래스의 라벨로 간주
#                 mask = ((mask_array == 2) | (mask_array == 3)).astype(np.uint8)
#
#                 if mask[mask == 1].any():
#                     label = base_label
#                 else:
#                     label = 0  # 배경
#             else:
#                 label = 0
#
#             labels.append(label)
#
#             # 마스크 추출 (해당 라벨의 마스크만)
#             mask = ((mask_array == 2) | (mask_array == 3)).astype(np.uint8)
#             masks.append(mask)
#
#         # Tensor 변환
#         if len(boxes) > 0:
#             boxes = torch.tensor(boxes, dtype=torch.float32)
#             labels = torch.tensor(labels, dtype=torch.int64)
#             masks = torch.tensor(np.array(masks), dtype=torch.uint8)
#         else:
#             # 객체가 없는 경우 빈 텐서 생성
#             height = target_size[1] if target_size else orig_height
#             width = target_size[0] if target_size else orig_width
#             boxes = torch.empty((0, 4), dtype=torch.float32)
#             labels = torch.empty((0,), dtype=torch.int64)
#             masks = torch.empty((0, height, width), dtype=torch.uint8)
#
#         size = target_size if target_size else (orig_width, orig_height)
#         return boxes, labels, masks, size
#
#     except ET.ParseError:
#         print(f"XML 파일 파싱 오류: {xml_path}")
#         # 파싱 오류 시 빈 텐서 반환
#         height = target_size[1] if target_size else 224
#         width = target_size[0] if target_size else 224
#         return (
#             torch.empty((0, 4), dtype=torch.float32),
#             torch.empty((0,), dtype=torch.int64),
#             torch.empty((0, height, width), dtype=torch.uint8),
#             target_size or (width, height)
#         )

def parse_annotation(xml_path, target_size=None, label_dict=None, mask_base_dir=None):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # size 정보가 없는 경우 이미지에서 직접 크기 정보 가져오기
        size_elem = root.find('size')
        if size_elem is None:
            img_path = xml_path.replace('.xml', '.jpg')  # 또는 다른 이미지 확장자
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                height, width = img.shape[:2]
            else:
                print(f"Warning: No size information and image not found for {xml_path}")
                return None
        else:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        
        # XML 파일의 디렉토리 구조에서 클래스 이름 추출
        class_name = os.path.basename(os.path.dirname(xml_path))
        
        # 이미지 크기 정보 추출
        size_elem = root.find('size')
        if size_elem is None:
            print(f"Warning: No size information in XML file {xml_path}")
            return None  # size 태그가 없는 경우 None 반환
        
        # 이미지 파일명 추출
        filename = root.find('filename').text
        basename = os.path.splitext(filename)[0]

        # 원본 이미지 크기 정보 추출
        width_elem = size_elem.find('width')
        height_elem = size_elem.find('height')

        orig_width = int(width_elem.text) if width_elem is not None else (target_size[0] if target_size else 224)
        orig_height = int(height_elem.text) if height_elem is not None else (target_size[1] if target_size else 224)

        boxes = []
        labels = []
        masks = []

        # 마스크 파일 로드
        if mask_base_dir:
            mask_path = os.path.join(mask_base_dir, f"{basename}.npy")
            if os.path.exists(mask_path):
                mask_array = np.load(mask_path)

                # 마스크 리사이즈
                if target_size and mask_array.shape[:2] != target_size[::-1]:
                    mask_array = cv2.resize(mask_array, target_size, interpolation=cv2.INTER_NEAREST)
            else:
                # 마스크 파일이 없는 경우 빈 마스크 생성
                mask_array = np.zeros((orig_height, orig_width), dtype=np.uint8)
        else:
            mask_array = np.zeros((orig_height, orig_width), dtype=np.uint8)

        # XML 파일 내의 객체 모두 처리
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
                
            # name이 ElementTree.Element인 경우 text 속성 사용, 문자열인 경우 그대로 사용
            name_text = name_elem.text if hasattr(name_elem, 'text') else name_elem
            
            bndbox = obj.find('bndbox')

            # 바운딩 박스가 없으면 스킵
            if bndbox is None:
                continue

            # 각 좌표 요소를 더 안전하게 추출
            xmin_elem = bndbox.find('xmin')
            ymin_elem = bndbox.find('ymin')
            xmax_elem = bndbox.find('xmax')
            ymax_elem = bndbox.find('ymax')

            # 모든 좌표 요소가 존재하지 않으면 스킵
            if not all(elem is not None and elem.text is not None
                       for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                continue

            try:
                # 정수 좌표 추출
                xmin = int(xmin_elem.text)
                ymin = int(ymin_elem.text)
                xmax = int(xmax_elem.text)
                ymax = int(ymax_elem.text)

                # 바운딩 박스 유효성 검증
                if xmin >= xmax or ymin >= ymax:
                    print(f"Warning: Invalid bounding box in {xml_path}. Skipping this object.")
                    continue

                # 타겟 크기로 조정 (비율 유지)
                if target_size:
                    # 원본 이미지 대비 비율 계산
                    width_ratio = target_size[0] / orig_width
                    height_ratio = target_size[1] / orig_height

                    # 좌표 조정
                    xmin = int(xmin * width_ratio)
                    ymin = int(ymin * height_ratio)
                    xmax = int(xmax * width_ratio)
                    ymax = int(ymax * height_ratio)

                    # 좌표 검증
                    xmin = max(0, min(xmin, target_size[0] - 1))
                    ymin = max(0, min(ymin, target_size[1] - 1))
                    xmax = max(0, min(xmax, target_size[0] - 1))
                    ymax = max(0, min(ymax, target_size[1] - 1))

                    # 최소 박스 크기 보장
                    if xmax - xmin < 1:
                        xmax = xmin + 1
                    if ymax - ymin < 1:
                        ymax = ymin + 1

            except (ValueError, TypeError):
                continue

            # 바운딩 박스 추가
            boxes.append([xmin, ymin, xmax, ymax])

            # 라벨 매핑 로직 수정
            if name_text == 'dish':
                label = 1  # 접시
            elif label_dict:
                # 클래스 폴더의 라벨 사용 (예: '1.밥 일부' -> 라벨 2)
                label = label_dict.get(class_name, 0)

            labels.append(label)

            # 마스크 추출
            mask = np.zeros_like(mask_array)
            if label == 1:  # 접시
                mask = (mask_array == 1).astype(np.uint8)
            else:  # 음식
                mask = ((mask_array == 2) | (mask_array == 3)).astype(np.uint8)

            masks.append(mask)

        # Tensor 변환
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        else:
            # 객체가 없는 경우 빈 텐서 생성
            height = target_size[1] if target_size else orig_height
            width = target_size[0] if target_size else orig_width
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, height, width), dtype=torch.uint8)

        size = target_size if target_size else (orig_width, orig_height)
        return boxes, labels, masks, size

    except ET.ParseError:
        print(f"XML 파일 파싱 오류: {xml_path}")
        return None  # 파싱 오류 시 None 반환


class MultiClassFoodDetectionDataset(Dataset):
    """
    다중 클래스 음식 탐지 데이터셋 클래스
    """
    def __init__(self, image_base_dir, xml_base_dir, mask_base_dir, transform=None, target_size=(224, 224)):
        # 동적 라벨 매핑 생성
        self.label_dict = create_label_mapping(image_base_dir)
        print("Generated label dictionary:", self.label_dict)
        
        self.image_files = []
        self.xml_files = []
        self.mask_files = []
        self.class_labels = []  # 각 데이터의 클래스 라벨 저장
        self.transform = transform or self._get_transform(target_size)
        self.target_size = target_size

        # 클래스별 파일 세트 카운트 (라벨 1 제외)
        class_counts = {class_name: 0 for class_name in self.label_dict.keys() if self.label_dict[class_name] != 1}

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

        # 3. 이미지 파일 검색 및 매칭 (라벨 1 제외)
        print("이미지 파일 수집 및 매칭 중...")
        for class_name in self.label_dict.keys():
            # 라벨 1(dish)은 완전히 건너뛰기
            if self.label_dict[class_name] == 1:
                continue

            class_dir = os.path.join(image_base_dir, class_name)

            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        basename = os.path.splitext(file)[0]
                        img_path = os.path.join(root, file)

                        # 인덱싱된 사전에서 해당 파일명으로 XML과 마스크 찾기
                        xml_path = xml_files_dict.get(basename)
                        mask_path = mask_files_dict.get(basename)

                        # XML 파일 검증
                        if xml_path:
                            mask_base_dir = os.path.dirname(mask_path) if mask_path else None
                            result = parse_annotation(xml_path, self.target_size, self.label_dict, mask_base_dir=mask_base_dir)
                            if result is None:
                                print(f"Warning: Skipping {basename} due to missing size information.")
                                continue

                        # 모든 파일이 존재하는 경우 데이터셋에 추가
                        if xml_path and mask_path:
                            self.image_files.append(img_path)
                            self.xml_files.append(xml_path)
                            self.mask_files.append(mask_path)

                            # 클래스 라벨 저장
                            self.class_labels.append(self.label_dict[class_name])

                            # 클래스별 카운트 증가
                            class_counts[class_name] += 1

        # data.py 파일의 마지막 부분 수정
        # 클래스별 데이터 수 출력
        print("\n--- 클래스별 데이터 세트 통계 ---")
        for class_name, label in self.label_dict.items():
            count = sum(1 for f in self.xml_files if class_name in f)
            print(f"클래스 '{class_name}' (라벨: {label}): {count}개 세트")

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

        # XML 파싱 시 마스크 기본 디렉토리 전달
        mask_base_dir = os.path.dirname(mask_path)
        result = parse_annotation(
            xml_path,
            self.target_size,
            label_dict=self.label_dict,
            mask_base_dir=mask_base_dir
        )

        if result is None:
            # size 태그가 없는 경우, 다음 항목으로 건너뛰기
            return self.__getitem__((idx + 1) % len(self))
        
        boxes, labels, masks, _ = result
        
        # 음식 양 클래스 계산
        mask_array = np.load(mask_path)

        # 마스크 리사이즈
        if mask_array.shape[:2] != self.target_size[::-1]:
            mask_array = cv2.resize(mask_array, self.target_size, interpolation=cv2.INTER_NEAREST)

        volume_class = calculate_volume_class(mask_array)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "volume_class": torch.tensor([volume_class], dtype=torch.long),
            "mask_path": mask_path
        }
        return image_tensor, target

    def custom_collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # 이미지가 동일한 크기라고 가정 (모두 리사이즈됨)
        images_tensor = torch.stack(images)

        return images_tensor, targets

def validate_dataset(dataset):
    class_counts = {}
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        for label in target['labels']:
            class_counts[label.item()] = class_counts.get(label.item(), 0) + 1
    
    print("\n클래스별 샘플 수:")
    for label, count in class_counts.items():
        print(f"클래스 {label}: {count}개 샘플")
    
    return class_counts


def visualize_comparison(image, gt_boxes, gt_labels, mask_path, pred_boxes, pred_labels, pred_masks, volume_labels,
                         label_dict):
    """
    원본 이미지, 바운딩 박스, 마스크, 예측 결과를 시각화합니다.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from PIL import Image

    # 이미지 변환
    img = image.cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    # 클래스별 색상 정의
    class_colors = {
        'dish': {'mask': np.array([0, 0, 255]), 'bbox': (255, 0, 0)},  # 파랑(마스크), 빨강(박스)
        'default': {'mask': np.array([0, 255, 0]), 'bbox': (0, 255, 0)}  # 초록(마스크&박스)
    }

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()

    # 1. 원본 이미지
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 2. 바운딩 박스 시각화
    image_with_bbox = img.copy()
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        class_name = next((k for k, v in label_dict.items() if v == label.item()), "Unknown")
        color = class_colors.get(class_name, class_colors['default'])['bbox']
        cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_with_bbox, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    axes[1].imshow(image_with_bbox)
    axes[1].set_title('Ground Truth Boxes')
    axes[1].axis('off')

    # 3. 마스크 시각화
    mask_display = np.zeros_like(img)
    if mask_path:
        try:
            mask_array = np.load(mask_path)
            mask_resized = cv2.resize(mask_array, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 마스크 값에 따른 색상 매핑
            for class_id in np.unique(mask_resized):
                if class_id > 0:
                    class_name = next((k for k, v in label_dict.items() if v == class_id), "Unknown")
                    color = class_colors.get(class_name, class_colors['default'])['mask']
                    mask_display[mask_resized == class_id] = color
        except Exception as e:
            print(f"마스크 로드 실패: {e}")
    
    axes[2].imshow(mask_display)
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')

    # 4. 예측 결과 시각화 (마스크 오버레이)
    blended = cv2.addWeighted(img, 0.7, mask_display, 0.3, 0)
    
    # 예측된 바운딩 박스 추가
    for box, label in zip(pred_boxes, pred_labels):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        class_name = next((k for k, v in label_dict.items() if v == label.item()), "Unknown")
        color = class_colors.get(class_name, class_colors['default'])['bbox']
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
        cv2.putText(blended, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    axes[3].imshow(blended)
    axes[3].set_title('Prediction Results (Mask Overlay + Boxes)')
    axes[3].axis('off')

    plt.tight_layout()
    return fig

def test_integrated_models(detection_model, volume_model, val_loader, device, label_dict, dataset, val_indices):
    detection_model.eval()
    volume_model.eval()
    
    # 클래스별 시각화를 위한 샘플 수집
    class_samples = {}
    
    print("\n----- 클래스별 순차적 시각화 -----")
    print("각 클래스별 시각화 샘플 수집 중...")
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            
            # 탐지 모델 예측
            predictions = detection_model(images)
            
            # 각 이미지에 대해 처리
            for j, (image, target, prediction) in enumerate(zip(images, targets, predictions)):
                # 클래스 라벨 가져오기
                if 'labels' in target and len(target['labels']) > 0:
                    label = target['labels'][0].item()
                    
                    # 해당 클래스의 샘플이 아직 수집되지 않았다면
                    if label not in class_samples and label in label_dict.values():
                        class_name = next((k for k, v in label_dict.items() if v == label), None)
                        if class_name:
                            print(f"\n===== 클래스 {label} ({class_name}) 시각화 중... =====")
                            
                            # 신뢰도가 0.1 이상인 객체 수 계산
                            high_conf_objects = sum(1 for score in prediction['scores'] if score > 0.1)
                            print(f"샘플 {len(class_samples) + 1}: 신뢰도 > 0.1인 객체 수: {high_conf_objects}")
                            
                            # 시각화
                            fig = visualize_comparison(
                                image,
                                target['boxes'],
                                target['labels'],
                                target['mask_path'],
                                prediction['boxes'],
                                prediction['labels'],
                                prediction['masks'],
                                target['volume_class'],
                                label_dict
                            )
                            
                            # 시각화 결과 저장
                            class_samples[label] = fig
                            
                            # 최대 4개 샘플만 수집
                            if len(class_samples) >= 4:
                                break
            
            # 모든 클래스의 샘플을 수집했으면 중단
            if len(class_samples) >= 4:
                break
    
    print("\n첫 번째 시각화 완료!")
    
    # 두 번째 시각화: 랜덤 이미지에서 음식 종류와 양 예측
    print("\n----- 랜덤 이미지에서 음식 종류와 양 예측 시각화 -----")
    
    # 지정된 경로에서 이미지 파일 목록 가져오기
    random_image_dir = r"C:\Users\furim\Desktop\Bumin_dataset\202306\전영숙_F76_45"
    image_files = [f for f in os.listdir(random_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 랜덤으로 4개 이미지 선택
    import random
    selected_images = random.sample(image_files, min(4, len(image_files)))
    
    # 이미지 변환을 위한 transform 정의
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 음식 양 클래스 이름 정의
    volume_classes = ["1/4", "1/2", "3/4", "1"]
    
    # 시각화를 위한 figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, img_file in enumerate(selected_images):
        # 이미지 로드 및 변환
        img_path = os.path.join(random_image_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 탐지 모델 예측
        with torch.no_grad():
            detection_output = detection_model(img_tensor)
            
            # 음식 양 분류 모델 예측
            # 탐지된 객체가 있는 경우에만 양 분류 수행
            if len(detection_output[0]['boxes']) > 0:
                # 가장 높은 신뢰도를 가진 객체 선택
                max_score_idx = torch.argmax(detection_output[0]['scores'])
                box = detection_output[0]['boxes'][max_score_idx]
                label = detection_output[0]['labels'][max_score_idx].item()
                
                # 바운딩 박스 영역 추출
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_tensor.shape[3], x2), min(img_tensor.shape[2], y2)
                
                # 객체 영역 추출
                obj_region = img_tensor[:, :, y1:y2, x1:x2]
                
                # 객체 영역이 너무 작으면 전체 이미지 사용
                if obj_region.shape[2] < 10 or obj_region.shape[3] < 10:
                    obj_region = img_tensor
                
                # 음식 양 분류
                volume_output = volume_model(obj_region)
                volume_class = torch.argmax(volume_output).item()
            else:
                # 탐지된 객체가 없는 경우
                label = 0  # 배경
                volume_class = 0  # 기본값
                x1, y1, x2, y2 = 0, 0, img_tensor.shape[3], img_tensor.shape[2]
        
        # 원본 이미지 표시
        axes[idx].imshow(img)
        
        # 클래스 이름 가져오기
        class_name = next((k for k, v in label_dict.items() if v == label), "배경")
        
        # 결과 표시
        axes[idx].set_title(f"음식: {class_name}\n양: {volume_classes[volume_class]}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n두 번째 시각화 완료!")