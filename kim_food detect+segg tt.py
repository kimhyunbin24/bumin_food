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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 사용 가능한 모든 GPU 사용
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.3, device=0)


def choose_model_mode():
    """
    모델 실행 모드 선택 함수
    """
    print("\n모델 실행 모드를 선택해주세요:")
    print("1. 저장된 가중치로 모델 로드")
    print("2. 새로운 모델 학습")

    while True:
        try:
            choice = input("번호를 입력하세요 (1/2): ").strip()
            if choice not in ['1', '2']:
                raise ValueError
            return int(choice)
        except ValueError:
            print("올바른 번호를 입력해주세요 (1 또는 2).")


def load_pretrained_models(num_classes, device):
    """
    저장된 모델 가중치 로드 함수
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


# def create_label_mapping(image_base_dir):
#     # 나머지 클래스 폴더 수집
#     class_folders = [
#         d for d in os.listdir(image_base_dir)
#         if os.path.isdir(os.path.join(image_base_dir, d)) and d[0].isdigit()
#     ]
#
#     # 폴더명 기준으로 정렬
#     class_folders.sort(key=lambda x: int(x.split('.')[0]))
#
#     # 라벨 매핑 생성 (2부터 시작)
#     label_dict = {name: idx + 2 for idx, name in enumerate(class_folders)}
#
#     return label_dict

def create_label_mapping(image_base_dir):
    """
    이미지 기본 디렉토리의 클래스 폴더를 기반으로 동적 라벨 매핑 생성

    Args:
        image_base_dir (str): 이미지 기본 디렉토리 경로

    Returns:
        dict: 클래스명과 해당 라벨의 매핑 딕셔너리
    """
    # 숫자로 시작하는 폴더만 수집
    class_folders = [
        d for d in os.listdir(image_base_dir)
        if os.path.isdir(os.path.join(image_base_dir, d)) and d[0].isdigit()
    ]

    # 폴더명 기준으로 정렬
    class_folders.sort(key=lambda x: int(x.split('.')[0]))

    # 라벨 매핑 생성 (2부터 시작)
    label_dict = {name: idx + 2 for idx, name in enumerate(class_folders)}

    # 접시는 고정적으로 1
    label_dict['dish'] = 1

    return label_dict


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
#             # 라벨 매핑
#             if label_dict:
#                 label = label_dict.get(name, 0)
#             else:
#                 label = 0
#             labels.append(label)
#
#             # 마스크 추출 (해당 라벨의 마스크만)
#             mask = (mask_array == label).astype(np.uint8)
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
    """
    XML 파일에서 어노테이션 정보를 추출하고 마스크를 로드합니다.

    Parameters:
    - xml_path: XML 파일 경로
    - target_size: 리사이징할 목표 크기 (선택적)
    - label_dict: 클래스 라벨 매핑 딕셔너리
    - mask_base_dir: 마스크 파일 기본 디렉토리

    Returns:
    - boxes: 바운딩 박스 텐서
    - labels: 클래스 라벨 텐서
    - masks: 마스크 텐서
    - size: 원본 또는 리사이즈된 이미지 크기
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 이미지 파일명 추출 (마스크 파일 로드에 사용)
        filename = root.find('filename').text
        basename = os.path.splitext(filename)[0]

        # 원본 이미지 크기 정보 추출
        size_elem = root.find('size')
        if size_elem is None:
            orig_width = target_size[0] if target_size else 224
            orig_height = target_size[1] if target_size else 224
        else:
            width_elem = size_elem.find('width')
            height_elem = size_elem.find('height')

            orig_width = int(width_elem.text) if width_elem is not None else (target_size[0] if target_size else 224)
            orig_height = int(height_elem.text) if height_elem is not None else (target_size[1] if target_size else 224)

        # 스케일링 계수 계산
        width_scale = target_size[0] / orig_width if target_size else 1.0
        height_scale = target_size[1] / orig_height if target_size else 1.0

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

        # 현재 이미지/XML의 클래스명 추출 (폴더명)
        current_class_name = os.path.basename(os.path.dirname(xml_path))

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            name = obj.find('name').text

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

            # 좌표 스케일링 및 변환
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

            # 바운딩 박스 추가
            boxes.append([xmin, ymin, xmax, ymax])

            # 라벨 매핑 로직
            if label_dict:
                # 현재 클래스의 기본 라벨 찾기
                base_label = label_dict[current_class_name]

                # 현재 클래스의 마스크 정보 확인
                # 마스크에서 2와 3을 해당 클래스의 라벨로 간주
                mask = ((mask_array == 2) | (mask_array == 3)).astype(np.uint8)

                if mask[mask == 1].any():
                    label = base_label
                else:
                    label = 0  # 배경
            else:
                label = 0

            labels.append(label)

            # 마스크 추출 (해당 라벨의 마스크만)
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
    모든 음식 라벨(2 이상)을 고려하여 계산합니다.
    """
    # 마스크가 3차원이면 첫 번째 차원 제거
    if mask_array.ndim > 2:
        mask_array = mask_array[0]  # 첫 번째 채널만 사용
    # 접시 마스크 (라벨 1)
    dish_mask = (mask_array == 1).astype(np.uint8)
    # 모든 음식 마스크 (라벨 2 이상)
    food_mask = (mask_array >= 2).astype(np.uint8)
    # 음식과 접시가 모두 존재하는 경우만 계산
    if np.sum(food_mask) > 0 and np.sum(dish_mask) > 0:
        # 전체 음식 픽셀 수 / 접시 픽셀 수 = 비율
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
        # 접시가 없는 경우: 전체 이미지 대비 음식 영역 비율로 판단
        if mask_array.ndim == 2:
            # 전체 음식 픽셀 비율 계산
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            food_pixels = np.sum(mask_array >= 2)  # 라벨 2 이상인 모든 픽셀
            pixel_ratio = food_pixels / total_pixels

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

    # 디버깅을 위한 정보 출력 (선택적)
    print(f"음식 픽셀 수: {food_pixel_count}")
    print(f"접시 픽셀 수: {dish_pixel_count}")
    print(f"비율: {ratio:.2f}")
    unique_labels = np.unique(mask_array)
    print(f"발견된 라벨: {unique_labels}")


class MultiClassFoodDetectionDataset(Dataset):
    def __init__(self, image_base_dir, xml_base_dir, mask_base_dir, transform=None, target_size=(224, 224)):
        # 클래스 라벨 매핑
        self.label_dict = create_label_mapping(image_base_dir)
        self.label_dict['dish'] = 1

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

        # 디버그용 추가 코드
        print("Image Dir Contents:")
        for class_name in self.label_dict.keys():
            # 라벨 1(dish)은 완전히 건너뛰기
            if self.label_dict[class_name] == 1:
                continue

            class_dir = os.path.join(image_base_dir, class_name)
            print(f"{class_name}: {len(os.listdir(class_dir))} 파일")

        # 디버그용 추가 코드
        print("Image Dir Contents:")
        for class_name in [name for name in self.label_dict.keys() if self.label_dict[name] != 1]:
            class_dir = os.path.join(image_base_dir, class_name)
            if os.path.exists(class_dir):
                print(f"{class_name}: {len(os.listdir(class_dir))} 파일")

        print("Image Dir Contents:")
        for class_name in [name for name in self.label_dict.keys() if name != 'dish']:
            class_dir = os.path.join(image_base_dir, class_name)
            print(f"{class_name}: {len(os.listdir(class_dir))} 파일")

        print("Image Dir Contents:")
        for class_name in self.label_dict.keys():
            # 라벨 1(dish)은 건너뛰기
            if self.label_dict[class_name] == 1:
                continue

            class_dir = os.path.join(image_base_dir, class_name)
            print(f"{class_name}: {len(os.listdir(class_dir))} 파일")

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
        boxes, labels, masks, _ = parse_annotation(
            xml_path,
            self.target_size,
            label_dict=self.label_dict,
            mask_base_dir=mask_base_dir
        )

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
    detection_model.eval()
    volume_model.eval()

    volume_labels = ['0-25%', '25-50%', '50-75%', '75-100%']

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
                volume_labels,
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

    # 1. 원본 이미지
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. 원본 이미지 + 그라운드 트루스 박스
    axes[1].imshow(img_np)

    # 그라운드 트루스 박스 그리기 (녹색)
    for box, label in zip(gt_boxes.cpu().detach().numpy(), gt_labels.cpu().detach().numpy()):
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

    try:
        # 실제 마스크 파일 로드
        mask_array = np.load(mask_path)

        # 마스크가 이미지 크기와 다른 경우 리사이즈
        if mask_array.shape[:2] != (img_np.shape[0], img_np.shape[1]):
            mask_array = cv2.resize(mask_array, (img_np.shape[1], img_np.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        # 유니크한 클래스 라벨 추출
        unique_labels = np.unique(mask_array)

        # 클래스 개수에 따라 컬러맵 생성
        num_classes = len(unique_labels)
        color_map = plt.cm.get_cmap('tab20', num_classes)

        # 각 클래스별로 다른 색상 적용
        mask_viz = np.zeros_like(img_np)
        for i, label in enumerate(unique_labels):
            if label > 0:
                mask = (mask_array == label)
                color = color_map(i)[:3]  # RGB 값 추출
                for j in range(3):
                    mask_viz[:, :, j] += mask * color[j] * 0.7

        axes[2].imshow(mask_viz, alpha=0.5)
    except Exception as e:
        print(f"마스크 로드 중 오류 발생: {e}")
        axes[2].text(0.5, 0.5, "마스크 로드 실패",
                     ha='center', va='center', transform=axes[2].transAxes,
                     color='red', fontsize=12)

    axes[2].set_title("Ground Truth Mask")
    axes[2].axis('off')

    # 4. 원본 이미지 + 예측 결과
    axes[3].imshow(img_np)

    # 예측 박스 그리기 (빨간색)
    for j, (box, label) in enumerate(zip(pred_boxes.cpu().detach().numpy(), pred_labels.cpu().detach().numpy())):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='r', linewidth=2)
        axes[3].add_patch(rect)

        # 라벨명 찾기
        label_name = next((k for k, v in label_dict.items() if v == label), 'Unknown')

        # 마스크 그리기
        mask = pred_masks[j]
        mask_np = mask.squeeze().cpu().detach().numpy() > 0.5

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
    # 기존 코드 유지...

    # GPU 성능 개선을 위한 추가 설정
    torch.backends.cudnn.benchmark = True

    # 디버깅 도구 설정: CUDA 에러를 더 명확하게 보기 위함
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 디렉토리 설정
    image_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\food data"
    xml_base_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\food data\라벨 일부\xml"
    mask_base_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\re mask\mask"

    # 타겟 크기 설정
    target_size = (224, 224)

    # 데이터셋 생성 (최적화된 파일 매칭 및 라벨 인덱스 수정)
    dataset = MultiClassFoodDetectionDataset(image_dir, xml_base_dir, mask_base_dir, target_size=target_size)

    print("\n데이터셋 상세 정보:")
    for idx in range(len(dataset)):
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

    # 전후 비교를 추가하면 더좋은가?