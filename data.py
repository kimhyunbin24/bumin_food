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

        # 이미지 파일명 추출
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

        # 현재 XML 파일의 경로에서 클래스 정보 추출
        xml_dir = os.path.dirname(xml_path)
        class_code = os.path.basename(xml_dir)  # 폴더명이 클래스 코드

        # XML 파일 내의 객체 모두 처리
        for obj in root.findall('object'):
            name = obj.find('name').text
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

            # 라벨 매핑
            label = 0  # 기본값

            if name == 'dish':
                # dish는 라벨 1
                label = 1
            elif label_dict:
                # 음식 코드를 클래스 라벨로 매핑
                # 현재 폴더명으로 클래스 이름 결정
                for class_name, class_id in label_dict.items():
                    class_code = class_name.split('.')[0] if '.' in class_name else class_name
                    if name.startswith(class_code):
                        label = class_id
                        break

                # 폴더명으로 찾지 못한 경우, 직접 음식 코드를 찾음
                if label == 0:
                    for class_name, class_id in label_dict.items():
                        class_code = class_name.split('.')[0] if '.' in class_name else class_name
                        if name.startswith(class_code):
                            label = class_id
                            break

            labels.append(label)

            # 마스크 추출 (해당 라벨의 마스크만)
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
        # 파싱 오류 시 빈 텐서 반환
        height = target_size[1] if target_size else 224
        width = target_size[0] if target_size else 224
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0, height, width), dtype=torch.uint8),
            target_size or (width, height)
        )


class MultiClassFoodDetectionDataset(Dataset):
    """
    다중 클래스 음식 탐지 데이터셋 클래스
    """
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

        # data.py 파일의 마지막 부분 수정
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