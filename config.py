import os
import gc
import torch

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 사용 가능한 모든 GPU 사용
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.3, device=0)

# 디렉토리 설정
# 유니코드 경로 처리
IMAGE_DIR = os.path.normpath(r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\food data")
XML_BASE_DIR = os.path.normpath(r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\food data\라벨 일부\xml")
MASK_BASE_DIR = os.path.normpath(r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\re mask\mask")

# 타겟 크기 설정
TARGET_SIZE = (224, 224)

# 음식 양 클래스 라벨
VOLUME_LABELS = ['0-25%', '25-50%', '50-75%', '75-100%']


def create_label_dict(base_dir):
    label_dict = {
        'background': 0,
        'dish': 1
    }

    # 숫자로 시작하는 폴더 찾기
    food_dirs = [d for d in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, d))
                 and d[0].isdigit()]

    # 폴더 이름으로 라벨 생성
    for idx, dir_name in enumerate(sorted(food_dirs), start=2):
        label_dict[dir_name] = idx

    return label_dict


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


def enable_memory_growth():
    """
    GPU 메모리 동적 할당 활성화
    """
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()


def clear_memory():
    """
    메모리 정리 함수
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 추가: 메모리 사용량 출력
        print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB Allocated")


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


def calculate_volume_class(mask_array):
    """
    모든 음식 라벨(2 이상)을 고려하여 음식 양 클래스를 계산합니다.

    Args:
        mask_array: 마스크 배열

    Returns:
        int: 음식 양 클래스 (0: 0-25%, 1: 25-50%, 2: 50-75%, 3: 75-100%)
    """
    import numpy as np

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