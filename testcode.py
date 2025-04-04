# test_xml.py 파일 생성
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from config import IMAGE_DIR, XML_BASE_DIR, MASK_BASE_DIR, TARGET_SIZE
from data import MultiClassFoodDetectionDataset, parse_annotation

# 테스트 데이터셋 생성
dataset = MultiClassFoodDetectionDataset(IMAGE_DIR, XML_BASE_DIR, MASK_BASE_DIR, target_size=TARGET_SIZE)


# 한글 경로에서 이미지 읽기 함수
def read_image_with_korean_path(img_path):
    # NumPy 파일로 읽고, OpenCV로 디코딩
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


# XML 파일 직접 분석 함수
def inspect_xml(xml_path, label_dict):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        print(f"  XML 파일 구조 분석:")
        print(f"    Root 태그: {root.tag}")

        # 파일명 확인
        filename = root.find('filename')
        if filename is not None:
            print(f"    파일명: {filename.text}")

        # 이미지 크기 정보
        size = root.find('size')
        if size is not None:
            width = size.find('width')
            height = size.find('height')
            depth = size.find('depth')
            print(f"    이미지 크기: {width.text}x{height.text}x{depth.text if depth is not None else '?'}")

        # 객체 정보
        objects = root.findall('object')
        print(f"    찾은 객체 수: {len(objects)}")

        for i, obj in enumerate(objects):
            name = obj.find('name')
            if name is not None:
                print(f"    객체 {i + 1} 이름: {name.text}")

                # 라벨 매핑 확인
                if name.text == 'dish':
                    print(f"      라벨 매핑: 1 (접시)")
                else:
                    # 음식 코드를 클래스 라벨로 매핑 시도
                    matched = False
                    for class_name, class_id in label_dict.items():
                        if class_name != 'dish' and name.text.startswith(class_name.split('.')[0]):
                            print(f"      라벨 매핑: {class_id} ({class_name})")
                            matched = True
                            break

                    if not matched:
                        print(f"      라벨 매핑 없음 (코드: {name.text})")

            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = bbox.find('xmin')
                ymin = bbox.find('ymin')
                xmax = bbox.find('xmax')
                ymax = bbox.find('ymax')

                if all([xmin, ymin, xmax, ymax]):
                    print(f"      바운딩 박스: ({xmin.text}, {ymin.text}), ({xmax.text}, {ymax.text})")
                else:
                    print(f"      바운딩 박스 정보 불완전")
            else:
                print(f"      바운딩 박스 정보 없음")
    except Exception as e:
        print(f"  XML 파일 분석 오류: {e}")


# 데이터 샘플 확인
for idx in range(5):
    img_path = dataset.image_files[idx]
    xml_path = dataset.xml_files[idx]
    mask_path = dataset.mask_files[idx]

    print(f"\n샘플 {idx}:")
    print(f"  이미지 경로: {img_path}")
    print(f"  XML 경로: {xml_path}")
    print(f"  마스크 경로: {mask_path}")

    # 현재 클래스 정보
    class_folder = os.path.basename(os.path.dirname(img_path))
    print(f"  이미지 클래스 폴더: {class_folder}")
    if class_folder in dataset.label_dict:
        print(f"  클래스 라벨: {dataset.label_dict[class_folder]}")

    # XML 파일 직접 분석
    inspect_xml(xml_path, dataset.label_dict)

    # 이미지 경로 유효성 확인
    if not os.path.exists(img_path):
        print(f"  경고: 이미지 파일이 존재하지 않습니다!")
        continue

    # 한글 경로 이미지 로드
    try:
        img = read_image_with_korean_path(img_path)
        if img is None:
            print(f"  경고: 이미지를 로드할 수 없습니다!")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  이미지 로드 오류: {e}")
        continue

    # 마스크 로드
    try:
        mask = np.load(mask_path)
    except Exception as e:
        print(f"  마스크 로드 오류: {e}")
        continue

    # parse_annotation 함수 호출 (수정된 코드의 동작 확인)
    boxes, labels, masks, _ = parse_annotation(
        xml_path,
        TARGET_SIZE,
        label_dict=dataset.label_dict,
        mask_base_dir=os.path.dirname(mask_path)
    )

    # 결과 출력
    print(f"  parse_annotation 결과:")
    print(f"    바운딩 박스 수: {len(boxes)}")

    if len(boxes) > 0:
        for i, (box, label) in enumerate(zip(boxes, labels)):
            class_name = next((name for name, idx in dataset.label_dict.items()
                               if idx == label.item()), f"Unknown-{label.item()}")
            print(f"    객체 {i + 1}: 라벨={label.item()} ({class_name}), 박스={box.numpy()}")

    print(f"    마스크 수: {masks.shape[0]}")
    print(f"  마스크 고유값: {np.unique(mask)}")
    print(f"  이미지 크기: {img.shape}")
    print(f"  마스크 크기: {mask.shape}")

    # 시각화
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("원본 이미지")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='viridis')  # 마스크를 더 잘 볼 수 있도록 색상맵 변경
    plt.title(f"마스크 (값: {np.unique(mask)})")

    plt.subplot(1, 3, 3)
    plt.imshow(img)

    # parse_annotation으로 추출한 바운딩 박스 그리기
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            label = labels[i].item() if i < len(labels) else 0

            # 클래스에 따라 색상 변경
            color = {
                1: 'r',  # 접시
                2: 'g',  # 밥
                3: 'b',  # 국, 탕, 찌개
                4: 'y',  # 조림
                5: 'm',  # 나물
                6: 'c',  # 김치
            }.get(label, 'w')

            # 클래스 이름 찾기
            class_name = next((name for name, idx in dataset.label_dict.items()
                               if idx == label), f"Unknown-{label}")

            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=color, linewidth=2))
            plt.text(x1, y1 - 5, class_name, color=color,
                     bbox=dict(facecolor='white', alpha=0.7))

    plt.title("XML에서 추출한 바운딩 박스")
    plt.tight_layout()
    plt.show()

    # 사용자 입력으로 다음 샘플 진행
    if idx < 4:  # 마지막 샘플이 아니라면
        response = input("다음 샘플을 보시겠습니까? (y/n): ")
        if response.lower() != 'y':
            break