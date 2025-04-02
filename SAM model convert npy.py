import os
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path


def safe_imread(image_path):
    """
    Safely read an image file with additional error checking
    """
    try:
        # Try reading with cv2 first
        image = cv2.imread(image_path)

        # Check if image is None or empty
        if image is None:
            # Try reading with PIL as a fallback
            from PIL import Image
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if image is None or image.size == 0:
            raise ValueError(f"Empty or invalid image: {image_path}")

        return image

    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        # Log the full path and filename for debugging
        print(f"Full path: {os.path.abspath(image_path)}")
        print(f"File exists: {os.path.exists(image_path)}")
        print(f"File size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'}")
        raise


def extract_bounding_boxes(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        class_names = []
        filename = root.find("filename").text

        for obj in root.findall("object"):
            name = obj.find("name").text
            # 모든 바운딩 박스 저장 (dish는 그대로, 나머지는 이름 그대로 사용)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            class_names.append(name.lower())

        return boxes, class_names, filename
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        raise


def segment_objects(xml_path, image_path, output_path, model_checkpoint="sam_vit_l_0b3195.pth"):
    # 이미지 파일 이름에서 확장자 제거
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 이미 생성된 파일인지 확인 (하나의 통합 마스크 파일)
    mask_file = os.path.join(output_path, f"{image_filename}.npy")
    if os.path.exists(mask_file):
        print(f"이미 처리된 파일, 건너뜁니다: {image_filename}")
        return

    # SAM 모델 로드
    sam = sam_model_registry["vit_l"](checkpoint=model_checkpoint).to("cuda")
    predictor = SamPredictor(sam)

    # 바운딩 박스 정보 가져오기
    bounding_boxes, class_names, _ = extract_bounding_boxes(xml_path)

    # 클래스 목록 확인
    unique_classes = set(class_names)
    print(f"이미지 {image_filename}에서 발견된 클래스: {unique_classes}")

    # 클래스가 없으면 처리 중단
    if not bounding_boxes:
        print(f"이미지 {image_filename}에서 객체가 발견되지 않았습니다. 건너뜁니다.")
        return

    print(f"이미지 경로: {image_path}")  # 경로 출력

    # 이미지 로드 (using safe_imread)
    image = safe_imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # 통합 마스크 초기화 (0: 배경)
    unified_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 바운딩 박스 기반 세그멘테이션 실행
    for i, bbox in enumerate(bounding_boxes):
        bbox_np = np.array(bbox)  # numpy 배열로 변환
        masks, _, _ = predictor.predict(box=bbox_np)

        # 마스크 값이 1인 부분을 해당 라벨로 할당
        mask = masks[0].astype(np.uint8)
        class_name = class_names[i]

        # 클래스별 ID 할당
        # dish는 1, 다른 클래스는 클래스마다 고유한 숫자(2, 3, 4...)로 할당
        class_id = 1 if class_name == "dish" else (list(unique_classes).index(class_name) + 2)

        # 객체 마스크를 통합 마스크에 반영
        # 이미 다른 객체가 있는 부분은 덮어쓰지 않도록 처리
        # (나중에 처리되는 객체가 이미 있는 객체를 덮어쓰지 않도록)
        object_pixels = (mask == 1) & (unified_mask == 0)
        unified_mask[object_pixels] = class_id

    # 통합 마스크를 .npy 파일로 저장 (확장자 없는 원본 파일명만 사용)
    np.save(os.path.join(output_path, f"{image_filename}"), unified_mask)

    print(f"SAM을 이용한 세그멘테이션 변환 완료: {image_filename}")


def find_matching_files(image_dir, xml_dir, output_base_dir):
    # 이미지 디렉토리에서 모든 이미지 파일 찾기
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

    # XML 디렉토리에서 모든 XML 파일 찾기
    xml_files = []
    for root, _, files in os.walk(xml_dir):
        for file in files:
            if file.lower().endswith('.xml'):
                xml_path = os.path.join(root, file)
                xml_files.append(xml_path)

    print(f"총 {len(xml_files)}개의 XML 파일을 찾았습니다.")

    # XML 파일을 딕셔너리로 변환하여 검색 속도 향상
    xml_dict = {}
    for xml_path in xml_files:
        xml_name = os.path.splitext(os.path.basename(xml_path))[0]
        xml_dict[xml_name] = xml_path

    print("XML 파일 인덱싱 완료")

    # 파일 이름 기반으로 매칭된 파일 찾기
    matches = []
    count = 0

    for image_path in image_files:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 이미지 이름으로 XML 파일 빠르게 검색
        if image_name in xml_dict:
            # 이미지 경로에서 하위 폴더 구조 추출
            rel_path = os.path.relpath(os.path.dirname(image_path), image_dir)

            # 출력 경로 구성
            if rel_path == '.':
                output_path = output_base_dir
            else:
                output_path = os.path.join(output_base_dir, rel_path)

            # 출력 디렉토리가 없으면 생성
            os.makedirs(output_path, exist_ok=True)

            matches.append((image_path, xml_dict[image_name], output_path))

            # 진행 상황 표시
            count += 1
            if count % 1000 == 0:
                print(f"매칭된 파일 {count}개 처리 중...")

    print(f"총 {len(matches)}개의 매칭된 파일을 찾았습니다.")
    return matches


def process_all_files(image_dir, xml_dir, output_dir, model_checkpoint="sam_vit_l_0b3195.pth"):
    # 매칭된 파일 찾기
    matches = find_matching_files(image_dir, xml_dir, output_dir)

    # 매칭된 파일 처리
    for idx, (image_path, xml_path, output_path) in enumerate(matches):
        try:
            print(f"[{idx + 1}/{len(matches)}] 처리 중: {os.path.basename(image_path)}")
            segment_objects(xml_path, image_path, output_path, model_checkpoint)
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {os.path.basename(image_path)}")
            print(f"오류 내용: {e}")
            continue


# 🚀 실행 코드
if __name__ == "__main__":
    # 경로 설정
    image_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\Training"
    xml_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\Training\[라벨]음식분류_TRAIN\xml"
    output_dir = r"F:\AI hub data\음식 이미지 및 영양정보 텍스트\Training\[라벨]음식분류_TRAIN\mask"

    # 전체 파일 처리 함수 호출
    process_all_files(image_dir, xml_dir, output_dir)