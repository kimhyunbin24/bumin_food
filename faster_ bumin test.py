import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.ops import MultiScaleRoIAlign
import os


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    return model


def detect_food(image_path, model, device, confidence_threshold=0.5):
    # 이미지 로드 및 전처리
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 이미지 크기 제한
    max_size = 640
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

    # 예측
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # confidence threshold 적용
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]

    return image_rgb, pred_boxes, pred_scores


def visualize_results(image, boxes, scores, save_path=None):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          fill=False, edgecolor='red', linewidth=2))
        plt.text(xmin, ymin, f'Food: {score:.2f}',
                 color='red', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')
    plt.title("Food Detection Result")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def process_folder(folder_path, model, device, save_results=True):
    # 결과 저장을 위한 폴더 생성
    if save_results:
        results_folder = os.path.join(os.path.dirname(folder_path), 'detection_results')
        os.makedirs(results_folder, exist_ok=True)

    # 이미지 확장자 목록
    image_extensions = ('.jpg', '.jpeg', '.png')

    # 폴더 내 모든 이미지 처리
    total_images = 0
    processed_images = 0

    # 전체 이미지 수 계산
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            total_images += 1

    print(f"\n총 {total_images}개의 이미지를 처리합니다.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            processed_images += 1
            image_path = os.path.join(folder_path, filename)
            print(f"\n[{processed_images}/{total_images}] 처리 중: {filename}")

            try:
                # 이미지 처리
                image, boxes, scores = detect_food(image_path, model, device)
                print(f"발견된 음식 객체 수: {len(boxes)}")

                # 결과 저장 또는 표시
                if save_results:
                    save_path = os.path.join(results_folder, f'result_{filename}')
                    visualize_results(image, boxes, scores, save_path)
                    print(f"결과 저장됨: {save_path}")
                else:
                    visualize_results(image, boxes, scores)
                    input("다음 이미지를 보려면 Enter를 누르세요...")

            except Exception as e:
                print(f"이미지 처리 중 에러 발생: {e}")
                continue


def main():
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    model = get_model(num_classes=3)  # 배경 + 2 음식 클래스로 num_classes=3

    try:
        model.load_state_dict(torch.load('food_detection_model_10.pth'))
        print("모델 가중치를 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"모델 가중치를 불러오는데 실패했습니다: {e}")
        return

    model = model.to(device)

    # 테스트할 폴더 경로
    folder_path = r"C:\Users\furim\Desktop\Bumin_dataset\202306\전영숙_F76_45"

    # 폴더 내 모든 이미지 처리
    process_folder(folder_path, model, device, save_results=True)


if __name__ == "__main__":
    main()