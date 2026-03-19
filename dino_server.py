#!/usr/bin/env python3
"""
DINO 모델 서버
Flutter 앱에서 크롭된 이미지를 받아 DINO 모델로 분류하는 HTTP 서버.
"""

import argparse
import base64
import io
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms


# live.py에서 DINOv2Classifier 클래스 재사용
class DINOv2Classifier(nn.Module):
    """DINOv2 분류 모델."""

    def __init__(self, backbone, embed_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


app = Flask(__name__)
CORS(app)  # Flutter 앱에서 접근 가능하도록 CORS 허용

# 전역 변수
models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_dino_model(model_path, model_size="small", num_classes=2):
    """DINOv2 모델 체크포인트 로드."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})

    model_size = config.get("model_size", model_size)
    num_classes = config.get("num_classes", num_classes)

    # 백본 로드
    model_map = {
        "small": ("dinov2_vits14", 384),
        "base": ("dinov2_vitb14", 768),
        "large": ("dinov2_vitl14", 1024),
        "giant": ("dinov2_vitg14", 1536),
    }
    model_name, embed_dim = model_map.get(model_size, ("dinov2_vits14", 384))

    backbone = torch.hub.load("facebookresearch/dinov2", model_name)
    model = DINOv2Classifier(backbone, embed_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, num_classes


@app.route("/health", methods=["GET"])
def health():
    """서버 상태 확인."""
    return jsonify({"status": "ok", "device": device, "models_loaded": list(models.keys())})


@app.route("/save_frame", methods=["POST"])
def save_frame():
    """정지된 프레임 이미지를 서버에 저장하고 YOLO 좌표로 크롭 Request: - image: 바이너리 이미지 파일 - detections: JSON 문자열 (YOLO 탐지 결과) - model_type:
    'bolt' 또는 'door' - filename: 저장할 파일명 (선택사항) Response: - success: bool - filepath: 저장된 파일 경로 - cropped_files: 크롭된
    이미지 파일 경로 리스트.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "image file is required"}), 400

        file = request.files["image"]
        image_bytes = file.read()

        # 이미지 디코딩
        import numpy as np

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # 파일명 생성
        filename = request.form.get("filename")
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"frozen_frame_{timestamp}.jpg"

        # image 폴더에 정지 프레임 저장
        image_dir = os.path.join(os.getcwd(), "image")
        os.makedirs(image_dir, exist_ok=True)
        image_filepath = os.path.join(image_dir, filename)
        cv2.imwrite(image_filepath, frame)
        print(f"  [DINO Server] 정지 프레임 저장: {image_filepath} ({len(image_bytes)} bytes)")

        # YOLO 좌표값 파싱 및 크롭
        detections_json = request.form.get("detections")
        model_type = request.form.get("model_type", "bolt")

        # 캡처된 프레임 크기
        frame_h, frame_w = frame.shape[:2]
        print(f"  [DINO Server] 캡처된 프레임 크기: {frame_w}x{frame_h}")

        # 클라이언트에서 전송한 프레임 크기 정보 (디버깅용)
        client_frame_w = request.form.get("frame_width")
        client_frame_h = request.form.get("frame_height")
        if client_frame_w and client_frame_h:
            print(f"  [DINO Server] 클라이언트가 전송한 프레임 크기: {client_frame_w}x{client_frame_h}")
            if int(client_frame_w) != frame_w or int(client_frame_h) != frame_h:
                print("  [DINO Server] ⚠️  경고: 프레임 크기가 일치하지 않습니다!")
                print(f"    서버 디코딩 크기: {frame_w}x{frame_h}")
                print(f"    클라이언트 전송 크기: {client_frame_w}x{client_frame_h}")

        # 화면 크기 및 원본 이미지 크기 정보 가져오기 (화면 변환 재현용)
        view_width = request.form.get("view_width")
        view_height = request.form.get("view_height")
        orig_width = request.form.get("orig_width")
        orig_height = request.form.get("orig_height")

        crop_result = {"cropped_files": [], "classification_results": []}
        if detections_json:
            try:
                import json

                detections = json.loads(detections_json)
                # 화면 변환 정보를 전달하여 정확한 크롭 수행
                crop_result = _crop_detections(
                    frame,
                    detections,
                    model_type,
                    view_width=int(view_width) if view_width else None,
                    view_height=int(view_height) if view_height else None,
                    orig_width=int(orig_width) if orig_width else None,
                    orig_height=int(orig_height) if orig_height else None,
                )
            except Exception as e:
                print(f"  [DINO Server] 크롭 처리 중 오류: {e}")
                import traceback

                traceback.print_exc()

        # Voting 로직 (live.py 참고)
        final_result = None
        if crop_result["classification_results"]:
            if model_type == "bolt":
                # 볼트: soft voting (평균 불량 확률)
                defect_confidences = [r["defect_confidence"] for r in crop_result["classification_results"]]
                avg_defect_conf = sum(defect_confidences) / len(defect_confidences) if defect_confidences else 0.0
                final_result = {
                    "is_good": avg_defect_conf < 0.5,
                    "result_text": "양품" if avg_defect_conf < 0.5 else "불량",
                    "avg_defect_confidence": avg_defect_conf,
                    "voting_method": "soft",
                }
            elif model_type == "door":
                # 도어: soft voting (평균 불량 확률)
                defect_confidences = [r["defect_confidence"] for r in crop_result["classification_results"]]
                avg_defect_conf = sum(defect_confidences) / len(defect_confidences) if defect_confidences else 0.0
                final_result = {
                    "is_good": avg_defect_conf < 0.5,
                    "result_text": "양품" if avg_defect_conf < 0.5 else "불량",
                    "avg_defect_confidence": avg_defect_conf,
                    "voting_method": "soft",
                }

        return jsonify(
            {
                "success": True,
                "filepath": image_filepath,
                "filename": filename,
                "size": len(image_bytes),
                "cropped_files": crop_result["cropped_files"],
                "classification_results": crop_result["classification_results"],
                "final_result": final_result,
            }
        )
    except Exception as e:
        print(f"DINO 서버 프레임 저장 중 오류: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _classify_cropped_image(cropped_img, model_key):
    """크롭된 이미지를 DINO 모델로 분류 (live.py의 _classify_with_dino 참고).

    Args:
        cropped_img: OpenCV 이미지 (numpy array, BGR)
        model_key: 'bolt', 'door_high', 'door_mid', 'door_low'

    Returns:
        분류 결과 딕셔너리
    """
    if model_key not in models:
        return {
            "is_defect": True,
            "confidence": [0.0, 1.0],
            "pred_class": 1,
            "defect_confidence": 1.0,
            "num_classes": 2,
            "error": f"Model {model_key} not loaded",
        }

    if cropped_img.size == 0:
        return {
            "is_defect": True,
            "confidence": [0.0, 1.0],
            "pred_class": 1,
            "defect_confidence": 1.0,
            "num_classes": 2,
            "error": "Empty image",
        }

    try:
        model, num_classes = models[model_key]

        # BGR -> RGB 변환
        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        # 전처리 및 텐서 변환
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # DINO 분류
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0].cpu().numpy().tolist()

        # 불량 판정 (live.py 참고)
        if num_classes == 4:
            is_defect = pred_class != 0
            defect_confidence = (
                sum(confidence[1:4]) if len(confidence) >= 4 else confidence[1] if len(confidence) >= 2 else 0.0
            )
        else:
            is_defect = pred_class == 1
            defect_confidence = confidence[1] if len(confidence) >= 2 else 0.0

        return {
            "is_defect": is_defect,
            "confidence": confidence,
            "pred_class": pred_class,
            "defect_confidence": defect_confidence,
            "num_classes": num_classes,
        }
    except Exception as e:
        print(f"  [DINO Server] 분류 중 오류: {e}")
        import traceback

        traceback.print_exc()
        return {
            "is_defect": True,
            "confidence": [0.0, 1.0],
            "pred_class": 1,
            "defect_confidence": 1.0,
            "num_classes": 2,
            "error": str(e),
        }


def _crop_detections(
    frame, detections, model_type, view_width=None, view_height=None, orig_width=None, orig_height=None
):
    """YOLO 탐지 결과를 이용하여 이미지 크롭 및 DINO 분류 (live.py 참고).

    화면에 그려진 박스와 정확히 같은 영역을 크롭하기 위해:
    1. 이미지를 화면에 그려진 것처럼 변환 (스케일 + 오프셋)
    2. 변환된 이미지에서 YOLO 좌표로 크롭
    3. 크롭된 이미지를 원본 비율로 복원

    Args:
        frame: OpenCV 이미지 (numpy array)
        detections: YOLO 탐지 결과 리스트
        model_type: 'bolt' 또는 'door'
        view_width: 화면 너비 (선택사항)
        view_height: 화면 높이 (선택사항)
        orig_width: YOLO 원본 이미지 너비 (선택사항)
        orig_height: YOLO 원본 이미지 높이 (선택사항)

    Returns:
        딕셔너리: {
        'cropped_files': 크롭된 이미지 파일 경로 리스트,
        'classification_results': 분류 결과 리스트 }
    """
    cropped_files = []
    classification_results = []

    try:
        # 프레임 크기 가져오기
        frame_h, frame_w = frame.shape[:2]

        # 화면 변환 정보 계산 (화면에 박스를 그릴 때 사용하는 변환 재현)
        use_screen_transform = False
        scale = 1.0
        dx = 0.0
        dy = 0.0

        if view_width and view_height and orig_width and orig_height:
            # 화면에 박스를 그릴 때 사용하는 변환 계산
            # Android YOLOView.kt 참고:
            # scale = max(vw/iw, vh/ih)
            # dx = (vw - iw*scale) / 2
            # dy = (vh - ih*scale) / 2
            scale_x = view_width / orig_width
            scale_y = view_height / orig_height
            scale = max(scale_x, scale_y)

            scaled_w = orig_width * scale
            scaled_h = orig_height * scale
            dx = (view_width - scaled_w) / 2.0
            dy = (view_height - scaled_h) / 2.0

            use_screen_transform = True
            print("  [DINO Server] 화면 변환 정보:")
            print(f"    원본 이미지 크기: {orig_width}x{orig_height}")
            print(f"    화면 크기: {view_width}x{view_height}")
            print(f"    스케일: {scale:.6f} (scaleX={scale_x:.6f}, scaleY={scale_y:.6f})")
            print(f"    오프셋: dx={dx:.2f}, dy={dy:.2f}")
            print(f"    프레임 크기: {frame_w}x{frame_h}")

        # debug_crop 폴더 생성
        debug_crop_dir = os.path.join(os.getcwd(), "debug_crop")
        os.makedirs(debug_crop_dir, exist_ok=True)

        if model_type == "bolt":
            # bolt 폴더 생성
            bolt_dir = os.path.join(debug_crop_dir, "bolt")
            os.makedirs(bolt_dir, exist_ok=True)

            # 볼트 클래스 매핑 (live.py 참고)
            bolt_class_names = {
                0: "bolt_frontside",
                1: "bolt_side",
                2: "sedan (trunklid)",
                3: "suv (trunklid)",
                4: "hood",
                5: "long (frontfender)",
                6: "mid (frontfender)",
                7: "short (frontfender)",
            }

            # 프레임 찾기
            frame_detection = None
            bolt_detections = []

            for det in detections:
                class_index = det.get("classIndex", -1)
                if class_index >= 2 and class_index <= 7:  # 프레임
                    frame_detection = det
                elif class_index == 0 or class_index == 1:  # 볼트
                    bolt_detections.append(det)

            if frame_detection:
                frame_class_index = frame_detection.get("classIndex", 2)
                frame_name = bolt_class_names.get(frame_class_index, "unknown")
                frame_bbox = frame_detection.get("boundingBox", {})

                # 프레임 내 볼트만 크롭 (live.py의 _inspect_bolt 참고)
                bolts_in_frame = []
                for bolt in bolt_detections:
                    bolt_bbox = bolt.get("boundingBox", {})
                    bolt_center = [
                        (bolt_bbox.get("left", 0) + bolt_bbox.get("right", 0)) / 2,
                        (bolt_bbox.get("top", 0) + bolt_bbox.get("bottom", 0)) / 2,
                    ]

                    # 프레임 내부에 있는지 확인 (간단한 AABB 체크)
                    if frame_bbox.get("left", 0) <= bolt_center[0] <= frame_bbox.get("right", 0) and frame_bbox.get(
                        "top", 0
                    ) <= bolt_center[1] <= frame_bbox.get("bottom", 0):
                        bolts_in_frame.append(bolt)

                # 각 볼트 크롭 (live.py: bolt_{i+1}_{frame_name}_{timestamp}.jpg)
                for i, bolt in enumerate(bolts_in_frame):
                    # ⚠️ 중요: 정규화 좌표만 사용 (화면 크기와 무관하게 정확한 크롭을 위해)
                    normalized_bbox = bolt.get("normalizedBox", {})
                    if not normalized_bbox:
                        print(f"  [DINO Server] ⚠️  볼트 #{i + 1}: 정규화 좌표가 없습니다. 건너뜁니다.")
                        continue

                    # YOLO가 내뱉은 정규화 좌표를 그대로 사용
                    norm_left = float(normalized_bbox.get("left", 0))
                    norm_top = float(normalized_bbox.get("top", 0))
                    norm_right = float(normalized_bbox.get("right", 0))
                    norm_bottom = float(normalized_bbox.get("bottom", 0))

                    if use_screen_transform:
                        # 화면에 그려진 박스와 정확히 같은 영역을 크롭하기 위해:
                        # 1. 정규화 좌표를 원본 이미지 크기 기준 픽셀 좌표로 변환
                        orig_x1 = norm_left * orig_width
                        orig_y1 = norm_top * orig_height
                        orig_x2 = norm_right * orig_width
                        orig_y2 = norm_bottom * orig_height

                        # 2. 화면 좌표로 변환 (화면에 그릴 때 사용하는 변환)
                        screen_x1 = orig_x1 * scale + dx
                        screen_y1 = orig_y1 * scale + dy
                        screen_x2 = orig_x2 * scale + dx
                        screen_y2 = orig_y2 * scale + dy

                        # 3. 화면 좌표를 프레임 크기에 맞게 스케일링
                        # 프레임 크기와 화면 크기의 비율 계산
                        frame_scale_x = frame_w / view_width
                        frame_scale_y = frame_h / view_height

                        # 프레임 좌표로 변환
                        x1 = int(screen_x1 * frame_scale_x)
                        y1 = int(screen_y1 * frame_scale_y)
                        x2 = int(screen_x2 * frame_scale_x)
                        y2 = int(screen_y2 * frame_scale_y)

                        print(f"  [DINO Server] 볼트 #{i + 1} 크롭 좌표 (화면 변환 적용):")
                        print(
                            f"    YOLO 정규화 좌표: left={norm_left:.6f}, top={norm_top:.6f}, right={norm_right:.6f}, bottom={norm_bottom:.6f}"
                        )
                        print(f"    원본 이미지 좌표: ({orig_x1:.2f}, {orig_y1:.2f}, {orig_x2:.2f}, {orig_y2:.2f})")
                        print(f"    화면 좌표: ({screen_x1:.2f}, {screen_y1:.2f}, {screen_x2:.2f}, {screen_y2:.2f})")
                        print(f"    프레임 좌표: ({x1}, {y1}, {x2}, {y2})")
                    else:
                        # 화면 변환 정보가 없으면 기존 방식 사용
                        x1_float = norm_left * frame_w
                        y1_float = norm_top * frame_h
                        x2_float = norm_right * frame_w
                        y2_float = norm_bottom * frame_h

                        x1 = int(x1_float)
                        y1 = int(y1_float)
                        x2 = int(x2_float)
                        y2 = int(y2_float)

                        print(f"  [DINO Server] 볼트 #{i + 1} 크롭 좌표 (기본 변환):")
                        print(
                            f"    YOLO 정규화 좌표: left={norm_left:.6f}, top={norm_top:.6f}, right={norm_right:.6f}, bottom={norm_bottom:.6f}"
                        )
                        print(f"    프레임 크기: {frame_w}x{frame_h}")
                        print(
                            f"    계산(float): x1={x1_float:.2f}, y1={y1_float:.2f}, x2={x2_float:.2f}, y2={y2_float:.2f}"
                        )
                        print(f"    변환된 픽셀 좌표: ({x1}, {y1}, {x2}, {y2})")

                    # 이미지 경계 확인 및 보정
                    x1 = max(0, min(x1, frame_w - 1))
                    y1 = max(0, min(y1, frame_h - 1))
                    x2 = max(x1 + 1, min(x2, frame_w))
                    y2 = max(y1 + 1, min(y2, frame_h))

                    print(f"    경계 보정 후: ({x1}, {y1}, {x2}, {y2})")

                    if x2 > x1 and y2 > y1:
                        # NumPy 배열 슬라이싱으로 크롭 (flip 없이 원본 그대로)
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            crop_filename = f"bolt_{i + 1}_{frame_name}_{timestamp}.jpg"
                            crop_filepath = os.path.join(bolt_dir, crop_filename)
                            cv2.imwrite(crop_filepath, cropped)
                            cropped_files.append(crop_filepath)
                            print(
                                f"  [DINO Server] 볼트 크롭 저장: {crop_filepath} (크기: {cropped.shape[1]}x{cropped.shape[0]})"
                            )

                            # DINO 모델로 분류 (live.py의 _inspect_bolt 참고)
                            print(f"  [DINO Server] 볼트 #{i + 1} DINO 분류 시작...")
                            result = _classify_cropped_image(cropped, "bolt")
                            result["bolt_index"] = i + 1
                            result["frame_name"] = frame_name
                            result["crop_filepath"] = crop_filepath
                            classification_results.append(result)

                            result_text = "불량" if result["is_defect"] else "양품"
                            conf_display = result["confidence"][result["pred_class"]]
                            print(f"  [DINO Server] 볼트 #{i + 1}: {result_text} (신뢰도: {conf_display:.2%})")

        elif model_type == "door":
            # door 폴더 생성
            door_dir = os.path.join(debug_crop_dir, "door")
            os.makedirs(door_dir, exist_ok=True)

            # door 파트별로 분류
            parts = {"high": [], "mid": [], "low": []}

            for det in detections:
                class_name = det.get("className", "").lower()
                if class_name in parts:
                    parts[class_name].append(det)

            # 각 파트 크롭 (live.py: frontdoor_{part}_{timestamp}.jpg)
            for part in ["high", "mid", "low"]:
                if parts[part]:
                    part_det = parts[part][0]  # 첫 번째 탐지만 사용
                    # ⚠️ 중요: 정규화 좌표만 사용 (화면 크기와 무관하게 정확한 크롭을 위해)
                    normalized_bbox = part_det.get("normalizedBox", {})
                    if not normalized_bbox:
                        print(f"  [DINO Server] ⚠️  도어 {part.upper()}: 정규화 좌표가 없습니다. 건너뜁니다.")
                        continue

                    # YOLO가 내뱉은 정규화 좌표를 그대로 사용
                    norm_left = float(normalized_bbox.get("left", 0))
                    norm_top = float(normalized_bbox.get("top", 0))
                    norm_right = float(normalized_bbox.get("right", 0))
                    norm_bottom = float(normalized_bbox.get("bottom", 0))

                    if use_screen_transform:
                        # 화면에 그려진 박스와 정확히 같은 영역을 크롭하기 위해:
                        # 1. 정규화 좌표를 원본 이미지 크기 기준 픽셀 좌표로 변환
                        orig_x1 = norm_left * orig_width
                        orig_y1 = norm_top * orig_height
                        orig_x2 = norm_right * orig_width
                        orig_y2 = norm_bottom * orig_height

                        # 2. 화면 좌표로 변환 (화면에 그릴 때 사용하는 변환)
                        screen_x1 = orig_x1 * scale + dx
                        screen_y1 = orig_y1 * scale + dy
                        screen_x2 = orig_x2 * scale + dx
                        screen_y2 = orig_y2 * scale + dy

                        # 3. 화면 좌표를 프레임 크기에 맞게 스케일링
                        # 프레임 크기와 화면 크기의 비율 계산
                        frame_scale_x = frame_w / view_width
                        frame_scale_y = frame_h / view_height

                        # 프레임 좌표로 변환
                        x1 = int(screen_x1 * frame_scale_x)
                        y1 = int(screen_y1 * frame_scale_y)
                        x2 = int(screen_x2 * frame_scale_x)
                        y2 = int(screen_y2 * frame_scale_y)

                        print(f"  [DINO Server] 도어 {part.upper()} 크롭 좌표 (화면 변환 적용):")
                        print(
                            f"    YOLO 정규화 좌표: left={norm_left:.6f}, top={norm_top:.6f}, right={norm_right:.6f}, bottom={norm_bottom:.6f}"
                        )
                        print(f"    원본 이미지 좌표: ({orig_x1:.2f}, {orig_y1:.2f}, {orig_x2:.2f}, {orig_y2:.2f})")
                        print(f"    화면 좌표: ({screen_x1:.2f}, {screen_y1:.2f}, {screen_x2:.2f}, {screen_y2:.2f})")
                        print(f"    프레임 좌표: ({x1}, {y1}, {x2}, {y2})")
                    else:
                        # 화면 변환 정보가 없으면 기존 방식 사용
                        x1_float = norm_left * frame_w
                        y1_float = norm_top * frame_h
                        x2_float = norm_right * frame_w
                        y2_float = norm_bottom * frame_h

                        x1 = int(x1_float)
                        y1 = int(y1_float)
                        x2 = int(x2_float)
                        y2 = int(y2_float)

                        print(f"  [DINO Server] 도어 {part.upper()} 크롭 좌표 (기본 변환):")
                        print(
                            f"    YOLO 정규화 좌표: left={norm_left:.6f}, top={norm_top:.6f}, right={norm_right:.6f}, bottom={norm_bottom:.6f}"
                        )
                        print(f"    프레임 크기: {frame_w}x{frame_h}")
                        print(
                            f"    계산(float): x1={x1_float:.2f}, y1={y1_float:.2f}, x2={x2_float:.2f}, y2={y2_float:.2f}"
                        )
                        print(f"    변환된 픽셀 좌표: ({x1}, {y1}, {x2}, {y2})")

                    # 이미지 경계 확인 및 보정
                    x1 = max(0, min(x1, frame_w - 1))
                    y1 = max(0, min(y1, frame_h - 1))
                    x2 = max(x1 + 1, min(x2, frame_w))
                    y2 = max(y1 + 1, min(y2, frame_h))

                    print(f"    경계 보정 후: ({x1}, {y1}, {x2}, {y2})")

                    if x2 > x1 and y2 > y1:
                        # NumPy 배열 슬라이싱으로 크롭 (flip 없이 원본 그대로)
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함
                            crop_filename = f"frontdoor_{part}_{timestamp}.jpg"
                            crop_filepath = os.path.join(door_dir, crop_filename)
                            cv2.imwrite(crop_filepath, cropped)
                            cropped_files.append(crop_filepath)
                            print(
                                f"  [DINO Server] 도어 {part.upper()} 크롭 저장: {crop_filepath} (크기: {cropped.shape[1]}x{cropped.shape[0]})"
                            )

                            # DINO 모델로 분류 (live.py의 _inspect_frontdoor 참고)
                            model_key = f"door_{part}"  # door_high, door_mid, door_low
                            print(f"  [DINO Server] 도어 {part.upper()} DINO 분류 시작...")
                            result = _classify_cropped_image(cropped, model_key)
                            result["part"] = part
                            result["crop_filepath"] = crop_filepath
                            classification_results.append(result)

                            result_text = "불량" if result["is_defect"] else "양품"
                            conf_display = result["confidence"][result["pred_class"]]
                            print(f"  [DINO Server] 도어 {part.upper()}: {result_text} (신뢰도: {conf_display:.2%})")

    except Exception as e:
        print(f"  [DINO Server] 크롭 처리 중 오류: {e}")
        import traceback

        traceback.print_exc()

    return {"cropped_files": cropped_files, "classification_results": classification_results}


@app.route("/classify", methods=["POST"])
def classify():
    """이미지 분류 엔드포인트 Request: - image: base64 인코딩된 이미지 또는 바이너리 - model_type: 'bolt' 또는 'door_high', 'door_mid',
    'door_low' - format: 'base64' 또는 'binary' (기본값: 'binary') Response: - is_defect: bool - confidence: List[float]
    (각 클래스별 확률) - pred_class: int (예측된 클래스 인덱스) - defect_confidence: float (불량 확률) - num_classes: int.
    """
    try:
        # 모델 타입 확인
        model_type = request.form.get("model_type") or (request.json.get("model_type") if request.is_json else None)
        if not model_type:
            return jsonify({"error": "model_type is required"}), 400

        # door_high, door_mid, door_low를 door로 매핑하여 확인
        model_key = model_type
        if model_type.startswith("door_"):
            # door_high, door_mid, door_low는 그대로 사용
            model_key = model_type
        elif model_type == "bolt":
            model_key = "bolt"

        if model_key not in models:
            return jsonify({"error": f"Model {model_key} not loaded. Available models: {list(models.keys())}"}), 404

        model, num_classes = models[model_key]

        # 이미지 받기
        image_format = request.form.get("format", "binary")

        if image_format == "base64":
            # Base64 인코딩된 이미지
            image_data = request.json.get("image") or request.form.get("image")
            if not image_data:
                return jsonify({"error": "image is required"}), 400

            # Base64 디코딩
            if image_data.startswith("data:image"):
                # data:image/png;base64,xxx 형식
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # 바이너리 이미지
            if "image" not in request.files:
                return jsonify({"error": "image file is required"}), 400

            file = request.files["image"]
            image = Image.open(io.BytesIO(file.read()))

        # RGB 변환
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 디버그: 크롭 이미지 저장 (서버 PC에 저장)
        # live.py 참고: bolt_{i+1}_{frame_name}_{timestamp}.jpg 또는 door_{part}_{timestamp}.jpg
        debug_crop_dir = os.path.join(os.getcwd(), "debug_crop")
        if not os.path.exists(debug_crop_dir):
            os.makedirs(debug_crop_dir, exist_ok=True)

        # 모델 타입에 따라 하위 폴더 결정
        if model_type == "bolt":
            sub_dir = os.path.join(debug_crop_dir, "bolt")
        elif model_type.startswith("door_"):
            sub_dir = os.path.join(debug_crop_dir, "door")
        else:
            sub_dir = os.path.join(debug_crop_dir, "other")

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)

        # 파일명 생성 (live.py 스타일)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함

        # 요청에서 파일명 정보 가져오기 (있다면)
        filename_prefix = request.form.get("filename_prefix", "")
        if filename_prefix:
            filename = f"{filename_prefix}_{timestamp}.png"
        else:
            if model_type == "bolt":
                filename = f"bolt_{timestamp}.png"
            elif model_type.startswith("door_"):
                part = model_type.replace("door_", "")
                filename = f"door_{part}_{timestamp}.png"
            else:
                filename = f"{model_type}_{timestamp}.png"

        crop_filepath = os.path.join(sub_dir, filename)

        # 원본 크롭 이미지 저장 (224x224 리사이즈 전)
        # 이미지를 numpy 배열로 변환하여 저장
        img_array = np.array(image)
        cv2.imwrite(crop_filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        print(f"  [서버] 크롭 이미지 저장: {crop_filepath} (크기: {image.size[0]}x{image.size[1]})")

        # 전처리 및 추론
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0].cpu().numpy().tolist()

        # 결과 해석
        if num_classes == 4:
            is_defect = pred_class != 0
            defect_confidence = (
                sum(confidence[1:4]) if len(confidence) >= 4 else confidence[1] if len(confidence) >= 2 else 0.0
            )
        else:
            is_defect = pred_class == 1
            defect_confidence = confidence[1] if len(confidence) >= 2 else 0.0

        return jsonify(
            {
                "is_defect": is_defect,
                "confidence": confidence,
                "pred_class": pred_class,
                "defect_confidence": float(defect_confidence),
                "num_classes": num_classes,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="DINO 모델 서버")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트 (기본값: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="서버 포트 (기본값: 5000)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="디바이스 (기본값: cuda)")

    # 모델 경로 설정
    parser.add_argument("--bolt-model", type=str, help="볼트 DINO 모델 경로")
    parser.add_argument("--door-high-model", type=str, help="도어 High DINO 모델 경로")
    parser.add_argument("--door-mid-model", type=str, help="도어 Mid DINO 모델 경로")
    parser.add_argument("--door-low-model", type=str, help="도어 Low DINO 모델 경로")

    args = parser.parse_args()

    global device
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    print("🚀 DINO 모델 서버 시작")
    print(f"  디바이스: {device}")
    print(f"  호스트: {args.host}")
    print(f"  포트: {args.port}\n")

    # 모델 로드
    if args.bolt_model:
        print(f"🔄 볼트 모델 로드 중: {args.bolt_model}")
        models["bolt"] = load_dino_model(args.bolt_model, num_classes=2)
        print("✓ 볼트 모델 로드 완료 (2-class)\n")

    if args.door_high_model:
        print(f"🔄 도어 High 모델 로드 중: {args.door_high_model}")
        models["door_high"] = load_dino_model(args.door_high_model)
        print("✓ 도어 High 모델 로드 완료\n")

    if args.door_mid_model:
        print(f"🔄 도어 Mid 모델 로드 중: {args.door_mid_model}")
        models["door_mid"] = load_dino_model(args.door_mid_model)
        print("✓ 도어 Mid 모델 로드 완료\n")

    if args.door_low_model:
        print(f"🔄 도어 Low 모델 로드 중: {args.door_low_model}")
        models["door_low"] = load_dino_model(args.door_low_model)
        print("✓ 도어 Low 모델 로드 완료\n")

    if not models:
        print("⚠️  경고: 로드된 모델이 없습니다. --bolt-model 또는 --door-*-model 옵션을 사용하세요.")

    print(f"✅ 서버 준비 완료! http://{args.host}:{args.port} 에서 실행 중...\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
