import argparse
import math
import os
import threading
import time
from datetime import datetime
from math import cos, sin

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


class ThreadedCamera:
    """안전장치가 추가된 최신 프레임 카메라 클래스."""

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.lock = threading.Lock()  # 쓰레드 충돌 방지
        self.status = False
        self.frame = None
        self.stopped = False

        # 카메라가 정상적으로 열렸는지 확인
        if self.capture.isOpened():
            self.status, self.frame = self.capture.read()
            if self.status:
                self.thread = threading.Thread(target=self.update, args=())
                self.thread.daemon = True
                self.thread.start()
            else:
                print("❌ 카메라에서 첫 프레임을 읽을 수 없습니다.")
        else:
            print(f"❌ 카메라를 열 수 없습니다: {src}")

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                # 버퍼 없이 읽기 (grab -> retrieve 방식이 더 빠름)
                status, frame = self.capture.read()
                with self.lock:
                    if status:
                        self.status = status
                        self.frame = frame
                    else:
                        # 읽기 실패 시 잠시 대기 (CPU 폭주 방지)
                        time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.status, self.frame

    def isOpened(self):
        return self.capture.isOpened()

    def release(self):
        self.stopped = True
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.capture.release()

    def set(self, propId, value):
        return self.capture.set(propId, value)

    def get(self, propId):
        return self.capture.get(propId)


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


class RealtimeInspectionSystem:
    def __init__(
        self,
        mode="frontdoor",
        yolo_model_path=None,
        dino_models=None,
        device="cuda",
        conf_threshold=0.25,
        voting_method="soft",
        use_obb=False,
        debug=False,
        detect_only=False,
    ):
        """실시간 카메라 검사 시스템."""
        self.mode = mode.lower()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.voting_method = voting_method
        self.use_obb = use_obb
        self.debug = debug
        self.detect_only = detect_only

        # YOLO 모델 로드
        print(f"🔄 YOLO 모델 로드 중: {yolo_model_path}")
        if os.path.exists(yolo_model_path):
            file_size = os.path.getsize(yolo_model_path) / (1024 * 1024)  # MB
            print(f"  모델 파일 경로: {yolo_model_path}")
            print(f"  파일 크기: {file_size:.2f} MB")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print("✓ YOLO 모델 로드 완료")
            if hasattr(self.yolo_model, "names"):
                print(f"  - 클래스 수: {len(self.yolo_model.names)}")
                print(f"  - 클래스 목록: {list(self.yolo_model.names.values())}")
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
            raise

        # DINOv2 모델 로드 및 클래스 수 확인 (detect_only 모드가 아닐 때만)
        self.dino_models = {}
        self.dino_num_classes = {}  # 각 모델의 클래스 수 저장

        if not self.detect_only:
            if self.mode == "frontdoor":
                for part in ["high", "mid", "low"]:
                    print(f"🔄 DINOv2 모델 로드 중 ({part}): {dino_models[part]}")
                    model, num_classes = self._load_dino_model(dino_models[part])
                    self.dino_models[part] = model
                    self.dino_num_classes[part] = num_classes
            else:  # bolt
                print(f"🔄 DINOv2 모델 로드 중 (bolt): {dino_models['bolt']}")
                model, num_classes = self._load_dino_model(dino_models["bolt"])
                self.dino_models["bolt"] = model
                self.dino_num_classes["bolt"] = num_classes
        else:
            print("ℹ️  검출 전용 모드: DINOv2 모델 로드 생략")

        # DINOv2 전처리
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # 조건 체크 변수
        self.condition_start_time = None
        self.condition_start_frame = None  # 비디오 파일용 프레임 카운터
        self.condition_met = False
        self.last_valid_frame = None
        self.last_valid_detections = None

        # 크롭 이미지 저장용 디렉토리 생성 (디버깅용)
        self.debug_crop_dir = None
        if self.debug:
            self.debug_crop_dir = "debug_crops"
            os.makedirs(self.debug_crop_dir, exist_ok=True)
            print(f"  - 디버그 크롭 이미지 저장 경로: {self.debug_crop_dir}/")

        # 타임아웃 설정
        if self.mode == "frontdoor":
            self.required_duration = 3.0  # 3초
        else:  # bolt
            self.required_duration = 3.0  # 3초

        # YOLO 클래스 매핑 (bolt 모드용)
        self.bolt_class_names = {
            0: "bolt_frontside",
            1: "bolt_side",
            2: "sedan (trunklid)",
            3: "suv (trunklid)",
            4: "hood",
            5: "long (frontfender)",
            6: "mid (frontfender)",
            7: "short (frontfender)",
        }

        # DINO 모드 확인 (config에서 읽어온 값 사용)
        self.dino_mode = None  # 나중에 config에서 설정

        print("✓ 실시간 검사 시스템 초기화 완료")
        print(f"  - 모드: {self.mode}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - YOLO 신뢰도: {self.conf_threshold}")
        if self.detect_only:
            print("  - 검출 전용 모드: 활성화 (검사 기능 비활성화)")
        else:
            print(f"  - 조건 유지 시간: {self.required_duration}초")
            print(f"  - Voting 방법: {self.voting_method}")
        if self.use_obb:
            print("  - OBB 모드: 활성화")

        # DINO 클래스 수 출력 (detect_only 모드가 아닐 때만)
        if not self.detect_only:
            if self.mode == "frontdoor":
                for part in ["high", "mid", "low"]:
                    num_cls = self.dino_num_classes.get(part, 2)
                    mode_text = "4-class" if num_cls == 4 else "2-class (simple)"
                    print(f"  - DINO {part}: {mode_text}")
            else:
                # 볼트는 항상 2-class
                print("  - DINO bolt: 2-class (simple)")

    def _load_dino_model(self, model_path):
        """DINOv2 모델 체크포인트 로드."""
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"  모델 파일 경로: {model_path}")
            print(f"  파일 크기: {file_size:.2f} MB")
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("config", {})

        model_size = config.get("model_size", "small")
        num_classes = config.get("num_classes", 2)

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
        model.to(self.device)
        model.eval()

        return model, num_classes

    def run(self, source=0):
        """실시간 검사 실행.

        Args:
            source: 카메라 소스 (0: 웹캠, 또는 RTSP URL 등).
        """
        print(f"\n{'=' * 60}")
        print(f"🎥 카메라 시작: {source}")
        print(f"{'=' * 60}\n")

        # 비디오 파일인지 확인
        is_video_file = False
        if isinstance(source, str) and (
            source.endswith(".mp4")
            or source.endswith(".avi")
            or source.endswith(".mov")
            or source.endswith(".mkv")
            or source.endswith(".flv")
            or source.endswith(".wmv")
        ):
            is_video_file = True

        # 비디오 파일인 경우 직접 VideoCapture 사용 (모든 프레임 처리)
        # 카메라는 ThreadedCamera 사용
        if is_video_file:
            cap = cv2.VideoCapture(source)
        else:
            cap = ThreadedCamera(source)

        if not cap.isOpened():
            print(f"❌ 카메라를 열 수 없습니다: {source}")
            return

        # 비디오 파일의 FPS 및 해상도 가져오기
        video_fps = None
        video_width = None
        video_height = None
        total_frames = None
        video_writer = None
        output_video_path = None

        if is_video_file:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps > 0:
                print(
                    f"📹 비디오 파일 감지: FPS = {video_fps:.2f}, 해상도 = {video_width}x{video_height}, 총 프레임 = {total_frames}"
                )
            else:
                video_fps = 30.0
                print("⚠️  비디오 FPS를 가져올 수 없어 기본값 30 FPS 사용")

            # 출력 비디오 파일 경로 생성
            import os

            base_name = os.path.splitext(source)[0]
            ext = os.path.splitext(source)[1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = f"{base_name}_output_{timestamp}{ext}"

            # VideoWriter 초기화 (여러 코덱 시도하여 호환성 확보)
            fourcc_options = [
                ("avc1", "H.264 (avc1)"),
                ("mp4v", "MPEG-4 (mp4v)"),
                ("XVID", "Xvid"),
                ("MJPG", "Motion JPEG"),
            ]

            video_writer = None
            for fourcc_code, codec_name in fourcc_options:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))
                if video_writer.isOpened():
                    print(f"💾 출력 비디오 저장 경로: {output_video_path}")
                    print(f"   코덱: {codec_name}, FPS: {video_fps:.2f}, 해상도: {video_width}x{video_height}")
                    break
                else:
                    if video_writer:
                        video_writer.release()
                    video_writer = None

            if video_writer is None or not video_writer.isOpened():
                print("⚠️  비디오 Writer 초기화 실패. 비디오 저장이 불가능할 수 있습니다.")
                video_writer = None
        else:
            # 카메라 속성 설정 (비디오 파일이 아닐 때만)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # 카메라 초기화 대기
        print("✓ 카메라 연결 성공")
        if not is_video_file:
            print("🔄 카메라 초기화 중...")
            time.sleep(1.0)

            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    print("✓ 카메라 준비 완료")
                    break
                time.sleep(0.2)
            else:
                print("⚠️  카메라에서 프레임을 읽을 수 없습니다. 재시도 중...")
                time.sleep(1.0)
                ret, frame = cap.read()
                if not ret:
                    print("❌ 카메라에서 프레임을 읽을 수 없습니다.")
                    cap.release()
                    if video_writer:
                        video_writer.release()
                    return

        if self.detect_only:
            print("📋 검출 전용 모드: YOLO 검출 결과만 표시됩니다")
            print("   종료하려면 'q' 키를 누르세요\n")
        else:
            print("📋 대기 중... (조건이 만족되면 자동으로 캡처됩니다)")
            print("   종료하려면 'q' 키를 누르세요\n")

        try:
            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file:
                        # 비디오 파일이 끝난 경우
                        elapsed_time = time.time() - start_time
                        print(
                            f"\n📹 비디오 파일 재생 완료 (총 {frame_count} 프레임 처리, 소요 시간: {elapsed_time:.2f}초)"
                        )
                        break
                    else:
                        # 카메라인 경우 재시도
                        print("⚠️  프레임을 읽을 수 없습니다. 재시도 중...")
                        for retry in range(3):
                            time.sleep(0.5)
                            ret, frame = cap.read()
                            if ret:
                                break
                        if not ret:
                            print("❌ 프레임 읽기 실패.")
                            break

                frame_count += 1

                # 비디오 파일인 경우 진행 상황 표시 (100프레임마다)
                if is_video_file and total_frames and frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    estimated_total = elapsed_time * total_frames / frame_count if frame_count > 0 else 0
                    remaining = max(0, estimated_total - elapsed_time)
                    print(
                        f"📊 진행 상황: {frame_count}/{total_frames} 프레임 ({progress:.1f}%) - 예상 남은 시간: {remaining:.1f}초"
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # YOLO 검출
                if self.use_obb:
                    results = self.yolo_model.predict(frame_rgb, conf=self.conf_threshold, verbose=False, task="obb")[0]
                else:
                    results = self.yolo_model.predict(frame_rgb, conf=self.conf_threshold, verbose=False)[0]

                # [수정됨] 검출 결과 확인 로직 (OBB 우선 순위 적용)
                boxes = None
                # 1. OBB 모드이고 OBB 결과가 있다면 그것을 우선순위로 가져옴
                if self.use_obb and hasattr(results, "obb") and results.obb is not None:
                    boxes = results.obb
                # 2. 그 외의 경우 일반 boxes를 가져옴
                elif hasattr(results, "boxes"):
                    boxes = results.boxes

                # 화면에 표시
                display_frame = self._draw_detections(frame.copy(), boxes)

                # 검출 전용 모드인 경우 YOLO 검출만 수행 (조건 확인, 타이머, 검사 없음)
                if self.detect_only:
                    # 검출된 객체 개수 표시
                    num_detections = len(boxes) if boxes is not None else 0
                    info_text = f"Detections: {num_detections}"
                    cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # 비디오 파일인 경우 모든 프레임을 출력 비디오에 저장
                    if is_video_file and video_writer and video_writer.isOpened():
                        video_writer.write(display_frame)

                    cv2.imshow("Real-time Inspection", display_frame)

                    # 비디오 파일인 경우 FPS에 맞춰 지연 시간 추가
                    if is_video_file and video_fps:
                        delay_ms = max(1, int(1000.0 / video_fps))
                        key = cv2.waitKey(delay_ms) & 0xFF
                    else:
                        key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("\n사용자가 종료함")
                        break
                    continue  # YOLO 검출만 하고 계속 진행 (조건 확인, 타이머, 검사 건너뜀)

                # 일반 모드: 조건 확인 및 검사 수행
                # 비디오 파일인 경우 모든 프레임을 출력 비디오에 저장
                if is_video_file and video_writer and video_writer.isOpened():
                    video_writer.write(display_frame)

                # 조건 확인
                condition_satisfied, detections = self._check_condition(boxes)

                # 조건 만족 여부에 따른 처리
                if condition_satisfied:
                    if not self.condition_met:
                        self.condition_met = True
                        self.condition_start_time = time.time()
                        self.condition_start_frame = frame_count if is_video_file else None
                        print("✓ 조건 만족! 타이머 시작...")

                    # 비디오 파일인 경우 프레임 기반 타이머, 카메라는 시간 기반 타이머
                    if is_video_file and video_fps and self.condition_start_frame is not None:
                        frames_elapsed = frame_count - self.condition_start_frame
                        required_frames = int(self.required_duration * video_fps)
                        elapsed = frames_elapsed / video_fps
                        timer_text = f"Timer: {elapsed:.1f}s / {self.required_duration}s ({frames_elapsed}/{required_frames} frames)"
                        should_inspect = frames_elapsed >= required_frames
                    else:
                        elapsed = time.time() - self.condition_start_time
                        timer_text = f"Timer: {elapsed:.1f}s / {self.required_duration}s"
                        should_inspect = elapsed >= self.required_duration

                    self.last_valid_frame = frame.copy()
                    self.last_valid_detections = detections

                    cv2.putText(display_frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if should_inspect:
                        print(f"\n{'=' * 60}")
                        print(f"📸 조건이 {self.required_duration}초 이상 유지됨! 검사 시작...")
                        print(f"{'=' * 60}\n")

                        # 비디오 파일인 경우 검사 후에도 계속 진행
                        if is_video_file:
                            # 검사 수행 (화면은 닫지 않음)
                            self._perform_inspection(self.last_valid_frame.copy(), self.last_valid_detections)
                            # 타이머 리셋하여 다음 조건 만족 시에도 검사 가능
                            self.condition_met = False
                            self.condition_start_time = None
                            self.condition_start_frame = None
                            self.last_valid_frame = None
                            self.last_valid_detections = None
                            print(
                                f"📹 비디오 파일 처리 계속 진행 중... (프레임 {frame_count}/{total_frames if total_frames else '?'})\n"
                            )
                        else:
                            # 카메라인 경우 검사 후 종료
                            cap.release()
                            cv2.destroyAllWindows()
                            self._perform_inspection(self.last_valid_frame, self.last_valid_detections)
                            return
                else:
                    if self.condition_met:
                        print("⚠️  조건 해제됨. 타이머 리셋.")
                        self.condition_met = False
                        self.condition_start_time = None
                        self.condition_start_frame = None
                        self.last_valid_frame = None
                        self.last_valid_detections = None

                    status_text = "Waiting for condition..."
                    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Real-time Inspection", display_frame)

                # 비디오 파일인 경우 FPS에 맞춰 지연 시간 추가
                if is_video_file and video_fps:
                    delay_ms = max(1, int(1000.0 / video_fps))
                    key = cv2.waitKey(delay_ms) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\n사용자가 종료함")
                    break

        finally:
            cap.release()
            if video_writer and video_writer.isOpened():
                video_writer.release()
                if output_video_path:
                    # 저장된 파일 크기 확인
                    import os

                    if os.path.exists(output_video_path):
                        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                        print(f"\n{'=' * 60}")
                        print(f"💾 비디오 저장 완료: {output_video_path}")
                        print(f"   파일 크기: {file_size:.2f} MB")
                        print(f"   총 프레임: {frame_count} 프레임")
                        if total_frames:
                            print(f"   원본 프레임: {total_frames} 프레임")
                        print(f"{'=' * 60}\n")
                    else:
                        print(f"\n⚠️  비디오 파일이 저장되지 않았습니다: {output_video_path}\n")
            elif is_video_file:
                print("\n⚠️  비디오 Writer가 초기화되지 않아 비디오가 저장되지 않았습니다.\n")
            cv2.destroyAllWindows()

    def _check_condition(self, boxes):
        """조건 확인."""
        if boxes is None:
            if self.mode == "frontdoor":
                return False, {"high": [], "mid": [], "low": []}
            else:  # bolt
                return False, {"bolts": [], "frames": []}

        if self.mode == "frontdoor":
            return self._check_frontdoor_condition(boxes)
        else:  # bolt
            return self._check_bolt_condition(boxes)

    def _check_frontdoor_condition(self, boxes):
        """프론트도어 조건 확인."""
        detections = {"high": [], "mid": [], "low": []}

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # [수정됨] OBB/일반 박스 자동 판별 (IndexError 방지)
            bbox = None
            if self.use_obb and hasattr(box, "xyxyxyxy"):
                xyxyxyxy = box.xyxyxyxy[0].cpu().numpy().flatten()  # flatten()을 추가하여 (4,2) 형태를 (8,)로 강제 변환
                if len(xyxyxyxy) == 8:
                    bbox = xyxyxyxy  # 8개 점 그대로 사용
                else:
                    bbox = xyxyxyxy[:4]  # 4개면 일반 박스처럼 사용

            if bbox is None:  # 위에서 처리 안됐으면 일반 xyxy
                bbox = box.xyxy[0].cpu().numpy()

            class_name = self.yolo_model.names[cls_id].lower()
            if class_name in detections:
                detections[class_name].append({"bbox": bbox, "conf": conf, "cls_id": cls_id})

        has_all_three = len(detections["high"]) == 1 and len(detections["mid"]) == 1 and len(detections["low"]) == 1
        has_high_low = len(detections["high"]) == 1 and len(detections["low"]) == 1 and len(detections["mid"]) == 0

        condition_met = has_all_three or has_high_low

        return condition_met, detections

    def _check_bolt_condition(self, boxes):
        """볼트 조건 확인."""
        bolt_detections = []
        frame_detections = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # [수정됨] OBB/일반 박스 자동 판별 및 중심점 계산 (IndexError 방지)
            bbox = None
            center = None

            if self.use_obb and hasattr(box, "xyxyxyxy"):
                xyxyxyxy = (
                    box.xyxyxyxy[0].cpu().numpy().flatten()
                )  # flatten()을 추가하여 (4,2) �形태를 (8,)로 강제 변환
                if len(xyxyxyxy) == 8:
                    # 진정한 OBB
                    center = [xyxyxyxy[::2].mean(), xyxyxyxy[1::2].mean()]
                    bbox = xyxyxyxy
                else:
                    # 무늬만 OBB (실제론 4좌표)
                    bbox = xyxyxyxy[:4]
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

            if bbox is None:  # 일반 xyxy
                bbox = box.xyxy[0].cpu().numpy()
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

            detection = {"class_id": cls_id, "bbox": bbox, "conf": conf, "center": center}

            if cls_id in [0, 1]:  # 볼트
                bolt_detections.append(detection)
            elif cls_id in [2, 3, 4, 5, 6, 7]:  # 프레임
                frame_detections.append(detection)

        condition_met = len(frame_detections) == 1

        detections = {"bolts": bolt_detections, "frames": frame_detections}

        return condition_met, detections

    def _draw_detections(self, frame, boxes):
        """검출 결과를 프레임에 그리기."""
        if boxes is None:
            return frame

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = self.yolo_model.names[cls_id]

            if self.mode == "frontdoor":
                color = (0, 255, 0) if class_name.lower() in ["high", "mid", "low"] else (128, 128, 128)
            else:  # bolt
                if cls_id in [0, 1]:
                    color = (255, 0, 0)
                elif cls_id in [2, 3, 4, 5, 6, 7]:
                    color = (0, 255, 0)
                else:
                    color = (128, 128, 128)

            # OBB 데이터 길이 확인 후 분기 처리 (IndexError 방지)
            is_obb_drawn = False
            if self.use_obb and hasattr(box, "xyxyxyxy"):
                xyxyxyxy = box.xyxyxyxy[0].cpu().numpy().flatten()  # flatten()을 추가하여 (4,2) 형태를 (8,)로 강제 변환

                # 데이터가 8개(점 4개)인 경우에만 OBB로 그리기
                if len(xyxyxyxy) == 8:
                    points = np.array(
                        [
                            [xyxyxyxy[0], xyxyxyxy[1]],
                            [xyxyxyxy[2], xyxyxyxy[3]],
                            [xyxyxyxy[4], xyxyxyxy[5]],
                            [xyxyxyxy[6], xyxyxyxy[7]],
                        ],
                        dtype=np.int32,
                    )
                    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
                    x1, y1 = int(points[0][0]), int(points[0][1])
                    is_obb_drawn = True
                else:
                    # OBB 모드지만 데이터가 4개라면 일반 박스로 취급
                    x1, y1, x2, y2 = map(int, xyxyxyxy[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    is_obb_drawn = False

            if not is_obb_drawn:
                # 일반 박스 (OBB 실패했거나 모드 아닐 때)
                if hasattr(box, "xyxy"):
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    continue  # 좌표 정보 없으면 건너뜀

            # 라벨 그리기
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(y1, label_size[1] + 10)
            cv2.rectangle(frame, (x1, y_label - label_size[1] - 10), (x1 + label_size[0], y_label), color, -1)
            cv2.putText(frame, label, (x1, y_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def _perform_inspection(self, frame, detections):
        """검사 수행."""
        if self.mode == "frontdoor":
            self._inspect_frontdoor(frame, detections)
        else:  # bolt
            self._inspect_bolt(frame, detections)

    def _inspect_frontdoor(self, frame, detections):
        """프론트도어 검사."""
        print("🔍 프론트도어 검사 중...\n")

        part_results = {}
        parts_to_process = []

        if len(detections["high"]) == 1 and len(detections["mid"]) == 1 and len(detections["low"]) == 1:
            parts_to_process = ["high", "mid", "low"]
        elif len(detections["high"]) == 1 and len(detections["low"]) == 1 and len(detections["mid"]) == 0:
            parts_to_process = ["high", "low"]

        for part in parts_to_process:
            if len(detections[part]) > 0:
                bbox = detections[part][0]["bbox"]

                # OBB 모드인 경우 회전된 객체 crop
                # bbox가 8개 좌표를 가진 진짜 OBB일때만 회전 크롭 시도
                if self.use_obb and len(bbox) == 8:
                    cropped = self._crop_obb_object(frame, bbox)
                else:
                    x1, y1, x2, y2 = map(int, bbox[:4])  # 4개만 사용
                    cropped = frame[y1:y2, x1:x2]

                if cropped is None or cropped.size == 0:
                    print(f"  [{part.upper()}] 크롭 실패")
                    continue

                # 크롭 이미지 저장 (디버깅용)
                if self.debug and self.debug_crop_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함
                    crop_filename = f"{self.debug_crop_dir}/frontdoor_{part}_{timestamp}.jpg"
                    cv2.imwrite(crop_filename, cropped)
                    print(
                        f"  [{part.upper()}] 크롭 이미지 저장: {crop_filename} (크기: {cropped.shape[1]}x{cropped.shape[0]})"
                    )

                result = self._classify_with_dino(cropped, part)
                part_results[part] = result

                if result["num_classes"] == 4:
                    result_text = "양품" if not result["is_defect"] else f"불량(클래스 {result['pred_class']})"
                    conf_display = result["confidence"][result["pred_class"]]
                else:
                    result_text = "양품" if not result["is_defect"] else "불량"
                    conf_display = result["confidence"][result["pred_class"]]

                print(f"  [{part.upper()}] {result_text} (신뢰도: {conf_display:.2%})")

        print(f"\n📊 최종 판정 ({self.voting_method.upper()} Voting):")
        if self.voting_method == "hard":
            final_result = self._hard_voting(part_results)
        else:
            final_result = self._soft_voting(part_results)

        print(f"  결과: {'✅ 양품' if final_result == 'good' else '❌ 불량'}")
        print(f"\n{'=' * 60}\n")

    def _inspect_bolt(self, frame, detections):
        """볼트 검사."""
        print("🔍 볼트 검사 중...\n")

        frame_obj = detections["frames"][0]
        frame_bbox = frame_obj["bbox"]
        frame_cls = frame_obj["class_id"]

        frame_name = self.bolt_class_names.get(frame_cls, "unknown")
        print(f"  프레임 타입: {frame_name}")

        bolts_in_frame = []
        for bolt in detections["bolts"]:
            cx, cy = bolt["center"]
            if self.use_obb and len(frame_bbox) == 8:
                if self._point_in_obb(cx, cy, frame_bbox):
                    bolts_in_frame.append(bolt)
            else:
                # 일반 bbox (또는 4좌표 OBB)
                if frame_bbox[0] <= cx <= frame_bbox[2] and frame_bbox[1] <= cy <= frame_bbox[3]:
                    bolts_in_frame.append(bolt)

        print(f"  프레임 내 볼트 개수: {len(bolts_in_frame)}")

        if frame_cls in [2, 3, 4]:
            if len(bolts_in_frame) != 2:
                print("\n📊 최종 판정:")
                print(f"  결과: ❌ 불량 (볼트 개수 불일치: {len(bolts_in_frame)}/2)")
                print(f"\n{'=' * 60}\n")
                return

        if len(bolts_in_frame) == 0:
            print("\n📊 최종 판정:")
            print("  결과: ❌ 불량 (프레임 내 볼트 없음)")
            print(f"\n{'=' * 60}\n")
            return

        print("\n  볼트별 검사:")
        bolt_results = []
        for i, bolt in enumerate(bolts_in_frame):
            bbox = bolt["bbox"]

            if self.use_obb and len(bbox) == 8:
                cropped = self._crop_obb_object(frame, bbox)
            else:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cropped = frame[y1:y2, x1:x2]

            if cropped is None or cropped.size == 0:
                print(f"    볼트 #{i + 1}: 크롭 실패")
                continue

            # 크롭 이미지 저장 (디버깅용)
            if self.debug and self.debug_crop_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                crop_filename = f"{self.debug_crop_dir}/bolt_{i + 1}_{frame_name}_{timestamp}.jpg"
                cv2.imwrite(crop_filename, cropped)
                print(
                    f"    볼트 #{i + 1}: 크롭 이미지 저장: {crop_filename} (크기: {cropped.shape[1]}x{cropped.shape[0]})"
                )

            result = self._classify_with_dino(cropped, "bolt")
            bolt_results.append(result)

            result_text = "양품" if not result["is_defect"] else "불량"
            conf_display = result["confidence"][result["pred_class"]]

            print(f"    볼트 #{i + 1}: {result_text} (신뢰도: {conf_display:.2%})")

        print(f"\n📊 최종 판정 ({self.voting_method.upper()} Voting):")
        if self.voting_method == "hard":
            final_result = self._hard_voting_bolt(bolt_results)
        else:
            final_result = self._soft_voting_bolt(bolt_results)

        print(f"  결과: {'✅ 양품' if final_result == 'good' else '❌ 불량'}")
        print(f"\n{'=' * 60}\n")

    def _classify_with_dino(self, cropped_img, part):
        """DINOv2로 분류."""
        is_bolt = part == "bolt"
        num_classes = 2 if is_bolt else self.dino_num_classes.get(part, 2)

        if cropped_img.size == 0:
            if num_classes == 4:
                confidence = [0.0, 0.0, 0.0, 1.0]
                defect_confidence = 1.0
                pred_class = 3
            else:
                confidence = [0.0, 1.0]
                defect_confidence = 1.0
                pred_class = 1
            return {
                "is_defect": True,
                "confidence": confidence,
                "pred_class": pred_class,
                "defect_confidence": defect_confidence,
                "num_classes": num_classes,
            }

        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.dino_models[part](img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0].cpu().numpy().tolist()

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

    def _hard_voting(self, part_results):
        has_defect = any(result["is_defect"] for result in part_results.values())
        return "defect" if has_defect else "good"

    def _soft_voting(self, part_results):
        if len(part_results) == 0:
            return "good"
        defect_confidences = [result["defect_confidence"] for result in part_results.values()]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences)
        return "defect" if avg_defect_conf >= 0.5 else "good"

    def _hard_voting_bolt(self, bolt_results):
        if len(bolt_results) == 0:
            return "good"
        has_defect = any(b["is_defect"] for b in bolt_results)
        return "defect" if has_defect else "good"

    def _soft_voting_bolt(self, bolt_results):
        if len(bolt_results) == 0:
            return "good"
        defect_confidences = [b["defect_confidence"] for b in bolt_results]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences)
        return "defect" if avg_defect_conf >= 0.5 else "good"

    def _point_in_obb(self, x, y, obb_points):
        if len(obb_points) != 8:
            return False
        points = [(obb_points[i], obb_points[i + 1]) for i in range(0, 8, 2)]
        n = len(points)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = points[i]
            xj, yj = points[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _compute_rotated_box_corners(self, cx, cy, w, h, angle):
        dx = w / 2.0
        dy = h / 2.0
        local_corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        c = cos(angle)
        s = sin(angle)
        corners = []
        for lx, ly in local_corners:
            rx = c * lx - s * ly + cx
            ry = s * lx + c * ly + cy
            corners.append((rx, ry))
        return corners

    def _correct_orientation_constrained(self, w, h, angle):
        pi = math.pi
        angle = (angle + pi) % (2 * pi) - pi
        if w >= h:
            if abs(angle) > pi / 2:
                angle -= pi
        else:
            if angle > 0:
                angle -= pi
            if angle < -pi + (pi / 4):
                angle += pi
        angle = (angle + pi) % (2 * pi) - pi
        return w, h, angle

    def _crop_obb_object(self, img, obb_points):
        if len(obb_points) != 8:
            return None
        img_h, img_w = img.shape[:2]
        points = np.array(
            [
                [obb_points[0], obb_points[1]],
                [obb_points[2], obb_points[3]],
                [obb_points[4], obb_points[5]],
                [obb_points[6], obb_points[7]],
            ],
            dtype=np.float32,
        )
        cx = points[:, 0].mean()
        cy = points[:, 1].mean()
        w = np.linalg.norm(points[1] - points[0])
        h = np.linalg.norm(points[2] - points[1])
        vx = points[1][0] - points[0][0]
        vy = points[1][1] - points[0][1]
        angle = math.atan2(vy, vx)
        w, h, angle = self._correct_orientation_constrained(w, h, angle)

        if abs(angle) < 1e-6:
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(img_w, int(cx + w / 2))
            y2 = min(img_h, int(cy + h / 2))
            if x1 >= x2 or y1 >= y2:
                return None
            crop = img[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            return crop_resized

        src_corners = self._compute_rotated_box_corners(cx, cy, w, h, angle)
        src_points = np.array(src_corners, dtype=np.float32)
        dst_corners = [(0, 0), (w, 0), (w, h), (0, h)]
        dst_points = np.array(dst_corners, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(
            img, M, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        return warped


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required_keys = ["mode", "yolo_model"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"설정 파일에 '{key}' 필드가 없습니다")
    return config


def main():
    parser = argparse.ArgumentParser(description="실시간 카메라 양불량 검사 시스템")
    parser.add_argument("--config", type=str, required=True, help="설정 YAML 파일 경로")
    parser.add_argument("--source", type=str, default="0", help="카메라 소스 (0: 웹캠, RTSP URL 등, 기본값: 0)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="디바이스 (기본값: cuda)")
    parser.add_argument("--obb", action="store_true", help="OBB(Oriented Bounding Box) 모드 사용")
    parser.add_argument("--debug", action="store_true", help="디버그 모드: 크롭 이미지를 debug_crops 폴더에 저장")
    parser.add_argument(
        "--detect-only", action="store_true", help="검출 전용 모드: YOLO 검출만 수행하고 검사는 하지 않음"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    mode = config["mode"].lower()
    yolo_model = config["yolo_model"]
    conf_threshold = config.get("conf_threshold", 0.25)
    dino_mode = config.get("dino_mode", "simple")

    dino_models = {}
    if mode == "frontdoor":
        dino_models = {"high": config["dino_high"], "mid": config["dino_mid"], "low": config["dino_low"]}
        voting_method = config.get("voting_method", "soft")
    else:
        dino_models = {"bolt": config["dino_bolt"]}
        voting_method = config.get("voting_method", "soft")

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    system = RealtimeInspectionSystem(
        mode=mode,
        yolo_model_path=yolo_model,
        dino_models=dino_models,
        device=args.device,
        conf_threshold=conf_threshold,
        voting_method=voting_method,
        use_obb=args.obb,
        debug=args.debug,
        detect_only=args.detect_only,
    )

    system.dino_mode = dino_mode
    system.run(source=source)


if __name__ == "__main__":
    main()
