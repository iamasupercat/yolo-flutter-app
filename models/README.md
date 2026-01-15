# DINO 모델 파일 위치

이 폴더에 DINO 모델 파일을 저장하세요.

## 모델 파일 구조

```
models/dino/
├── BoltDINO.pt          # 볼트 DINO 모델
├── DoorDINO_high.pt     # 도어 High DINO 모델
├── DoorDINO_mid.pt      # 도어 Mid DINO 모델
└── DoorDINO_low.pt      # 도어 Low DINO 모델
```

## 사용 방법

DINO 서버는 YOLO 탐지 후 크롭된 이미지를 분류하기 위해 실행됩니다.

### Flutter 앱 실행

Flutter 앱이 실행되면:
1. YOLO로 실시간 탐지
2. 조건 만족 시 프레임 캡처
3. 바운딩 박스 좌표로 이미지 크롭
4. 크롭된 이미지를 DINO 서버로 전송
5. DINO 분류 결과로 최종 판정

### 3. Flutter 앱에서 서버 URL 설정

```dart
// Android 에뮬레이터인 경우
controller.setDinoServerUrl('http://10.0.2.2:5000');

// 실제 기기인 경우 (PC의 IP 주소)
controller.setDinoServerUrl('http://192.168.0.100:5000');
```

