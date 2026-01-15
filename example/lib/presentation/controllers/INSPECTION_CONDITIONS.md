# 검사 조건 설명

## 조건 만족 시간
- 조건이 만족되면 **2초 동안 유지**되어야 검사가 시작됩니다.

---

## 볼트 모델 조건

### 조건
**프레임이 정확히 1개 감지되어야 함**

### 설명
- YOLO가 탐지하는 객체:
  - **볼트**: 클래스 ID 0, 1 (bolt_frontside, bolt_side)
  - **프레임**: 클래스 ID 2-7 (sedan trunklid, suv trunklid, hood, frontfender 등)

### 조건 만족 기준
```
프레임 감지 개수 == 1
```

### 예시
- ✅ 조건 만족: 프레임 1개 감지 (볼트는 개수 무관)
- ❌ 조건 불만족: 프레임 0개 또는 2개 이상

### 검사 프로세스
1. 프레임 1개 감지 → 조건 만족
2. 2초 동안 유지
3. 프레임 내에 있는 볼트들을 찾음
4. 각 볼트를 크롭하여 DINO로 분류
5. 최종 판정 (Soft/Hard Voting)

---

## 도어 모델 조건

### 조건
**다음 중 하나를 만족해야 함:**

#### 옵션 1: 모든 파트 감지
```
high == 1개 AND mid == 1개 AND low == 1개
```

#### 옵션 2: High + Low만 감지
```
high == 1개 AND low == 1개 AND mid == 0개
```

### 설명
- YOLO가 탐지하는 객체:
  - **high**: 도어 상단 부분
  - **mid**: 도어 중간 부분
  - **low**: 도어 하단 부분

### 조건 만족 기준
```
(high == 1 AND mid == 1 AND low == 1) 
OR 
(high == 1 AND low == 1 AND mid == 0)
```

### 예시
- ✅ 조건 만족: high 1개 + mid 1개 + low 1개
- ✅ 조건 만족: high 1개 + low 1개 (mid 없음)
- ❌ 조건 불만족: high만 1개, 또는 low만 1개

### 검사 프로세스
1. 조건 만족 (high/mid/low 또는 high/low)
2. 2초 동안 유지
3. 각 파트를 크롭하여 DINO로 분류
4. 최종 판정 (Soft/Hard Voting)

---

## Voting 방법

### Soft Voting (기본값)
- **방식**: 평균 불량 확률이 0.5 이상이면 불량
- **계산**: 모든 검사 결과의 불량 확률을 평균냄
- **예시**: 
  - 볼트 2개 검사 → 불량 확률 [0.3, 0.7] → 평균 0.5 → **불량**
  - 볼트 2개 검사 → 불량 확률 [0.2, 0.4] → 평균 0.3 → **양품**

### Hard Voting
- **방식**: 하나라도 불량이면 불량
- **계산**: 모든 검사 결과 중 하나라도 `is_defect == true`면 불량
- **예시**:
  - 볼트 2개 검사 → [양품, 불량] → **불량**
  - 볼트 2개 검사 → [양품, 양품] → **양품**

---

## Voting 방법 변경 (개발자 전용)

**주의**: Voting 방법은 개발자가 코드에서만 설정할 수 있습니다. 사용자가 앱에서 선택할 수 있는 기능이 아닙니다.

### 기본값
- **기본값**: `VotingMethod.soft` (Soft Voting)
- **설정 위치**: `example/lib/presentation/controllers/camera_inference_controller.dart` 79번째 줄

### 변경 방법 1: Controller 초기화 후 설정
`example/lib/presentation/screens/camera_inference_screen.dart` 파일에서:

```dart
@override
void initState() {
  super.initState();
  _controller = CameraInferenceController();
  
  // Hard Voting으로 변경하려면 여기에 추가
  _controller.setVotingMethod(VotingMethod.hard);
  
  _controller.initialize().catchError((error) {
    // ...
  });
}
```

### 변경 방법 2: Controller 생성자에서 기본값 변경
`example/lib/presentation/controllers/camera_inference_controller.dart` 파일에서:

```dart
// 79번째 줄 근처
// Voting method (기본값: soft)
VotingMethod _votingMethod = VotingMethod.hard;  // 기본값을 hard로 변경
```

### 변경 위치 요약
- **파일**: `example/lib/presentation/controllers/camera_inference_controller.dart`
- **메서드**: `setVotingMethod(VotingMethod method)` (128번째 줄 근처)
- **기본값**: `VotingMethod.soft` (79번째 줄)

