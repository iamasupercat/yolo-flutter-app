// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../../models/models.dart';
import '../../services/model_manager.dart';
import '../../services/inspection_service.dart';
import '../../services/dino_client.dart';

/// Inspection result data class
class InspectionResult {
  final bool isGood;
  final String resultText;
  final double defectConfidence;
  final String? details;
  final DateTime timestamp;

  InspectionResult({
    required this.isGood,
    required this.resultText,
    this.defectConfidence = 0.0,
    this.details,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}

/// Controller that manages the state and business logic for camera inference
class CameraInferenceController extends ChangeNotifier {
  // Detection state
  int _detectionCount = 0;
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  // Threshold state
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  SliderType _activeSlider = SliderType.none;

  // Model state
  ModelType _selectedModel = ModelType.bolt;
  bool _isModelLoading = false;
  String? _modelPath;
  String _loadingMessage = '';
  double _downloadProgress = 0.0;

  // Camera state
  double _currentZoomLevel = 1.0;
  LensFacing _lensFacing = LensFacing.front;
  bool _isFrontCamera = false;

  // Controllers
  final _yoloController = YOLOViewController();
  late final ModelManager _modelManager;
  late InspectionService _inspectionService;
  DINOClient? _dinoClient; // DINO 서버 클라이언트 (선택사항)

  // Camera freeze state (for condition-based inspection)
  bool _isCameraFrozen = false;
  Uint8List? _frozenFrame;
  List<YOLOResult>? _frozenDetections; // 정지된 프레임의 탐지 결과
  String? _frozenFramePath; // 정지된 프레임 이미지 파일 경로
  
  // Inspection result
  InspectionResult? _inspectionResult;

  // Performance optimization
  bool _isDisposed = false;
  Future<void>? _loadingFuture;

  // Getters
  int get detectionCount => _detectionCount;
  double get currentFps => _currentFps;
  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  SliderType get activeSlider => _activeSlider;
  ModelType get selectedModel => _selectedModel;
  bool get isModelLoading => _isModelLoading;
  String? get modelPath => _modelPath;
  String get loadingMessage => _loadingMessage;
  double get downloadProgress => _downloadProgress;
  double get currentZoomLevel => _currentZoomLevel;
  bool get isFrontCamera => _isFrontCamera;
  LensFacing get lensFacing => _lensFacing;
  YOLOViewController get yoloController => _yoloController;
  bool get isCameraFrozen => _isCameraFrozen;
  Uint8List? get frozenFrame => _frozenFrame;
  List<YOLOResult>? get frozenDetections => _frozenDetections; // 정지된 프레임의 YOLO 좌표
  String? get frozenFramePath => _frozenFramePath; // 정지된 프레임 이미지 파일 경로
  InspectionResult? get inspectionResult => _inspectionResult; // 검사 결과
  double? get elapsedTime => _inspectionService.getElapsedTime();

  CameraInferenceController() {
    _isFrontCamera = _lensFacing == LensFacing.front;

    _modelManager = ModelManager(
      onDownloadProgress: (progress) {
        _downloadProgress = progress;
        notifyListeners();
      },
      onStatusUpdate: (message) {
        _loadingMessage = message;
        notifyListeners();
      },
    );

    _inspectionService = InspectionService(
      modelType: _selectedModel,
      debug: false,
    );
    
    // DINO 서버 클라이언트 초기화 (기본값: PC IP 주소)
    // 실제 기기인 경우: http://192.168.0.198:5001 (포트 5000은 macOS ControlCenter가 사용 중)
    // Android 에뮬레이터인 경우: http://10.0.2.2:5001
    // 필요시 setDinoServerUrl()로 변경 가능
    setDinoServerUrl('http://192.168.0.198:5001');
  }
  
  /// DINO 서버 URL 설정 (정지 프레임을 서버로 전송하려면 설정)
  void setDinoServerUrl(String url) {
    _dinoClient = DINOClient(baseUrl: url);
    print('✅ DINO 서버 URL 설정: $url');
  }
  
  /// DINO 서버 클라이언트 가져오기
  DINOClient? get dinoClient => _dinoClient;

  /// Initialize the controller
  Future<void> initialize() async {
    await _loadModelForPlatform();
    _yoloController.setThresholds(
      confidenceThreshold: _confidenceThreshold,
      iouThreshold: _iouThreshold,
      numItemsThreshold: _numItemsThreshold,
    );
  }

  /// Handle detection results and calculate FPS
  void onDetectionResults(List<YOLOResult> results) {
    if (_isDisposed || _isCameraFrozen) return; // 카메라가 정지되면 처리하지 않음

    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsUpdate).inMilliseconds;

    if (elapsed >= 1000) {
      _currentFps = _frameCount * 1000 / elapsed;
      _frameCount = 0;
      _lastFpsUpdate = now;
    }

    if (_detectionCount != results.length) {
      _detectionCount = results.length;
      notifyListeners();
    }

    // 조건 확인 (live.py 참고)
    final conditionResult = _inspectionService.checkCondition(results);
    
    if (conditionResult['satisfied'] == true) {
      // 조건 만족 후 2초 지났는지 확인
      if (_inspectionService.shouldInspect()) {
        // 카메라 정지 및 프레임 캡처 (마지막 탐지 결과와 함께)
        print('🔍 shouldInspect() = true, _freezeCameraAndCapture 호출 시작...');
        _freezeCameraAndCapture(results).catchError((error, stackTrace) {
          print('❌ _freezeCameraAndCapture 실행 중 오류: $error');
          print('  스택 트레이스: $stackTrace');
        });
      } else {
        // 타이머 진행 중 - UI 업데이트만
        notifyListeners();
      }
    }
  }

  /// Handle performance metrics
  void onPerformanceMetrics(double fps) {
    if (_isDisposed) return;

    if ((_currentFps - fps).abs() > 0.1) {
      _currentFps = fps;
      notifyListeners();
    }
  }

  void onZoomChanged(double zoomLevel) {
    if (_isDisposed) return;

    if ((_currentZoomLevel - zoomLevel).abs() > 0.01) {
      _currentZoomLevel = zoomLevel;
      notifyListeners();
    }
  }

  void toggleSlider(SliderType type) {
    if (_isDisposed) return;

    if (_activeSlider != type) {
      _activeSlider = _activeSlider == type ? SliderType.none : type;
      notifyListeners();
    }
  }

  void updateSliderValue(double value) {
    if (_isDisposed) return;

    bool changed = false;
    switch (_activeSlider) {
      case SliderType.numItems:
        final newValue = value.toInt();
        if (_numItemsThreshold != newValue) {
          _numItemsThreshold = newValue;
          _yoloController.setNumItemsThreshold(_numItemsThreshold);
          changed = true;
        }
        break;
      case SliderType.confidence:
        if ((_confidenceThreshold - value).abs() > 0.01) {
          _confidenceThreshold = value;
          _yoloController.setConfidenceThreshold(value);
          changed = true;
        }
        break;
      case SliderType.iou:
        if ((_iouThreshold - value).abs() > 0.01) {
          _iouThreshold = value;
          _yoloController.setIoUThreshold(value);
          changed = true;
        }
        break;
      default:
        break;
    }

    if (changed) {
      notifyListeners();
    }
  }

  void setZoomLevel(double zoomLevel) {
    if (_isDisposed) return;

    if ((_currentZoomLevel - zoomLevel).abs() > 0.01) {
      _currentZoomLevel = zoomLevel;
      _yoloController.setZoomLevel(zoomLevel);
      notifyListeners();
    }
  }

  void flipCamera() {
    if (_isDisposed) return;

    _isFrontCamera = !_isFrontCamera;
    _lensFacing = _isFrontCamera ? LensFacing.front : LensFacing.back;
    if (_isFrontCamera) _currentZoomLevel = 1.0;
    _yoloController.switchCamera();
    notifyListeners();
  }

  void setLensFacing(LensFacing facing) {
    if (_isDisposed) return;

    if (_lensFacing != facing) {
      _lensFacing = facing;
      _isFrontCamera = facing == LensFacing.front;

      _yoloController.switchCamera();

      if (_isFrontCamera) {
        _currentZoomLevel = 1.0;
      }

      notifyListeners();
    }
  }

  void changeModel(ModelType model) {
    if (_isDisposed) return;

    if (!_isModelLoading && model != _selectedModel) {
      _selectedModel = model;
      _inspectionService = InspectionService(
        modelType: model,
        debug: false,
      );
      _isCameraFrozen = false;
      _frozenFrame = null;
      _frozenDetections = null;
      _frozenFramePath = null;
      _loadModelForPlatform();
    }
  }

  /// 카메라 정지 및 프레임 캡처 (live.py의 검사 시작 시점과 유사)
  /// [lastResults] 마지막 탐지 결과 (YOLO 좌표 포함)
  Future<void> _freezeCameraAndCapture(List<YOLOResult> lastResults) async {
    if (_isCameraFrozen) {
      print('⚠️  이미 카메라가 정지되어 있음, _freezeCameraAndCapture 건너뜀');
      return; // 이미 정지된 경우 중복 실행 방지
    }

    // 먼저 카메라 정지 상태로 설정하여 중복 호출 방지
    _isCameraFrozen = true;
    notifyListeners();

    print('\n${'='*60}');
    print('📸 조건이 ${InspectionService.requiredDuration}초 이상 유지됨! 카메라 정지...');
    print('${'='*60}\n');

    try {
      print('📋 1단계: 탐지 결과 저장 시작...');
      // 먼저 마지막 탐지 결과 저장 (YOLO 좌표 포함) - 서버 전송 전에 필요
      _frozenDetections = List.from(lastResults);
      print('✅ 탐지 결과 저장 완료: ${_frozenDetections!.length}개 객체');
      
      // 각 탐지 결과의 좌표 정보 출력 (디버깅용)
      for (int i = 0; i < _frozenDetections!.length; i++) {
        final result = _frozenDetections![i];
        print('  객체 #${i + 1}:');
        print('    - 클래스: ${result.className} (인덱스: ${result.classIndex})');
        print('    - 신뢰도: ${(result.confidence * 100).toStringAsFixed(1)}%');
        print('    - 픽셀 좌표: left=${result.boundingBox.left.toStringAsFixed(1)}, top=${result.boundingBox.top.toStringAsFixed(1)}, right=${result.boundingBox.right.toStringAsFixed(1)}, bottom=${result.boundingBox.bottom.toStringAsFixed(1)}');
        print('    - 정규화 좌표: left=${result.normalizedBox.left.toStringAsFixed(3)}, top=${result.normalizedBox.top.toStringAsFixed(3)}, right=${result.normalizedBox.right.toStringAsFixed(3)}, bottom=${result.normalizedBox.bottom.toStringAsFixed(3)}');
      }
      
      print('📋 2단계: 프레임 캡처 시작...');
      // 현재 프레임 캡처 (카메라가 닫히기 전에!)
      final frameBytes = await _yoloController.captureFrame();
      if (frameBytes != null) {
        _frozenFrame = frameBytes;
        print('✅ 프레임 캡처 완료: ${frameBytes.length} bytes');
        
        print('📋 3단계: 정지된 프레임 로컬 저장 시작...');
        // 정지된 프레임을 파일로 저장
        _frozenFramePath = await _saveFrozenFrame(frameBytes);
        if (_frozenFramePath != null) {
          print('✅ 정지된 프레임 저장 완료: $_frozenFramePath');
        } else {
          print('⚠️  정지된 프레임 저장 실패');
        }
        
        print('📋 4단계: DINO 서버로 전송 시작...');
        // DINO 서버가 설정되어 있으면 서버로도 전송 (이미지 + YOLO 좌표)
        if (_dinoClient != null) {
          // 서버 상태 확인 및 자동 시작 시도
          final isServerRunning = await _dinoClient!.checkHealth();
          if (!isServerRunning) {
            print('⚠️  DINO 서버가 실행되지 않았습니다. 서버를 시작하세요:');
            print('     cd /Users/csj/yolo-flutter-app');
            print('     ./start_dino_server.sh');
            print('  또는 수동으로:');
            print('     python dino_server.py --port 5001 \\');
            print('       --bolt-model models/dino/BoltDINO.pt \\');
            print('       --door-high-model models/dino/DoorDINO_high.pt \\');
            print('       --door-mid-model models/dino/DoorDINO_mid.pt \\');
            print('       --door-low-model models/dino/DoorDINO_low.pt');
          } else {
            await _sendFrozenFrameToServer(frameBytes);
          }
        } else {
          print('⚠️  DINO 클라이언트가 null입니다. 서버 전송 건너뜀');
        }
      } else {
        print('⚠️  프레임 캡처 실패 (frameBytes == null)');
      }

      print('📋 5단계: 카메라 정지 시작...');
      // 카메라 정지 (프레임 캡처 후에!)
      await _yoloController.stop();
      print('✅ 카메라 정지 완료');
      notifyListeners();
      print('✅ 모든 단계 완료!');
    } catch (e, stackTrace) {
      print('❌ 카메라 정지 중 오류: $e');
      print('  스택 트레이스: $stackTrace');
      _isCameraFrozen = false;
      _frozenFrame = null;
      _frozenDetections = null;
      _frozenFramePath = null;
      notifyListeners();
    }
  }

  /// 카메라 재시작 (필요한 경우)
  Future<void> restartCamera() async {
      _isCameraFrozen = false;
      _frozenFrame = null;
      _frozenDetections = null;
      _frozenFramePath = null;
      _inspectionResult = null;
      _inspectionService.reset();
      await _yoloController.restartCamera();
      notifyListeners();
    }
  
  /// 검사 결과 저장
  void _saveInspectionResult(bool isGood, String resultText, String details, {double? defectConfidence}) {
    _inspectionResult = InspectionResult(
      isGood: isGood,
      resultText: resultText,
      defectConfidence: defectConfidence ?? 0.0,
      details: details,
      timestamp: DateTime.now(),
    );
    notifyListeners();
  }
  
  /// 검사 결과 초기화
  void clearInspectionResult() {
    _inspectionResult = null;
    notifyListeners();
  }

  /// 정지된 프레임 이미지를 파일로 저장
  Future<String?> _saveFrozenFrame(Uint8List frameBytes) async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().toIso8601String()
          .replaceAll(':', '-')
          .replaceAll('.', '-')
          .substring(0, 19); // YYYY-MM-DDTHH-MM-SS
      
      final filename = 'frozen_frame_$timestamp.jpg';
      final file = File('${directory.path}/$filename');
      await file.writeAsBytes(frameBytes);
      
      return file.path;
    } catch (e) {
      print('❌ 정지된 프레임 저장 중 오류: $e');
      return null;
    }
  }
  
  /// 정지된 프레임을 DINO 서버로 전송 (이미지 + YOLO 좌표)
  Future<void> _sendFrozenFrameToServer(Uint8List frameBytes) async {
    if (_dinoClient == null || _frozenDetections == null) return;
    
    try {
      final timestamp = DateTime.now().toIso8601String()
          .replaceAll(':', '-')
          .replaceAll('.', '-')
          .substring(0, 19);
      final filename = 'frozen_frame_$timestamp.jpg';
      
      // YOLOResult를 Map으로 변환 (정규화 좌표 포함)
      final detectionsList = _frozenDetections!.map((result) {
        return {
          'classIndex': result.classIndex,
          'className': result.className,
          'confidence': result.confidence,
          'boundingBox': {
            'left': result.boundingBox.left,
            'top': result.boundingBox.top,
            'right': result.boundingBox.right,
            'bottom': result.boundingBox.bottom,
          },
          'normalizedBox': {
            'left': result.normalizedBox.left,
            'top': result.normalizedBox.top,
            'right': result.normalizedBox.right,
            'bottom': result.normalizedBox.bottom,
          },
        };
      }).toList();
      
      // 모델 타입 결정
      final modelType = _selectedModel == ModelType.bolt ? 'bolt' : 'door';
      
      print('📤 정지 프레임과 YOLO 좌표를 서버로 전송 중...');
      final result = await _dinoClient!.saveFrame(
        frameBytes,
        detectionsList,
        modelType,
        filename: filename,
      );
      
      if (result != null && result['success'] == true) {
        print('✅ 서버 저장 완료: ${result['filepath']}');
        final croppedFiles = result['cropped_files'] as List<dynamic>?;
        if (croppedFiles != null && croppedFiles.isNotEmpty) {
          print('✅ 크롭된 이미지 ${croppedFiles.length}개 저장 완료');
          for (final file in croppedFiles) {
            print('  - $file');
          }
        }
        
        // DINO 분류 결과 출력
        final classificationResults = result['classification_results'] as List<dynamic>?;
        if (classificationResults != null && classificationResults.isNotEmpty) {
          print('\n📊 DINO 분류 결과:');
          for (final res in classificationResults) {
            final isDefect = res['is_defect'] as bool;
            final confidence = res['confidence'] as List<dynamic>;
            final predClass = res['pred_class'] as int;
            final defectConf = res['defect_confidence'] as double;
            final resultText = isDefect ? '불량' : '양품';
            final confDisplay = confidence[predClass] as double;
            
            if (_selectedModel == ModelType.bolt) {
              final boltIndex = res['bolt_index'] as int? ?? 0;
              final frameName = res['frame_name'] as String? ?? 'unknown';
              print('  볼트 #$boltIndex ($frameName): $resultText (신뢰도: ${(confDisplay * 100).toStringAsFixed(1)}%, 불량확률: ${(defectConf * 100).toStringAsFixed(1)}%)');
            } else {
              final part = res['part'] as String? ?? 'unknown';
              print('  도어 ${part.toUpperCase()}: $resultText (신뢰도: ${(confDisplay * 100).toStringAsFixed(1)}%, 불량확률: ${(defectConf * 100).toStringAsFixed(1)}%)');
            }
          }
        }
        
        // 최종 판정 결과
        final finalResult = result['final_result'] as Map<String, dynamic>?;
        if (finalResult != null) {
          final isGood = finalResult['is_good'] as bool;
          final resultText = finalResult['result_text'] as String;
          final avgDefectConf = finalResult['avg_defect_confidence'] as double;
          final votingMethod = finalResult['voting_method'] as String;
          
          print('\n📊 최종 판정 (${votingMethod.toUpperCase()} Voting):');
          print('  평균 불량 확률: ${(avgDefectConf * 100).toStringAsFixed(1)}%');
          print('  결과: ${isGood ? '✅ 양품' : '❌ 불량'}');
          
          // UI에 최종 결과 표시
          _saveInspectionResult(
            isGood,
            resultText,
            "평균 불량 확률: ${(avgDefectConf * 100).toStringAsFixed(1)}%",
            defectConfidence: avgDefectConf,
          );
        }
      } else {
        print('⚠️  서버 저장 실패');
      }
    } catch (e, stackTrace) {
      print('❌ 서버 전송 중 오류: $e');
      print('  스택 트레이스: $stackTrace');
    }
  }

  /// 정지된 프레임 이미지에서 YOLO 좌표로 크롭
  /// 
  /// [detectionIndex] 크롭할 탐지 결과의 인덱스 (frozenDetections 리스트의 인덱스)
  /// [savePath] 크롭된 이미지를 저장할 경로 (null이면 자동 생성)
  /// 
  /// Returns: 크롭된 이미지 파일 경로 또는 null
  Future<String?> cropFrozenFrameByDetection({
    required int detectionIndex,
    String? savePath,
  }) async {
    if (_frozenFrame == null || _frozenDetections == null) {
      print('⚠️  정지된 프레임 또는 탐지 결과가 없습니다.');
      return null;
    }

    if (detectionIndex < 0 || detectionIndex >= _frozenDetections!.length) {
      print('⚠️  잘못된 탐지 인덱스: $detectionIndex');
      return null;
    }

    final detection = _frozenDetections![detectionIndex];
    final bbox = detection.boundingBox;
    
    // 바운딩 박스 좌표를 리스트로 변환 [x1, y1, x2, y2]
    final bboxList = [
      bbox.left,
      bbox.top,
      bbox.right,
      bbox.bottom,
    ];

    // InspectionService의 cropImage 메서드 사용
    final croppedBytes = await _inspectionService.cropImage(
      _frozenFrame!,
      bboxList,
      debugLabel: '${detection.className}_$detectionIndex',
    );

    if (croppedBytes == null) {
      print('⚠️  이미지 크롭 실패');
      return null;
    }

    // 크롭된 이미지 저장
    try {
      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().toIso8601String()
          .replaceAll(':', '-')
          .replaceAll('.', '-')
          .substring(0, 19);
      
      final filename = savePath ?? 
          'cropped_${detection.className}_${detectionIndex}_$timestamp.png';
      final file = File('${directory.path}/$filename');
      await file.writeAsBytes(croppedBytes);
      
      print('✅ 크롭된 이미지 저장 완료: ${file.path}');
      return file.path;
    } catch (e) {
      print('❌ 크롭된 이미지 저장 중 오류: $e');
      return null;
    }
  }

  /// 정지된 프레임의 모든 탐지 결과를 크롭하여 저장
  /// 
  /// Returns: 크롭된 이미지 파일 경로 리스트
  Future<List<String>> cropAllFrozenDetections() async {
    if (_frozenDetections == null || _frozenDetections!.isEmpty) {
      print('⚠️  탐지 결과가 없습니다.');
      return [];
    }

    final croppedPaths = <String>[];
    
    for (int i = 0; i < _frozenDetections!.length; i++) {
      final path = await cropFrozenFrameByDetection(detectionIndex: i);
      if (path != null) {
        croppedPaths.add(path);
      }
    }

    print('✅ 총 ${croppedPaths.length}개 이미지 크롭 완료');
    return croppedPaths;
  }

  Future<void> _loadModelForPlatform() async {
    if (_isDisposed) return;

    if (_loadingFuture != null) {
      await _loadingFuture;
      return;
    }

    _loadingFuture = _performModelLoading();
    try {
      await _loadingFuture;
    } finally {
      _loadingFuture = null;
    }
  }

  Future<void> _performModelLoading() async {
    if (_isDisposed) return;

    _isModelLoading = true;
    _loadingMessage = 'Loading ${_selectedModel.modelName} model...';
    _downloadProgress = 0.0;
    _detectionCount = 0;
    _currentFps = 0.0;
    notifyListeners();

    try {
      final modelPath = await _modelManager.getModelPath(_selectedModel);

      if (_isDisposed) return;

      _modelPath = modelPath;
      _isModelLoading = false;
      _loadingMessage = '';
      _downloadProgress = 0.0;
      notifyListeners();

      if (modelPath == null) {
        throw Exception('Failed to load ${_selectedModel.modelName} model');
      }
    } catch (e) {
      if (_isDisposed) return;

      final error = YOLOErrorHandler.handleError(
        e,
        'Failed to load model ${_selectedModel.modelName} for task ${_selectedModel.task.name}',
      );

      _isModelLoading = false;
      _loadingMessage = 'Failed to load model: ${error.message}';
      _downloadProgress = 0.0;
      notifyListeners();
      rethrow;
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    super.dispose();
  }
}
