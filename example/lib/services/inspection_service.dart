// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path_provider_android/path_provider_android.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import '../../models/models.dart';

/// Inspection service that handles condition checking and image processing
/// Similar to live.py's RealtimeInspectionSystem logic
class InspectionService {
  final ModelType modelType;
  final bool debug;
  String? _debugCropDir;
  
  // Condition tracking (similar to live.py)
  DateTime? _conditionStartTime;
  bool _conditionMet = false;
  static const double requiredDuration = 2.0; // 2 seconds
  
  InspectionService({
    required this.modelType,
    this.debug = false,
  });
  
  /// Initialize debug directory (similar to live.py)
  /// Must be called before using debug features
  /// 
  /// PC에서 볼 수 있도록 외부 저장소(Download 폴더)에 debug_crop 폴더 생성
  Future<void> initializeDebugDir() async {
    if (!debug) return;
    
    try {
      if (Platform.isAndroid) {
        // Android: 외부 저장소의 Download 폴더 사용 (PC에서 볼 수 있음)
        _debugCropDir = '/storage/emulated/0/Download/debug_crop';
      } else if (Platform.isIOS) {
        // iOS는 Documents 폴더 사용
        final directory = await getApplicationDocumentsDirectory();
        _debugCropDir = '${directory.path}/debug_crop';
      } else {
        // 기타 플랫폼
        final directory = await getApplicationDocumentsDirectory();
        _debugCropDir = '${directory.path}/debug_crop';
      }
      
      final dir = Directory(_debugCropDir!);
      if (!await dir.exists()) {
        await dir.create(recursive: true);
      }
      
      // 하위 폴더 생성 (bolt, door)
      final boltDir = Directory('$_debugCropDir/bolt');
      final doorDir = Directory('$_debugCropDir/door');
      if (!await boltDir.exists()) {
        await boltDir.create(recursive: true);
      }
      if (!await doorDir.exists()) {
        await doorDir.create(recursive: true);
      }
      
      print('  - 디버그 크롭 이미지 저장 경로: $_debugCropDir/');
      print('    - 볼트: $_debugCropDir/bolt/');
      print('    - 도어: $_debugCropDir/door/');
    } catch (e) {
      print('⚠️  디버그 폴더 생성 실패: $e');
      // 실패 시 앱 내부 저장소로 폴백
      try {
        final directory = await getApplicationDocumentsDirectory();
        _debugCropDir = '${directory.path}/debug_crop';
        final dir = Directory(_debugCropDir!);
        if (!await dir.exists()) {
          await dir.create(recursive: true);
        }
        final boltDir = Directory('$_debugCropDir/bolt');
        final doorDir = Directory('$_debugCropDir/door');
        if (!await boltDir.exists()) {
          await boltDir.create(recursive: true);
        }
        if (!await doorDir.exists()) {
          await doorDir.create(recursive: true);
        }
        print('  - 폴백: 앱 내부 저장소 사용: $_debugCropDir/');
      } catch (e2) {
        print('⚠️  폴백도 실패: $e2');
        _debugCropDir = null;
      }
    }
  }
  
  /// Check if condition is satisfied (similar to live.py's _check_condition)
  /// Returns: (conditionSatisfied, detections)
  Map<String, dynamic> checkCondition(List<YOLOResult> results) {
    if (results.isEmpty) {
      _resetCondition();
      return {
        'satisfied': false,
        'detections': _getEmptyDetections(),
      };
    }
    
    bool satisfied = false;
    Map<String, dynamic> detections = {};
    
    if (modelType == ModelType.bolt) {
      satisfied = _checkBoltCondition(results, detections);
    } else if (modelType == ModelType.door) {
      satisfied = _checkDoorCondition(results, detections);
    }
    
    if (satisfied) {
      if (!_conditionMet) {
        _conditionMet = true;
        _conditionStartTime = DateTime.now();
        print('✓ 조건 만족! 타이머 시작...');
      }
    } else {
      if (_conditionMet) {
        print('⚠️  조건 해제됨. 타이머 리셋.');
        _resetCondition();
      }
    }
    
    return {
      'satisfied': satisfied,
      'detections': detections,
    };
  }
  
  /// Check if condition has been met for required duration
  bool shouldInspect() {
    if (!_conditionMet || _conditionStartTime == null) {
      return false;
    }
    
    final elapsed = DateTime.now().difference(_conditionStartTime!).inMilliseconds / 1000.0;
    return elapsed >= requiredDuration;
  }
  
  /// Get elapsed time since condition was met
  double getElapsedTime() {
    if (!_conditionMet || _conditionStartTime == null) {
      return 0.0;
    }
    return DateTime.now().difference(_conditionStartTime!).inMilliseconds / 1000.0;
  }
  
  void _resetCondition() {
    _conditionMet = false;
    _conditionStartTime = null;
  }
  
  Map<String, dynamic> _getEmptyDetections() {
    if (modelType == ModelType.bolt) {
      return {'bolts': [], 'frames': []};
    } else {
      return {'high': [], 'mid': [], 'low': []};
    }
  }
  
  /// Check bolt condition (similar to live.py's _check_bolt_condition)
  bool _checkBoltCondition(List<YOLOResult> results, Map<String, dynamic> detections) {
    List<Map<String, dynamic>> boltDetections = [];
    List<Map<String, dynamic>> frameDetections = [];
    
    for (final result in results) {
      final detection = {
        'classIndex': result.classIndex,
        'bbox': [
          result.boundingBox.left,
          result.boundingBox.top,
          result.boundingBox.right,
          result.boundingBox.bottom,
        ],
        'conf': result.confidence,
        'center': [
          (result.boundingBox.left + result.boundingBox.right) / 2,
          (result.boundingBox.top + result.boundingBox.bottom) / 2,
        ],
      };
      
      // Class IDs: 0,1 = bolts, 2-7 = frames
      if (result.classIndex == 0 || result.classIndex == 1) {
        boltDetections.add(detection);
      } else if (result.classIndex >= 2 && result.classIndex <= 7) {
        frameDetections.add(detection);
      }
    }
    
    detections['bolts'] = boltDetections;
    detections['frames'] = frameDetections;
    
    // Condition: exactly 1 frame detected
    return frameDetections.length == 1;
  }
  
  /// Check door condition (similar to live.py's _check_frontdoor_condition)
  bool _checkDoorCondition(List<YOLOResult> results, Map<String, dynamic> detections) {
    Map<String, List<Map<String, dynamic>>> parts = {
      'high': [],
      'mid': [],
      'low': [],
    };
    
    for (final result in results) {
      final className = result.className.toLowerCase();
      if (parts.containsKey(className)) {
        parts[className]!.add({
          'bbox': [
            result.boundingBox.left,
            result.boundingBox.top,
            result.boundingBox.right,
            result.boundingBox.bottom,
          ],
          'conf': result.confidence,
          'classIndex': result.classIndex,
        });
      }
    }
    
    detections['high'] = parts['high']!;
    detections['mid'] = parts['mid']!;
    detections['low'] = parts['low']!;
    
    // Condition: all three parts OR high + low (without mid)
    final hasAllThree = parts['high']!.length == 1 &&
                        parts['mid']!.length == 1 &&
                        parts['low']!.length == 1;
    final hasHighLow = parts['high']!.length == 1 &&
                       parts['low']!.length == 1 &&
                       parts['mid']!.isEmpty;
    
    return hasAllThree || hasHighLow;
  }
  
  /// Crop image based on bounding box coordinates
  /// Similar to live.py's _crop_obb_object or simple crop
  /// 
  /// [imageBytes] 원본 이미지 바이트
  /// [bbox] 바운딩 박스 좌표 [x1, y1, x2, y2]
  /// [debugLabel] 디버그 모드일 때 파일명에 사용할 라벨 (예: 'bolt_1', 'door_high')
  Future<Uint8List?> cropImage(
    Uint8List imageBytes,
    List<double> bbox, {
    String? debugLabel,
  }) async {
    try {
      // Decode image
      final codec = await ui.instantiateImageCodec(imageBytes);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      
      // Get bounding box coordinates
      final x1 = bbox[0].toInt();
      final y1 = bbox[1].toInt();
      final x2 = bbox[2].toInt();
      final y2 = bbox[3].toInt();
      
      // Ensure valid coordinates
      final width = image.width;
      final height = image.height;
      final cropX = x1.clamp(0, width);
      final cropY = y1.clamp(0, height);
      final cropW = (x2 - x1).clamp(1, width - cropX);
      final cropH = (y2 - y1).clamp(1, height - cropY);
      
      // Crop image
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder);
      canvas.drawImageRect(
        image,
        Rect.fromLTWH(cropX.toDouble(), cropY.toDouble(), cropW.toDouble(), cropH.toDouble()),
        Rect.fromLTWH(0, 0, cropW.toDouble(), cropH.toDouble()),
        Paint(),
      );
      
      final picture = recorder.endRecording();
      final croppedImage = await picture.toImage(cropW, cropH);
      final byteData = await croppedImage.toByteData(format: ui.ImageByteFormat.png);
      
      image.dispose();
      croppedImage.dispose();
      
      final croppedBytes = byteData?.buffer.asUint8List();
      
      // Debug: Save cropped image (similar to live.py)
      if (debug && _debugCropDir != null && croppedBytes != null && debugLabel != null) {
        await _saveDebugCrop(croppedBytes, debugLabel, cropW, cropH);
      }
      
      return croppedBytes;
    } catch (e) {
      print('Error cropping image: $e');
      return null;
    }
  }
  
  /// Save cropped image for debugging (similar to live.py)
  /// 
  /// [label] 'bolt_1', 'door_high' 등의 라벨
  /// label이 'bolt'로 시작하면 bolt 폴더에, 'door'로 시작하면 door 폴더에 저장
  Future<void> _saveDebugCrop(
    Uint8List croppedBytes,
    String label,
    int width,
    int height,
  ) async {
    if (_debugCropDir == null) return;
    
    try {
      final timestamp = DateTime.now().toIso8601String()
          .replaceAll(':', '-')
          .replaceAll('.', '-')
          .substring(0, 19); // YYYY-MM-DDTHH-MM-SS
      
      // label에 따라 하위 폴더 결정
      String subFolder;
      String fileName;
      if (label.startsWith('bolt')) {
        subFolder = 'bolt';
        fileName = label.replaceFirst('bolt_', ''); // 'bolt_1_sedan' -> '1_sedan'
      } else if (label.startsWith('door')) {
        subFolder = 'door';
        fileName = label.replaceFirst('door_', ''); // 'door_high' -> 'high'
      } else {
        subFolder = 'other';
        fileName = label;
      }
      
      final filename = '$_debugCropDir/$subFolder/${fileName}_$timestamp.png';
      final file = File(filename);
      await file.writeAsBytes(croppedBytes);
      
      print('  크롭 이미지 저장: $filename (크기: ${width}x${height})');
    } catch (e) {
      print('⚠️  크롭 이미지 저장 실패: $e');
    }
  }
  
  /// Reset inspection state
  void reset() {
    _resetCondition();
  }
}

