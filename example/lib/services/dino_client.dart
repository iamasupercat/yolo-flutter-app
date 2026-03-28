// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'dart:convert';
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

/// DINO 모델 서버 클라이언트
/// Python 서버에 이미지를 보내고 분류 결과를 받아옴
class DINOClient {
  final String baseUrl;
  final Duration timeout;

  DINOClient({
    required this.baseUrl,
    this.timeout = const Duration(seconds: 10),
  });

  /// 서버 상태 확인 및 자동 시작 시도
  Future<bool> checkHealth({bool autoStart = true}) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(timeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['status'] == 'ok';
      }
      return false;
    } catch (e) {
      print('DINO 서버 health check 실패: $e');
      return false;
    }
  }

  /// 이미지를 224x224로 리사이즈
  Future<Uint8List> _resizeImage(Uint8List imageBytes, int targetSize) async {
    // 이미지 디코드
    final codec = await ui.instantiateImageCodec(imageBytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;

    // 224x224로 리사이즈
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    final paint = ui.Paint()..filterQuality = ui.FilterQuality.high;

    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTWH(0, 0, targetSize.toDouble(), targetSize.toDouble()),
      paint,
    );

    final picture = recorder.endRecording();
    final resizedImage = await picture.toImage(targetSize, targetSize);
    final byteData = await resizedImage.toByteData(
      format: ui.ImageByteFormat.png,
    );

    // 리소스 정리
    image.dispose();
    resizedImage.dispose();
    codec.dispose();

    return byteData!.buffer.asUint8List();
  }

  /// 이미지 분류 요청
  ///
  /// [imageBytes] 원본 크롭된 이미지 바이트 (서버에서 224x224로 리사이즈됨)
  /// [modelType] 'bolt', 'door_high', 'door_mid', 'door_low'
  /// [filenamePrefix] 서버에서 파일명 생성용 prefix (live.py 스타일)
  ///
  /// Returns: 분류 결과 맵
  Future<Map<String, dynamic>?> classifyImage(
    Uint8List imageBytes,
    String modelType, {
    String? filenamePrefix,
  }) async {
    try {
      // Multipart request 생성
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/classify'),
      );

      // 원본 이미지 파일 추가 (서버에서 224x224로 리사이즈)
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes, // 원본 크롭 이미지 (리사이즈 전)
          filename: 'cropped_image.png',
        ),
      );

      // 모델 타입 추가
      request.fields['model_type'] = modelType;
      request.fields['format'] = 'binary';

      // 파일명 prefix 추가 (서버에서 파일명 생성용)
      // live.py 스타일: bolt_{i+1}_{frame_name} 또는 door_{part}
      if (filenamePrefix != null) {
        request.fields['filename_prefix'] = filenamePrefix;
      }

      // 요청 전송
      final streamedResponse = await request.send().timeout(timeout);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        if (response.body.isEmpty) {
          print('DINO 서버 응답이 비어있습니다.');
          return null;
        }
        final result = json.decode(response.body) as Map<String, dynamic>;
        return result;
      } else {
        print('DINO 분류 실패: HTTP ${response.statusCode}');
        if (response.body.isNotEmpty) {
          try {
            final error = json.decode(response.body);
            print('  오류 메시지: ${error['error']}');
          } catch (_) {
            print('  응답 본문: ${response.body}');
          }
        }
        return null;
      }
    } catch (e) {
      print('DINO 분류 요청 중 오류: $e');
      return null;
    }
  }

  /// Base64 인코딩된 이미지로 분류 요청 (대안)
  Future<Map<String, dynamic>?> classifyImageBase64(
    Uint8List imageBytes,
    String modelType,
  ) async {
    try {
      final base64Image = base64Encode(imageBytes);

      final response = await http
          .post(
            Uri.parse('$baseUrl/classify'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({
              'image': base64Image,
              'model_type': modelType,
              'format': 'base64',
            }),
          )
          .timeout(timeout);

      if (response.statusCode == 200) {
        final result = json.decode(response.body) as Map<String, dynamic>;
        return result;
      } else {
        final error = json.decode(response.body);
        print('DINO 분류 실패: ${error['error']}');
        return null;
      }
    } catch (e) {
      print('DINO 분류 요청 중 오류: $e');
      return null;
    }
  }

  /// 정지된 프레임 이미지를 서버에 저장하고 YOLO 좌표로 크롭
  ///
  /// [imageBytes] 정지된 프레임 이미지 바이트
  /// [detections] YOLO 탐지 결과 리스트 (JSON으로 변환하여 전송)
  /// [modelType] 'bolt' 또는 'door'
  /// [filename] 저장할 파일명 (선택사항, 없으면 자동 생성)
  /// [frameWidth] 캡처된 프레임 너비 (선택사항, 디버깅용)
  /// [frameHeight] 캡처된 프레임 높이 (선택사항, 디버깅용)
  /// [origWidth] YOLO 원본 이미지 너비 (선택사항, 화면 변환 재현용)
  /// [origHeight] YOLO 원본 이미지 높이 (선택사항, 화면 변환 재현용)
  ///
  /// Returns: 저장 결과 맵 (success, filepath, filename, size, cropped_files)
  Future<Map<String, dynamic>?> saveFrame(
    Uint8List imageBytes,
    List<Map<String, dynamic>> detections,
    String modelType, {
    String? filename,
    int? frameWidth,
    int? frameHeight,
    int? origWidth,
    int? origHeight,
  }) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/save_frame'),
      );

      // 이미지 파일 추가
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: filename ?? 'frozen_frame.jpg',
        ),
      );

      // 파일명 추가 (선택사항)
      if (filename != null) {
        request.fields['filename'] = filename;
      }

      // 모델 타입 추가
      request.fields['model_type'] = modelType;

      // 프레임 크기 정보 추가 (디버깅 및 좌표 변환용)
      if (frameWidth != null && frameHeight != null) {
        request.fields['frame_width'] = frameWidth.toString();
        request.fields['frame_height'] = frameHeight.toString();
        print('  📐 프레임 크기 정보 전송: ${frameWidth}x${frameHeight}');
      }

      // 원본 이미지 크기 정보 추가 (화면 변환 재현용)
      if (origWidth != null && origHeight != null) {
        request.fields['orig_width'] = origWidth.toString();
        request.fields['orig_height'] = origHeight.toString();
        print('  📐 원본 이미지 크기 정보 전송: ${origWidth}x${origHeight}');
      }

      // 화면 크기 정보 추가 (화면 변환 재현용)
      // 화면 크기는 프레임 크기와 동일하다고 가정 (실제로는 YOLOView의 크기를 전송해야 함)
      if (frameWidth != null && frameHeight != null) {
        request.fields['view_width'] = frameWidth.toString();
        request.fields['view_height'] = frameHeight.toString();
        print('  📐 화면 크기 정보 전송: ${frameWidth}x${frameHeight}');
      }

      // YOLO 탐지 결과를 JSON으로 변환하여 전송
      // YOLOResult를 Map으로 변환 (정규화 좌표 포함)
      final detectionsList = detections.map((det) {
        final detection = {
          'classIndex': det['classIndex'],
          'className': det['className'],
          'confidence': det['confidence'],
          'boundingBox': {
            'left': det['boundingBox']['left'],
            'top': det['boundingBox']['top'],
            'right': det['boundingBox']['right'],
            'bottom': det['boundingBox']['bottom'],
          },
        };
        // 정규화 좌표가 있으면 추가
        if (det.containsKey('normalizedBox')) {
          detection['normalizedBox'] = {
            'left': det['normalizedBox']['left'],
            'top': det['normalizedBox']['top'],
            'right': det['normalizedBox']['right'],
            'bottom': det['normalizedBox']['bottom'],
          };
        }
        return detection;
      }).toList();

      request.fields['detections'] = json.encode(detectionsList);

      // 요청 전송
      print('  📡 서버 연결 시도: $baseUrl/save_frame');
      print('  📦 전송 데이터 크기: ${imageBytes.length} bytes');
      print('  📋 탐지 결과 개수: ${detections.length}개');
      final streamedResponse = await request.send().timeout(timeout);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        if (response.body.isEmpty) {
          print('DINO 서버 응답이 비어있습니다.');
          return null;
        }
        final result = json.decode(response.body) as Map<String, dynamic>;
        return result;
      } else {
        print('DINO 프레임 저장 실패: HTTP ${response.statusCode}');
        if (response.body.isNotEmpty) {
          try {
            final error = json.decode(response.body);
            print('  오류 메시지: ${error['error']}');
          } catch (_) {
            print('  응답 본문: ${response.body}');
          }
        } else {
          print('  응답 본문이 비어있습니다.');
        }
        // HTTP 403은 보통 서버가 실행되지 않았거나 CORS 문제일 수 있음
        if (response.statusCode == 403) {
          print('  ⚠️  HTTP 403: 서버가 요청을 거부했습니다.');
          print('     - DINO 서버가 실행 중인지 확인하세요');
          print('     - 서버 URL이 올바른지 확인하세요: $baseUrl');
          print('     - CORS 설정이 올바른지 확인하세요');
        }
        return null;
      }
    } catch (e) {
      print('DINO 프레임 저장 요청 중 오류: $e');
      if (e.toString().contains('Connection refused')) {
        print('  ❌ 연결 거부됨: 서버가 실행되지 않았거나 연결할 수 없습니다.');
        print('     - DINO 서버가 실행 중인지 확인하세요');
        print('     - 서버 URL이 올바른지 확인하세요: $baseUrl');
        print('     - PC와 핸드폰이 같은 네트워크에 연결되어 있는지 확인하세요');
        print('     - 방화벽에서 포트 5001이 열려있는지 확인하세요');
      } else if (e.toString().contains('SocketException')) {
        print('  ❌ 소켓 오류: 네트워크 연결 문제가 있습니다.');
        print('     - PC IP 주소가 올바른지 확인하세요: $baseUrl');
        print('     - PC와 핸드폰이 같은 Wi-Fi 네트워크에 연결되어 있는지 확인하세요');
      } else if (e.toString().contains('TimeoutException')) {
        print('  ❌ 타임아웃: 서버 응답이 너무 느립니다.');
        print('     - 서버가 실행 중인지 확인하세요');
        print('     - 네트워크 연결 상태를 확인하세요');
      }
      return null;
    }
  }
}
