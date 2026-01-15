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
  
  /// 서버 상태 확인
  Future<bool> checkHealth() async {
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
    final byteData = await resizedImage.toByteData(format: ui.ImageByteFormat.png);
    
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
          imageBytes,  // 원본 크롭 이미지 (리사이즈 전)
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
}

