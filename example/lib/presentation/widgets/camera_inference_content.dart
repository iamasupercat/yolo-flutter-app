// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../controllers/camera_inference_controller.dart';
import 'model_loading_overlay.dart';

/// Main content widget that handles the camera view and loading states
class CameraInferenceContent extends StatelessWidget {
  const CameraInferenceContent({
    super.key,
    required this.controller,
    this.rebuildKey = 0,
  });

  final CameraInferenceController controller;
  final int rebuildKey;

  @override
  Widget build(BuildContext context) {
    if (controller.modelPath != null && !controller.isModelLoading) {
      // 카메라가 정지된 경우 정지된 프레임 표시
      if (controller.isCameraFrozen && controller.frozenFrame != null) {
        return Image.memory(
          controller.frozenFrame!,
          fit: BoxFit.cover,
          width: double.infinity,
          height: double.infinity,
        );
      }

      // 정상적으로 카메라 스트림 표시
      return YOLOView(
        key: ValueKey(
          'yolo_view_${controller.modelPath}_${controller.selectedModel.task.name}_$rebuildKey',
        ),
        controller: controller.yoloController,
        modelPath: controller.modelPath!,
        task: controller.selectedModel.task,
        streamingConfig: const YOLOStreamingConfig.minimal(),
        onResult: controller.onDetectionResults,
        onPerformanceMetrics: (metrics) =>
            controller.onPerformanceMetrics(metrics.fps),
        onZoomChanged: controller.onZoomChanged,
        lensFacing: controller.lensFacing,
      );
    } else if (controller.isModelLoading) {
      return ModelLoadingOverlay(
        loadingMessage: controller.loadingMessage,
        downloadProgress: controller.downloadProgress,
      );
    } else {
      return const Center(
        child: Text('No model loaded', style: TextStyle(color: Colors.white)),
      );
    }
  }
}
