// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/models/yolo_task.dart';

enum ModelType {
  bolt('yolo11_bolt_float32', YOLOTask.obb),
  door('yolo11_door_float32', YOLOTask.obb);

  final String modelName;

  final YOLOTask task;

  const ModelType(this.modelName, this.task);
}

enum SliderType { none, numItems, confidence, iou }
