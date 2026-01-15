// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../controllers/camera_inference_controller.dart';

/// Overlay widget that displays inspection result on frozen frame
class InspectionResultOverlay extends StatelessWidget {
  const InspectionResultOverlay({
    super.key,
    required this.result,
    required this.onRestart,
  });

  final InspectionResult result;
  final VoidCallback onRestart;

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 0,
      left: 0,
      right: 0,
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: result.isGood 
              ? Colors.green.withOpacity(0.95) 
              : Colors.red.withOpacity(0.95),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.3),
              blurRadius: 10,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Icon
            Icon(
              result.isGood ? Icons.check_circle : Icons.cancel,
              size: 64,
              color: Colors.white,
            ),
            const SizedBox(height: 16),
            
            // Result text
            Text(
              result.isGood ? '✅ 양품' : '❌ 불량',
              style: const TextStyle(
                fontSize: 32,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            
            // Details
            if (result.details != null) ...[
              Text(
                result.details!,
                style: const TextStyle(
                  fontSize: 16,
                  color: Colors.white70,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
            ],
            
            // Timestamp
            Text(
              '검사 시간: ${_formatTime(result.timestamp)}',
              style: const TextStyle(
                fontSize: 12,
                color: Colors.white60,
              ),
            ),
            const SizedBox(height: 24),
            
            // Restart button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: onRestart,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: result.isGood ? Colors.green.shade900 : Colors.red.shade900,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text(
                  '다시 검사',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  String _formatTime(DateTime time) {
    return '${time.year}-${time.month.toString().padLeft(2, '0')}-${time.day.toString().padLeft(2, '0')} '
           '${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}:${time.second.toString().padLeft(2, '0')}';
  }
}

