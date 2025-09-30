import 'package:flutter/material.dart';
import '../models/bounding_box.dart';

class OverlayView extends StatelessWidget {
  final List<BoundingBox> results;
  final Size previewSize;

  const OverlayView({
    super.key,
    required this.results,
    required this.previewSize,
  });

  Color getColorByPrefix(String label) {
    if (label.startsWith('HR_')) return Colors.green;
    if (label.startsWith('MR_')) return Colors.orange;
    if (label.startsWith('LR_')) return Colors.red;
    return Colors.blueGrey;
  }

  String getLabelWithoutPrefix(String label) {
    final parts = label.split('_');
    if (parts.length > 1) {
      return parts.sublist(1).join('_'); // Handles underscores in class name
    }
    return label;
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final screenWidth = constraints.maxWidth;
        final screenHeight = constraints.maxHeight;

        final previewWidth = previewSize.height;
        final previewHeight = previewSize.width;

        final scaleX = screenWidth / previewWidth;
        final scaleY = screenHeight / previewHeight;

        return Stack(
          children: results.map((box) {
            final left = box.x1 * previewWidth * scaleX;
            final top = box.y1 * previewHeight * scaleY;
            final width = (box.x2 - box.x1) * previewWidth * scaleX;
            final height = (box.y2 - box.y1) * previewHeight * scaleY;

            final color = getColorByPrefix(box.className);
            final label = getLabelWithoutPrefix(box.className);

            return Positioned(
              left: left,
              top: top,
              width: width,
              height: height,
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: color, width: 2),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Align(
                  alignment: Alignment.topLeft,
                  child: Container(
                    color: color.withOpacity(0.6),
                    padding: const EdgeInsets.symmetric(
                      vertical: 2,
                      horizontal: 4,
                    ),
                    child: Text(
                      '$label ${(box.confidence * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ),
            );
          }).toList(),
        );
      },
    );
  }
}
