import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../models/bounding_box.dart';

class DetectorService {

  Interpreter? _interpreter;

  List<String> _labels = [];
  int _tensorWidth = 0;
  int _tensorHeight = 0;
  int _numChannel = 0;
  int _numElements = 0;

  static const double confidenceThreshold = 0.25;
  static const double iouThreshold = 0.45;

  bool _isSetupDone = false;

  Future<void> setup(Interpreter? interpreter, String labelData) async {
    if (_isSetupDone || interpreter == null) return;
    try {
      _interpreter = interpreter;
      var inputShape = interpreter.getInputTensor(0).shape;
      var outputShape = interpreter.getOutputTensor(0).shape;

      _tensorWidth = inputShape[1];
      _tensorHeight = inputShape[2];
      _numChannel = outputShape[1];
      _numElements = outputShape[2];


      _labels =
          labelData.split('\n').where((label) => label.isNotEmpty).toList();
      _isSetupDone = true;
    } catch (e) {
      debugPrint('Error setting up the detector: $e');
    }
  }

  Future<List<BoundingBox>?> detect(
      img.Image? image) async {
    if (!_isSetupDone || _interpreter == null || _tensorWidth == 0 || _tensorHeight == 0) {
      debugPrint('Detector not setup. Call setup() before detect().');
      return null;
    }

    if (image == null) return null;
    var preInferenceTime = DateTime.now().millisecondsSinceEpoch;
    final rotatedImage = img.copyRotate(image, angle: 90);
    final resizedImage = img.copyResize(rotatedImage,
        width: _tensorWidth, height: _tensorHeight);
    final tensorImage = _preprocessImage(resizedImage);

    final output = List.generate(
        1,
            (_) =>
            List.generate(_numChannel, (_) => List.filled(_numElements, 0.0)));

    _interpreter!.run(tensorImage, output);
    final outputWithoutBatch = output[0];

    final bestBoxes = _extractBestBoxes(outputWithoutBatch);
    var inferenceTime = DateTime.now().millisecondsSinceEpoch - preInferenceTime;
    debugPrint("InferenceTime: $inferenceTime");
    return bestBoxes;
  }

  Uint8List _preprocessImage(img.Image image) {
    final Float32List buffer = Float32List(_tensorWidth * _tensorHeight * 3);
    int index = 0;

    for (int y = 0; y < _tensorHeight; y++) {
      for (int x = 0; x < _tensorWidth; x++) {
        final pixel = image.getPixel(x, y);

        final red = pixel.r;
        final green = pixel.g;
        final blue = pixel.b;

        buffer[index++] = red / 255.0;
        buffer[index++] = green / 255.0;
        buffer[index++] = blue / 255.0;
      }
    }
    return buffer.buffer.asUint8List();
  }

  List<BoundingBox> _extractBestBoxes(List<List<double>> output) {
    final boxes = <BoundingBox>[];

    for (int i = 0; i < _numElements; i++) {
      var maxConf = -1.0;
      var maxIdx = -1;

      for (int j = 4; j < _numChannel; j++) {
        if (output[j][i] > maxConf) {
          maxConf = output[j][i];
          maxIdx = j - 4;
        }
      }

      if (maxConf > confidenceThreshold) {
        final cx = output[0][i];
        final cy = output[1][i];
        final w = output[2][i];
        final h = output[3][i];

        final x1 = cx - (w / 2.0);
        final y1 = cy - (h / 2.0);
        final x2 = cx + (w / 2.0);
        final y2 = cy + (h / 2.0);

        if (x1 < 0 || y1 < 0 || x2 > 1 || y2 > 1) continue;

        boxes.add(BoundingBox(
          x1: x1,
          y1: y1,
          x2: x2,
          y2: y2,
          cx: cx,
          cy: cy,
          w: w,
          h: h,
          confidence: maxConf,
          classIndex: maxIdx,
          className: _labels[maxIdx],
        ));
      }
    }
    final nmsBoxes = _applyNMS(boxes);
    return nmsBoxes;
  }

  List<BoundingBox> _applyNMS(List<BoundingBox> boxes) {
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));
    final selectedBoxes = <BoundingBox>[];

    while (boxes.isNotEmpty) {
      final first = boxes.removeAt(0);
      selectedBoxes.add(first);

      boxes.removeWhere((box) => _calculateIoU(first, box) >= iouThreshold);
    }

    return selectedBoxes;
  }

  double _calculateIoU(BoundingBox a, BoundingBox b) {
    final x1 = a.x1 > b.x1 ? a.x1 : b.x1;
    final y1 = a.y1 > b.y1 ? a.y1 : b.y1;
    final x2 = a.x2 < b.x2 ? a.x2 : b.x2;
    final y2 = a.y2 < b.y2 ? a.y2 : b.y2;

    final intersection = (x2 - x1).clamp(0, 1) * (y2 - y1).clamp(0, 1);
    final union = (a.w * a.h) + (b.w * b.h) - intersection;

    return intersection / union;
  }
}
