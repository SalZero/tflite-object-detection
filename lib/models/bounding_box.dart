class BoundingBox {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double cx;
  final double cy;
  final double w;
  final double h;
  final double confidence;
  final int classIndex;
  final String className;

  BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.cx,
    required this.cy,
    required this.w,
    required this.h,
    required this.confidence,
    required this.classIndex,
    required this.className,
  });

  @override
  String toString() {
    return 'BoundingBox(x1: $x1, y1: $y1, x2: $x2, y2: $y2, '
        'cx: $cx, cy: $cy, w: $w, h: $h, '
        'confidence: $confidence, classIndex: $classIndex, className: $className)';
  }
}
