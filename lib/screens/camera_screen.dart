import 'dart:async';
import 'dart:io';

import 'package:gallery_saver_plus/gallery_saver.dart';
import 'package:path_provider/path_provider.dart' as path_provider;
import 'package:opencv_core/opencv.dart' as cv;
import 'package:image/image.dart' as img;
import 'package:animated_icon/animated_icon.dart';
import 'package:camera/camera.dart';
import 'package:clickable_list_wheel_view/clickable_list_wheel_widget.dart';
import 'package:combine/combine.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tap_debouncer/tap_debouncer.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_object_detection/models/bounding_box.dart';
import 'package:tflite_object_detection/services/detector_service.dart';
import 'package:tflite_object_detection/widgets/overlay_view_widget.dart';
import '../utils/image_utils.dart';

Future<List<BoundingBox>> _detectObject(args) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(
    args[2] as RootIsolateToken,
  );
  final image = await convertCameraImageToImage(args[0] as CameraImage);
  DetectorService detector = args[1] as DetectorService;
  final results = await detector.detect(
    image,
  );
  if (results != null) {
    return results;
  }
  return [];
}

Future<void> _detectImage(args) async {
  try {
    BackgroundIsolateBinaryMessenger.ensureInitialized(
      args[2] as RootIsolateToken,
    );
    XFile imageFile = args[0] as XFile;
    Uint8List bytes = await imageFile.readAsBytes();
    img.Image? image = img.decodeImage(bytes);
    DetectorService detector = args[1] as DetectorService;
    List<BoundingBox>? results = await detector.detect(
      image,
    );
    results ??= [];
    final annotatedImage = await CombineWorker().executeWithArg(
      drawBoundingBoxes,
      [image!, results],
    );
    final imageBytes = img.decodeImage(annotatedImage!);
    final pngData = img.encodePng(imageBytes!);
    final directory = await path_provider.getTemporaryDirectory();
    final tempFilePath = '${directory.path}/temp_image.png';
    final tempFile = File(tempFilePath);
    await tempFile.writeAsBytes(pngData);
    await GallerySaver.saveImage(tempFile.path);
  } catch (e) {
    debugPrint("ERROR: $e");
  }
}

Future<Uint8List?> drawBoundingBoxes(args) async {
  try {
    img.Image imageTemp = args[0] as img.Image;
    Uint8List imageBytes = img.encodeJpg(imageTemp);
    List<BoundingBox> boxes = args[1] as List<BoundingBox>;
    cv.Mat image = await cv.imdecodeAsync(imageBytes, cv.IMREAD_COLOR);

    int imgWidth = image.cols;
    int imgHeight = image.rows;

    cv.Scalar boxColor = cv.Scalar(0, 0, 255);
    cv.Scalar textColor = cv.Scalar(255, 255, 255);

    for (BoundingBox box in boxes) {
      int x1 = (box.x1 * imgHeight).toInt();
      int y1 = (box.y1 * imgWidth).toInt();
      int x2 = (box.x2 * imgHeight).toInt();
      int y2 = (box.y2 * imgWidth).toInt();

      cv.Rect rect = cv.Rect(y1, x1, y2 - y1, x2 - x1);

      await cv.rectangleAsync(image, rect, boxColor, thickness: 2);

      String label =
          "${box.className} ${(box.confidence * 100).toStringAsFixed(2)}%";

      var textSizeResult = await cv.getTextSizeAsync(
        label,
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
      );

      cv.Size textSize = textSizeResult.$1;
      int baseline = textSizeResult.$2;

      cv.Rect labelRect = cv.Rect(
        y1,
        x1 - textSize.height - baseline,
        textSize.width,
        textSize.height + baseline,
      );
      await cv.rectangleAsync(
        image,
        labelRect,
        cv.Scalar(0, 0, 0),
        thickness: -1,
      );

      await cv.putTextAsync(
        image,
        label,
        cv.Point(y1, x1 - 5),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        textColor,
        thickness: 1,
      );
    }

    var (success, resultBytes) = await cv.imencodeAsync('.png', image);
    if (success) {
      return resultBytes;
    } else {
      debugPrint("Image encoding failed.");
      return null;
    }
  } catch (e) {
    debugPrint("Error drawing bounding boxes: $e");
    return null;
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  RootIsolateToken rootIsolateToken = RootIsolateToken.instance!;
  late DetectorService _detectorService;
  List<CameraDescription> _cameras = [];
  CameraController? _controller;
  Future? _cameraValue;
  Timer? _processingTimer;
  bool _isProcessing = false;
  List<BoundingBox> _recognitions = [];
  int _selectedIndex = 1;

  int _selectedCameraIndex = 0;
  Size? _previewSize;
  bool _isFrontCamera = false;
  bool _isFlashOn = false;
  double _maxZoom = 5.0;
  //double _currentZoom = 1.0;
  //File? _capturedImage;

  bool _isRecording = false;

  final List<String> _cameraModes = ["Video", "Photo"];
  bool _isPhotoMode = true;

  Interpreter? _interpreter;
  String? _labelData;
  static const String modelPath = "assets/model.tflite";
  static const String labelPath = "assets/labels.txt";
  InterpreterOptions options = InterpreterOptions()..threads = 4;
  FixedExtentScrollController _scrollController = FixedExtentScrollController(
    initialItem: 1,
  );

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initStateAsync();
  }

  @override
  void dispose() async {
    await _controller?.stopImageStream();
    _cameraValue = null;
    await _controller?.dispose();
    _interpreter?.close();
    _interpreter = null;
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  _initStateAsync() async {
    _cameras = await availableCameras();
    await _initInterpreter();
    _detectorService = DetectorService();
    await _detectorService.setup(_interpreter!, _labelData!);
    _initCameraController();
  }

  Future<void> _initInterpreter() async {
    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate()); //TODO: Change as per need
    }
    _interpreter = await Interpreter.fromAsset(modelPath, options: options);
    _labelData = await rootBundle.loadString(labelPath);
  }

  _initCameraController() async {
    _controller = CameraController(
      _cameras[_selectedCameraIndex],
      ResolutionPreset.max,
      fps: 60,
      enableAudio: true,
    );
    _controller?.initialize().then((_) async {
      _cameraValue = _getCameraValue();
      _maxZoom = await _getMaxZoomValue();
      _controller?.lockCaptureOrientation(DeviceOrientation.portraitUp);
      if (_isPhotoMode) {
        await _controller?.startImageStream(onLatestImageAvailable);
      }
      if (!_isPhotoMode) {
        await _controller?.prepareForVideoRecording();
      }
      setState(() {
        _previewSize = _controller?.value.previewSize!;
      });
    });
  }

  _getCameraValue() async {
    return _controller?.value;
  }

  _getMaxZoomValue() async {
    return _controller?.getMaxZoomLevel() ?? 5.0;
  }

  onLatestImageAvailable(CameraImage cameraImage) async {
    if (_isProcessing) return;
    if (_processingTimer == null || !_processingTimer!.isActive) {
      _isProcessing = true;
      _recognitions = await CombineWorker().executeWithArg(_detectObject, [
        cameraImage,
        _detectorService,
        rootIsolateToken,
      ]);
      setState(() {});
      _processingTimer = Timer(const Duration(milliseconds: 10), (() {
        _isProcessing = false;
      }));
    }
  }

  _toggleFlashLight() {
    if (_isFlashOn) {
      _controller?.setFlashMode(FlashMode.off);
      setState(() {
        _isFlashOn = false;
      });
    } else {
      _controller?.setFlashMode(FlashMode.torch);
      setState(() {
        _isFlashOn = true;
      });
    }
  }

  _switchCamera() async {
    if (_controller != null) {
      if (_controller!.value.isStreamingImages) {
        await _controller?.stopImageStream();
      }
      _cameraValue = null;
      await _controller?.dispose();
    }
    _selectedCameraIndex = (_selectedCameraIndex + 1) % _cameras.length;
      _reInitCameraController();
  }

  _reInitCameraController() async {
    _controller = CameraController(
      _cameras[_selectedCameraIndex],
      ResolutionPreset.max,
      fps: 60,
      enableAudio: true,
    );
    _initCameraController();
    if (_selectedCameraIndex == 0) {
      setState(() {
        _isFrontCamera = false;
      });
    } else {
      setState(() {
        _controller?.setFlashMode(FlashMode.off);
        _isFlashOn = false;
        _isFrontCamera = true;
      });
    }
    if (mounted) {
      setState(() {});
    }
  }

  _changeCameraMode(int index) {
    if (index == 1) {
      setState(() {
        _selectedIndex = index;
        _isPhotoMode = true;
        _scrollController = FixedExtentScrollController(initialItem: index);
      });
    } else {
      setState(() {
        _selectedIndex = index;
        _isPhotoMode = false;
        _scrollController = FixedExtentScrollController(initialItem: index);
      });
    }
    _reInitCameraController();
  }

  _capturePhoto() async {
    if (!_controller!.value.isInitialized) return;
    if (_controller!.value.isTakingPicture) return;
    try {
      setState(() {
        _isRecording = true;
      });
      if (_controller!.value.isStreamingImages) {
        await _controller?.stopImageStream();
      }
      final imageFile = await _controller!.takePicture();
      await CombineWorker().executeWithArg(_detectImage, [
        imageFile,
        _detectorService,
        rootIsolateToken,
      ]);
      await _controller?.startImageStream(onLatestImageAvailable);
    } catch (e) {
    } finally {
      setState(() {
        _isRecording = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.sizeOf(context);
    if (_controller == null || !_controller!.value.isInitialized || _controller!.value.previewSize == null) {
      return const Center(child: CircularProgressIndicator());
    }

    final previewSize = _controller?.value.previewSize!;
    final aspectRatio = previewSize!.height / previewSize.width;

    return SafeArea(
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(
          children: [
            Center(
              child: AspectRatio(
                aspectRatio: _controller!.value.previewSize!.height / _controller!.value.previewSize!.width,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    buildCameraPreview(),
                    buildOverlayView(),
                  ],
                ),
              ),
            ),
            buildTopBar(),
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                height: 220,
                child: Column(
                  children: [
                    buildCameraMode(size),
                    Expanded(
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          buildGalleryPreview(),
                          buildRecording(),
                          buildSwapCamera(),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Positioned buildTopBar() {
    return Positioned(
      left: 0,
      right: 0,
      top: 0,
      child: Container(
        color: Colors.black,
        height: 55,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                color: Colors.white,
                disabledColor: Colors.grey,
                onPressed:
                    _isFrontCamera
                        ? null
                        : () {
                          _toggleFlashLight();
                        },
                icon: Icon(_isFlashOn ? Icons.flash_on : Icons.flash_off),
              ),
              IconButton(
                onPressed: () {},
                icon: Icon(Icons.settings, color: Colors.white),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Positioned buildOverlayView() {
    return Positioned.fill(
      child: Center(
        child: _isPhotoMode
            ? _previewSize != null ? OverlayView(results: _recognitions, previewSize: _previewSize!,) : const SizedBox.shrink()
            : const SizedBox.shrink(),
      ),
    );
  }

  Positioned buildCameraPreview() {
    return Positioned.fill(
      child: FutureBuilder(
        future: _cameraValue,
        builder: ((context, snapshot) {
          if (_cameraValue != null) {
            return Center(child: CameraPreview(_controller!));
          } else {
            return const Center(child: CircularProgressIndicator.adaptive());
          }
        }),
      ),
    );
  }

  SizedBox buildCameraMode(Size size) {
    if (!_isRecording) {
      return SizedBox(
        width: size.width,
        height: 45,
        child: RotatedBox(
          quarterTurns: 1,
          child: ClickableListWheelScrollView(
            scrollController: _scrollController,
            itemHeight: 100,
            itemCount: _cameraModes.length,
            onItemTapCallback: (index) {},
            child: ListWheelScrollView.useDelegate(
              physics: FixedExtentScrollPhysics(),
              controller: _scrollController,
              perspective: 0.0001,
              itemExtent: 100,
              onSelectedItemChanged: (index) {
                _changeCameraMode(index);
              },
              childDelegate: ListWheelChildListDelegate(
                children: [
                  Container(
                    alignment: Alignment.center,
                    child: RotatedBox(
                      quarterTurns: -1,
                      child: AnimatedDefaultTextStyle(
                        duration: Duration(milliseconds: 50),
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: _selectedIndex == 0 ? 20 : 18,
                          color:
                              _selectedIndex == 0 ? Colors.amber : Colors.white,
                        ),
                        child: Text(_cameraModes[0]),
                      ),
                    ),
                  ),
                  Container(
                    alignment: Alignment.center,
                    child: RotatedBox(
                      quarterTurns: -1,
                      child: AnimatedDefaultTextStyle(
                        duration: Duration(milliseconds: 50),
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: _selectedIndex == 1 ? 20 : 18,
                          color:
                              _selectedIndex == 1 ? Colors.amber : Colors.white,
                        ),
                        child: Text(_cameraModes[1]),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    } else {
      return SizedBox(height: 45);
    }
  }

  Container buildGalleryPreview() {
    return Container(
      height: 60,
      width: 60,
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.5),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Icon(Icons.image, color: Colors.white.withOpacity(0.5)),
    );
  }

  Container buildSwapCamera() {
    return Container(
      height: 60,
      width: 60,
      child: TapDebouncer(
        cooldown: Duration(seconds: 1),
        onTap: () async {
          _switchCamera();
        },
        builder: (BuildContext context, Future<void> Function()? onTap) {
          return Container(
            margin: EdgeInsets.all(5),
            child: Container(
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.black.withOpacity(0.5),
              ),
              child: AnimateIcon(
                onTap: onTap ?? () {},
                iconType: IconType.animatedOnTap,
                animateIcon: AnimateIcons.refresh,
                color: Colors.white,
              ),
            ),
          );
        },
      ),
    );
  }

  Container buildRecording() {
    if (_selectedIndex == 0) {
      return Container(
        height: 80,
        width: 80,
        child: GestureDetector(
          onTap: () {
            setState(() {
              _isRecording = !_isRecording;
            });
            debugPrint("Recording Triggered");
          },
          child: Container(
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 5),
            ),
            child: AnimatedContainer(
              duration: Duration(milliseconds: 110),
              margin: EdgeInsets.all(_isRecording ? 23 : 4),
              decoration: BoxDecoration(
                color: Colors.red,
                borderRadius: BorderRadius.circular(_isRecording ? 2 : 100),
              ),
            ),
          ),
        ),
      );
    } else {
      return Container(
        height: 80,
        width: 80,
        child: GestureDetector(
          onTap: () {
            _capturePhoto();
          },
          child: Container(
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 5),
            ),
            child: AnimatedContainer(
              duration: Duration(milliseconds: 110),
              margin: EdgeInsets.all(4),
              decoration: BoxDecoration(
                color: _isRecording ? Colors.transparent : Colors.amber,
                borderRadius: BorderRadius.circular(100),
              ),
            ),
          ),
        ),
      );
    }
  }
}
