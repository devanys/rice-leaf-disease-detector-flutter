// [IMPORTS & GLOBAL]
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'package:tflite_v2/tflite_v2.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras().catchError((e) => []);
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) => MaterialApp(
        debugShowCheckedModeBanner: false,
        home: ImagePickerDemo(),
      );
}

// [GALLERY / CAMERA SCREEN]
class ImagePickerDemo extends StatefulWidget {
  @override
  _ImagePickerDemoState createState() => _ImagePickerDemoState();
}

class _ImagePickerDemoState extends State<ImagePickerDemo> {
  final picker = ImagePicker();
  XFile? _image;
  File? _file;
  List? _results;
  bool _modelLoaded = false, _loading = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() => _loading = true);
    try {
      await Tflite.loadModel(
        model: "assets/model.tflite",
        labels: "assets/labels.txt",
      );
      setState(() {
        _modelLoaded = true;
        _loading = false;
      });
    } catch (e) {
      setState(() => _loading = false);
    }
  }

  Future<void> _predict(File imageFile) async {
    setState(() => _loading = true);
    try {
      var recognitions = await Tflite.runModelOnImage(
        path: imageFile.path,
        imageMean: 0.0,
        imageStd: 255.0,
        numResults: 6,
        threshold: 0.1,
        asynch: true,
      );
      setState(() {
        _results = recognitions;
        _loading = false;
      });
    } catch (e) {
      setState(() => _loading = false);
    }
  }

  Future<void> _fromGallery() async {
    final img = await picker.pickImage(source: ImageSource.gallery);
    if (img != null) {
      _file = File(img.path);
      setState(() => _image = img);
      await _predict(_file!);
    }
  }

  Future<void> _fromCamera() async {
    final img = await picker.pickImage(source: ImageSource.camera);
    if (img != null) {
      _file = File(img.path);
      setState(() => _image = img);
      await _predict(_file!);
    }
  }

  void _openLive() {
    if (_modelLoaded && cameras.isNotEmpty) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => CameraLiveView(cameras: cameras),
        ),
      );
    }
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Rice Leaf Detection'), backgroundColor: Colors.green),
      body: _loading
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: EdgeInsets.all(16),
              child: Column(
                children: [
                  if (_image != null)
                    Image.file(File(_image!.path), height: 300, fit: BoxFit.cover),
                  SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton.icon(
                        onPressed: _fromGallery,
                        icon: Icon(Icons.photo),
                        label: Text('Gallery'),
                      ),
                      ElevatedButton.icon(
                        onPressed: _fromCamera,
                        icon: Icon(Icons.camera_alt),
                        label: Text('Camera'),
                      ),
                      ElevatedButton.icon(
                        onPressed: _openLive,
                        icon: Icon(Icons.videocam),
                        label: Text('Live'),
                      ),
                    ],
                  ),
                  SizedBox(height: 20),
                  if (_results != null)
                    ..._results!.map((res) => ListTile(
                          title: Text(res['label']),
                          trailing:
                              Text('${(res['confidence'] * 100).toStringAsFixed(1)}%'),
                        )),
                ],
              ),
            ),
    );
  }
}


class CameraLiveView extends StatefulWidget {
  final List<CameraDescription> cameras;
  CameraLiveView({required this.cameras});
  @override
  _CameraLiveViewState createState() => _CameraLiveViewState();
}

class _CameraLiveViewState extends State<CameraLiveView> {
  late CameraController _controller;
  bool _isInitialized = false;
  Map<String, double> _result = {};
  DateTime _lastFrameTime = DateTime.now();
  final int scopeSize = 200;

  final List<String> allLabels = [
    'bacterial_leaf_blight',
    'brown_spot',
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'narrow_brown_spot',
    'neck_blast',
    'rice_hispa',
    'sheath_blight',
    'tungro'
  ];

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _controller = CameraController(widget.cameras[0], ResolutionPreset.medium, enableAudio: false);
    await _controller.initialize();
    if (!mounted) return;
    setState(() => _isInitialized = true);

    _controller.startImageStream((CameraImage image) async {
      if (DateTime.now().difference(_lastFrameTime).inMilliseconds < 1000) return;
      _lastFrameTime = DateTime.now();

      if (_isGreenDominantRGB(image)) {
        var recognitions = await Tflite.runModelOnFrame(
          bytesList: image.planes.map((plane) => plane.bytes).toList(),
          imageHeight: image.height,
          imageWidth: image.width,
          imageMean: 0.0,
          imageStd: 255.0,
          rotation: 90,
          numResults: allLabels.length,
          threshold: 0.01,
        );

        Map<String, double> confidences = {
          for (var label in allLabels) label: 0.0,
        };

        if (recognitions != null) {
          for (var res in recognitions) {
            String label = res['label'].toString().toLowerCase();
            if (confidences.containsKey(label)) {
              confidences[label] = res['confidence'];
            }
          }
        }

        setState(() => _result = confidences);
      } else {
        setState(() => _result = {
          for (var label in allLabels) label: 0.0,
        });
      }
    });
  }

  bool _isGreenDominantRGB(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uvRowStride = image.planes[1].bytesPerRow;

    int startX = width ~/ 2 - scopeSize ~/ 2;
    int startY = height ~/ 2 - scopeSize ~/ 2;
    int greenish = 0;
    int total = 0;

    for (int y = startY; y < startY + scopeSize; y += 5) {
      for (int x = startX; x < startX + scopeSize; x += 5) {
        final uvIndex = (uvRowStride * (y ~/ 2)) + (x & ~1);
        final u = image.planes[1].bytes[uvIndex];
        final v = image.planes[2].bytes[uvIndex];
        final yVal = yPlane.bytes[y * width + x];

        int r = (yVal + 1.403 * (v - 128)).clamp(0, 255).toInt();
        int g = (yVal - 0.344 * (u - 128) - 0.714 * (v - 128)).clamp(0, 255).toInt();
        int b = (yVal + 1.770 * (u - 128)).clamp(0, 255).toInt();

        if (g > r + 20 && g > b + 20 && g > 100) greenish++;
        total++;
      }
    }

    return total > 0 && (greenish / total) > 0.3;
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: Colors.white)),
      );
    }

    String? maxLabel;
    double maxValue = 0;
    _result.forEach((key, value) {
      if (value > maxValue) {
        maxValue = value;
        maxLabel = key;
      }
    });

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(child: CameraPreview(_controller)),
          Center(
            child: Container(
              width: scopeSize.toDouble(),
              height: scopeSize.toDouble(),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.greenAccent, width: 2),
              ),
            ),
          ),
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            left: 16,
            right: 16,
            child: Container(
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: allLabels.map((label) {
                  final confidence = _result[label] ?? 0.0;
                  final isMax = label == maxLabel && confidence > 0.0;

                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Text(
                      '${label.padRight(25)} ${(confidence * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        color: isMax ? Colors.greenAccent : Colors.white,
                        fontWeight: isMax ? FontWeight.bold : FontWeight.normal,
                        fontFamily: 'monospace',
                        fontSize: 14,
                      ),
                    ),
                  );
                }).toList(),
              ),
            ),
          ),
          Positioned(
            top: MediaQuery.of(context).padding.top,
            left: 0,
            child: IconButton(
              icon: Icon(Icons.arrow_back, color: Colors.white),
              onPressed: () => Navigator.pop(context),
            ),
          ),
        ],
      ),
    );
  }
}