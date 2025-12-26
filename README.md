# Label Detector

Ứng dụng desktop phát hiện và phân đoạn nhãn sản phẩm (product labels) trong thời gian thực sử dụng mô hình YOLO11n-seg (Instance Segmentation).

![Label Detector UI](assets/template.png)

## Tính năng

- **Camera Management**: Tự động phát hiện và chọn camera, bật/tắt camera
- **Instance Segmentation**: Phát hiện và phân đoạn nhãn với YOLO11n-seg (ONNX Runtime hoặc OpenVINO Runtime)
- **Mask Visualization**: Hiển thị segmentation mask với màu sắc và opacity tùy chỉnh
- **Image Preprocessing**: Crop, rotate và sửa hướng ảnh nhãn tự động
  - Crop theo minimum area rectangle từ segmentation mask
  - Force landscape orientation (width >= height)
  - AI 180° fix sử dụng PaddleOCR (PP-LCNet)
- **Image Enhancement**: Cải thiện chất lượng ảnh nhãn
  - Brightness Enhancement (CLAHE) - tăng cường độ sáng ảnh tối
  - Sharpness Enhancement (Unsharp Mask) - làm sắc nét ảnh mờ
- **OCR Pipeline**: Trích xuất thông tin từ nhãn sản phẩm
  - QR Code Detection (ZXing-cpp hoặc WeChat QRCode) - phát hiện và parse QR code
  - Component Extraction - cắt vùng text dựa trên vị trí QR
  - Text Extraction (PaddleOCR) - trích xuất text từ vùng đã cắt
  - Fuzzy Matching - so khớp với database products/sizes/colors
  - Validation - kiểm tra position từ QR khớp với OCR
- **Adjustable Threshold**: Điều chỉnh ngưỡng confidence (0.0 - 1.0)
- **Size Filtering**: Lọc bỏ đối tượng quá lớn theo tỷ lệ diện tích
- **Top N Selection**: Chỉ hiển thị N đối tượng có confidence cao nhất
- **Image Capture**: Chụp và lưu ảnh gốc
- **Debug Mode**: Tự động lưu ảnh có annotation và ảnh đã xử lý khi phát hiện đối tượng
- **Dark Theme**: Giao diện tối, thân thiện với mắt
- **Configurable**: Tất cả màu sắc và tham số có thể cấu hình từ file JSON

## Yêu cầu hệ thống

- Python 3.12 trở lên
- Camera (USB hoặc built-in)
- Hệ điều hành: Windows, Linux, macOS

## Cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/nguyentrungtin1709/YOLO11-Seg-Label-Detector.git
cd YOLO11-Seg-Label-Detector
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Bước 3: Cài đặt các gói phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra models

Đảm bảo các file model cần thiết đã có trong thư mục `models/`:

```
models/
├── yolo11n-seg-version-1-0-0.onnx                      # ONNX Runtime backend
├── yolo11n-seg-version-1-0-0_int8_openvino_model/      # OpenVINO Runtime backend
│   └── yolo11n-seg-version-1-0-0.xml
├── paddle/
│   └── PP-LCNet_x1_0_doc_ori/                          # Orientation classifier
└── wechat/                                              # WeChat QR models (optional)
    ├── detect.prototxt
    ├── detect.caffemodel
    ├── sr.prototxt
    └── sr.caffemodel
```

**Lưu ý:**
- YOLO model hỗ trợ 2 backends: ONNX Runtime (cross-platform) và OpenVINO Runtime (Intel-optimized)
- WeChat QR models là tùy chọn, chỉ cần khi dùng backend `"wechat"` cho QR detection
- Cấu hình backend trong `config/application_config.json`

## Khởi chạy ứng dụng

```bash
python main.py
```

## Hướng dẫn sử dụng

1. **Chọn camera**: Chọn camera từ dropdown "Camera"
2. **Bật camera**: Bật toggle "Camera Power" (màu cam)
3. **Bật detection**: Bật toggle "Enable Detection" (màu xanh lá)
4. **Điều chỉnh threshold**: Thay đổi giá trị "Confidence" nếu cần
5. **Chụp ảnh**: Nhấn nút "Capture Image" để lưu ảnh gốc
6. **Debug mode**: Bật toggle "Debug Mode" để tự động lưu ảnh có annotation

## Cấu trúc thư mục

```
label-detector/
├── config/
│   └── application_config.json   # Cấu hình ứng dụng
├── core/                     # Core layer (interfaces & implementations)
│   ├── interfaces/           # Abstraction layer (ICameraCapture, IDetector, IImageWriter, IImagePreprocessor, IImageEnhancer, IQrDetector, IComponentExtractor, IOcrExtractor, ITextProcessor)
│   ├── camera/               # Camera implementation (OpenCVCamera)
│   ├── detector/             # YOLO detector implementation (YOLODetector, OpenVINODetector)
│   ├── preprocessor/         # Preprocessing (GeometricTransformer, OrientationCorrector, DocumentPreprocessor)
│   ├── enhancer/             # Enhancement (BrightnessEnhancer, SharpnessEnhancer, ImageEnhancer)
│   ├── qr/                   # QR detection (ZxingQrDetector, WechatQrDetector, QrImagePreprocessor)
│   ├── extractor/            # Component extraction (LabelComponentExtractor)
│   ├── ocr/                  # OCR extraction (PaddleOcrExtractor)
│   ├── processor/            # Text processing (FuzzyMatcher, LabelTextProcessor)
│   └── writer/               # File writer implementation (LocalImageWriter)
├── data/                     # Reference data for fuzzy matching
│   ├── colors.json           # Valid colors database (4904 entries)
│   ├── products.json         # Valid product codes database (2172 entries)
│   └── sizes.json            # Valid sizes database (138 entries)
├── services/                 # Service layer (business logic)
│   ├── interfaces/           # Service interfaces (DIP)
│   │   ├── camera_service_interface.py
│   │   ├── detection_service_interface.py
│   │   ├── preprocessing_service_interface.py
│   │   ├── enhancement_service_interface.py
│   │   ├── qr_detection_service_interface.py
│   │   ├── component_extraction_service_interface.py
│   │   ├── ocr_service_interface.py
│   │   └── postprocessing_service_interface.py
│   └── impl/                 # Service implementations
│       ├── s1_camera_service.py
│       ├── s2_detection_service.py
│       ├── s3_preprocessing_service.py
│       ├── s4_enhancement_service.py
│       ├── s5_qr_detection_service.py
│       ├── s6_component_extraction_service.py
│       ├── s7_ocr_service.py
│       └── s8_postprocessing_service.py
├── ui/                       # UI layer (PySide6 widgets)
│   ├── main_window.py
│   ├── pipeline_orchestrator.py  # Pipeline coordination
│   └── widgets/
│       ├── camera_widget.py  # Video display with overlays
│       ├── config_panel.py   # Control panel
│       ├── preprocessed_image_widget.py  # Preprocessed image display
│       ├── ocr_result_widget.py  # OCR results display
│       └── toggle_switch.py  # Custom toggle widget
├── models/
│   ├── yolo11n-seg-version-x-x-x.onnx  # YOLO11n-seg model
│   └── paddle/
│       └── PP-LCNet_x1_0_doc_ori/      # PaddleOCR orientation model
├── output/
│   ├── captures/             # Ảnh chụp thủ công (raw)
│   └── debug/                # Ảnh debug tự động
│       ├── display/          # Ảnh có annotation (bbox + mask)
│       ├── original/         # Ảnh gốc (PNG)
│       ├── bbox/             # Crop theo bounding box
│       ├── mask/             # Crop theo mask (PNG với alpha)
│       ├── cropped/          # Ảnh sau crop, rotate, orientation fix
│       ├── preprocessing/    # Ảnh sau enhancement (kết quả cuối cùng)
│       └── txt/              # Tọa độ contour của mask
├── main.py                   # Entry point với Dependency Injection
├── requirements.txt          # Dependencies
└── README.md
```

## Cấu hình

File cấu hình: `config/application_config.json`

Cấu hình được tổ chức theo 8 bước trong pipeline:

### App Settings (Cài đặt chung)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `app.windowMinWidth` | Chiều rộng tối thiểu cửa sổ | `1000` |
| `app.windowMinHeight` | Chiều cao tối thiểu cửa sổ | `700` |
| `app.jpegQuality` | Chất lượng JPEG khi lưu ảnh (1-100) | `95` |
| `app.classNames` | Danh sách class để detect | `["label"]` |
| `app.captureDirectory` | Thư mục lưu ảnh chụp | `output/captures` |

### Debug Settings

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `debug.enabled` | Bật/tắt debug mode | `false` |
| `debug.basePath` | Thư mục gốc cho debug output | `output/debug` |
| `debug.saveCooldown` | Thời gian chờ giữa các lần lưu (giây) | `2.0` |
| `debug.performanceLogging.enabled` | Bật hiển thị performance metrics | `true` |

### S1: Camera Service

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s1_camera.frameWidth` | Chiều rộng khung hình | `640` |
| `s1_camera.frameHeight` | Chiều cao khung hình | `640` |
| `s1_camera.fps` | FPS camera | `60` |
| `s1_camera.maxCameraSearch` | Số camera tối đa tìm kiếm | `2` |

### S2: Detection Service

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s2_detection.backend` | Backend inference: `"onnx"` hoặc `"openvino"` | `"openvino"` |
| `s2_detection.modelPath` | Đường dẫn model (ONNX hoặc OpenVINO XML) | Tùy backend |
| `s2_detection.isSegmentation` | Bật chế độ segmentation | `true` |
| `s2_detection.inputSize` | Kích thước đầu vào model | `640` |
| `s2_detection.confidenceThreshold` | Ngưỡng confidence | `0.5` |
| `s2_detection.maxAreaRatio` | Lọc đối tượng > X% diện tích ảnh | `0.40` |
| `s2_detection.topNDetections` | Số đối tượng tối đa hiển thị | `2` |

#### OpenVINO Settings (khi backend = "openvino")

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s2_detection.openvino.numThreads` | Số CPU threads (0 = auto) | `2` |
| `s2_detection.openvino.numStreams` | Số inference streams (0 = auto) | `1` |
| `s2_detection.openvino.performanceHint` | Chế độ hiệu suất: `"LATENCY"` hoặc `"THROUGHPUT"` | `"LATENCY"` |
| `s2_detection.openvino.enableHyperThreading` | Sử dụng hyper-threading | `false` |
| `s2_detection.openvino.enableCpuPinning` | Pin threads vào CPU cores | `true` |

### S3: Preprocessing Service

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s3_preprocessing.enabled` | Bật/tắt preprocessing | `true` |
| `s3_preprocessing.forceLandscape` | Xoay ảnh về hướng ngang | `true` |
| `s3_preprocessing.aiOrientationFix` | Sửa ảnh ngược 180° bằng AI | `true` |
| `s3_preprocessing.aiConfidenceThreshold` | Ngưỡng confidence cho AI fix | `0.6` |
| `s3_preprocessing.paddleModelPath` | Đường dẫn model PaddleOCR | `models/paddle/PP-LCNet_x1_0_doc_ori` |
| `s3_preprocessing.displayWidth` | Chiều rộng hiển thị preview | `230` |
| `s3_preprocessing.displayHeight` | Chiều cao hiển thị preview | `100` |

### S4: Enhancement Service

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s4_enhancement.brightnessEnabled` | Bật tăng cường độ sáng | `true` |
| `s4_enhancement.brightnessClipLimit` | CLAHE clip limit (1.0-5.0) | `2.5` |
| `s4_enhancement.brightnessTileSize` | CLAHE tile grid size | `8` |
| `s4_enhancement.sharpnessEnabled` | Bật làm sắc nét | `false` |
| `s4_enhancement.sharpnessSigma` | Gaussian blur sigma | `1.0` |
| `s4_enhancement.sharpnessAmount` | Sharpen amount (1.0-3.0) | `1.5` |

### S5: QR Detection Service

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s5_qr_detection.enabled` | Bật/tắt QR detection | `true` |
| `s5_qr_detection.backend` | Backend: `"zxing"` hoặc `"wechat"` | `"wechat"` |

#### ZXing Backend Settings

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s5_qr_detection.zxing.tryRotate` | Thử barcode xoay 90°/270° | `true` |
| `s5_qr_detection.zxing.tryDownscale` | Thử downscale ảnh | `true` |

#### WeChat Backend Settings

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s5_qr_detection.wechat.modelDir` | Thư mục model WeChat QRCode | `models/wechat` |

#### QR Preprocessing Settings

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s5_qr_detection.preprocessing.enabled` | Bật preprocessing cho QR | `true` |
| `s5_qr_detection.preprocessing.mode` | Chế độ: `"minimal"` hoặc `"full"` | `"minimal"` |
| `s5_qr_detection.preprocessing.targetWidth` | Chiều rộng mục tiêu (pixels) | `640` |

### S6: Component Extraction Service

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s6_component_extraction.aboveQrWidthRatio` | Tỷ lệ chiều rộng vùng trên QR | `0.35` |
| `s6_component_extraction.aboveQrHeightRatio` | Tỷ lệ chiều cao vùng trên QR | `0.18` |
| `s6_component_extraction.belowQrWidthRatio` | Tỷ lệ chiều rộng vùng dưới QR | `0.65` |
| `s6_component_extraction.belowQrHeightRatio` | Tỷ lệ chiều cao vùng dưới QR | `0.55` |
| `s6_component_extraction.padding` | Padding (pixels) | `5` |
| `s6_component_extraction.aboveQrScaleFactor` | Scale factor cho vùng trên QR | `2.0` |

### S7: OCR Service

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s7_ocr.enabled` | Bật/tắt OCR | `true` |
| `s7_ocr.lang` | Ngôn ngữ OCR | `"en"` |
| `s7_ocr.precision` | Độ chính xác: `"fp32"`, `"fp16"`, `"int8"` | `"fp32"` |
| `s7_ocr.enableMkldnn` | Bật MKL-DNN acceleration | `true` |
| `s7_ocr.cpuThreads` | Số CPU threads | `2` |
| `s7_ocr.textDetThresh` | Ngưỡng detection | `0.15` |
| `s7_ocr.textDetBoxThresh` | Ngưỡng box | `0.15` |
| `s7_ocr.textRecScoreThresh` | Ngưỡng recognition | `0.3` |

### S8: Postprocessing Service (Fuzzy Matching)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s8_postprocessing.minFuzzyScore` | Ngưỡng similarity tối thiểu (0.0-1.0) | `0.90` |
| `s8_postprocessing.productsJsonPath` | Đường dẫn database sản phẩm | `data/products.json` |
| `s8_postprocessing.sizesJsonPath` | Đường dẫn database kích thước | `data/sizes.json` |
| `s8_postprocessing.colorsJsonPath` | Đường dẫn database màu sắc | `data/colors.json` |

### Visualization (Hiển thị)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `s2_detection.visualization.maskOpacity` | Độ trong suốt mask (0.0-1.0) | `0.4` |
| `s2_detection.visualization.maskColors` | Danh sách màu mask (BGR) | `[[128,0,128], ...]` |
| `s2_detection.visualization.boxColor` | Màu bounding box (BGR) | `[0, 255, 0]` |
| `s2_detection.visualization.textColor` | Màu text label (BGR) | `[0, 0, 0]` |
| `s2_detection.visualization.lineThickness` | Độ dày đường viền | `2` |
| `s2_detection.visualization.fontSize` | Kích thước font chữ | `0.8` |

## Debug Mode

Khi bật Debug Mode (`debug.enabled: true`), ứng dụng tự động lưu ảnh debug theo từng bước trong pipeline:

### Debug Directory Structure

```
output/debug/
├── s2_detection/
│   ├── annotated/           # Ảnh có annotation (bbox + mask + label)
│   ├── cropped/             # Ảnh crop theo mask (PNG với alpha channel)
│   └── masks/               # Tọa độ contour của mask (TXT)
├── s3_preprocessing/
│   ├── rotated/             # Ảnh sau khi rotate về landscape
│   ├── orientation/         # Ảnh sau AI orientation fix
│   └── final/               # Ảnh sau preprocessing hoàn chỉnh
├── s4_enhancement/
│   ├── brightness/          # Ảnh sau CLAHE
│   ├── sharpness/           # Ảnh sau sharpen
│   └── final/               # Ảnh sau enhancement hoàn chỉnh
├── s5_qr_detection/
│   ├── inputs/              # Ảnh input sau preprocessing
│   └── qr/                  # Kết quả QR detection (JSON)
├── s6_component_extraction/
│   ├── above_qr/            # Vùng trên QR
│   ├── below_qr/            # Vùng dưới QR
│   └── merged/              # Vùng merged đầy đủ
├── s7_ocr/
│   └── results/             # Kết quả OCR (JSON)
└── s8_postprocessing/
    └── results/             # Kết quả fuzzy matching (JSON)
```

### Debug Save Cooldown

Để tránh lưu quá nhiều ảnh, debug mode có cooldown time (`debug.saveCooldown: 2.0` giây) giữa các lần lưu.

## Performance Logging

Khi bật `debug.performanceLogging.enabled`, ứng dụng hiển thị:
- FPS thời gian thực trên status bar
- Thời gian xử lý từng bước: S1 (Camera) → S2 (Detection) → S3 (Preprocessing) → S4 (Enhancement) → S5 (QR) → S6 (Extraction) → S7 (OCR) → S8 (Postprocessing)

## Tài liệu bổ sung

- [QR-DETECTION-DOCS.md](config/QR-DETECTION-DOCS.md) - Hướng dẫn cấu hình QR Detection backends
- [YOLOv11-DOCS.md](config/YOLOv11-DOCS.md) - Hướng dẫn cấu hình YOLO Detection backends
- [SPECIFICATION.md](SPECIFICATION.md) - Đặc tả hệ thống
- [CHANGELOG.md](CHANGELOG.md) - Lịch sử thay đổi
- [VERSIONS.md](metadata/VERSIONS.md) - Thông tin phiên bản các tài nguyên ML

## Kiến trúc hệ thống

Dự án tuân thủ nguyên tắc SOLID và Clean Architecture với 3 layer:

1. **Core Layer** - Business logic độc lập với framework
   - Interfaces (abstractions)
   - Implementations (concrete classes)

2. **Service Layer** - Application logic
   - Service interfaces (DIP - Dependency Inversion)
   - Service implementations (8-step pipeline)

3. **UI Layer** - Presentation
   - PySide6 widgets
   - PipelineOrchestrator (dependency injection)

Chi tiết về kiến trúc: [ARCHITECTURE.md](ARCHITECTURE.md)


