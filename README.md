# Label Detector

Ứng dụng desktop phát hiện và phân đoạn nhãn sản phẩm (product labels) trong thời gian thực sử dụng mô hình YOLO11n-seg (Instance Segmentation).

![Label Detector UI](assets/template.png)

## Tính năng

- **Camera Management**: Tự động phát hiện và chọn camera, bật/tắt camera
- **Instance Segmentation**: Phát hiện và phân đoạn nhãn với YOLO11n-seg (ONNX)
- **Mask Visualization**: Hiển thị segmentation mask với màu sắc và opacity tùy chỉnh
- **Image Preprocessing**: Crop, rotate và sửa hướng ảnh nhãn tự động
  - Crop theo minimum area rectangle từ segmentation mask
  - Force landscape orientation (width >= height)
  - AI 180° fix sử dụng PaddleOCR (PP-LCNet)
- **Image Enhancement**: Cải thiện chất lượng ảnh nhãn
  - Brightness Enhancement (CLAHE) - tăng cường độ sáng ảnh tối
  - Sharpness Enhancement (Unsharp Mask) - làm sắc nét ảnh mờ
- **OCR Pipeline**: Trích xuất thông tin từ nhãn sản phẩm
  - QR Code Detection (pyzbar) - phát hiện và parse QR code
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
git clone https://github.com/nguyentrungtin1709/yolov11-label-detector.git
cd yolov11-label-detector
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

### Bước 4: Kiểm tra model

Đảm bảo file model YOLO đã có trong thư mục `models/`:
```bash
# x-x-x là phiên bản model hiện tại. Ví dụ: 1-0-0
models/
└── yolo11n-seg-version-x-x-x.onnx
```

## Khởi chạy ứng dụng

```bash
python main.py
```

Hoặc chạy với chế độ debug logging:
```bash
DEBUG=true python main.py
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
│   └── app_config.json       # Cấu hình ứng dụng
├── core/                     # Core layer (interfaces & implementations)
│   ├── interfaces/           # Abstraction layer (ICameraCapture, IDetector, IImageWriter, IImagePreprocessor, IImageEnhancer, IQrDetector, IComponentExtractor, IOcrExtractor, ITextProcessor)
│   ├── camera/               # Camera implementation (OpenCVCamera)
│   ├── detector/             # YOLO detector implementation (YOLODetector)
│   ├── preprocessor/         # Preprocessing (GeometricTransformer, OrientationCorrector, DocumentPreprocessor)
│   ├── enhancer/             # Enhancement (BrightnessEnhancer, SharpnessEnhancer, ImageEnhancer)
│   ├── qr/                   # QR detection (PyzbarQrDetector)
│   ├── extractor/            # Component extraction (LabelComponentExtractor)
│   ├── ocr/                  # OCR extraction (PaddleOcrExtractor)
│   ├── processor/            # Text processing (FuzzyMatcher, LabelTextProcessor)
│   └── writer/               # File writer implementation (LocalImageWriter)
├── data/                     # Reference data for fuzzy matching
│   ├── colors.json           # Valid colors database (4904 entries)
│   ├── products.json         # Valid product codes database (2172 entries)
│   └── sizes.json            # Valid sizes database (138 entries)
├── services/                 # Service layer (business logic)
│   ├── camera_service.py     # Camera orchestration
│   ├── detection_service.py  # Detection + filtering logic
│   ├── image_saver_service.py # Image saving with annotations
│   ├── preprocessing_service.py # Image preprocessing orchestration
│   ├── ocr_pipeline_service.py # OCR pipeline orchestration
│   └── performance_logger.py # FPS and timing metrics
├── ui/                       # UI layer (PySide6 widgets)
│   ├── main_window.py
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

File cấu hình: `config/app_config.json`

### Tham số cơ bản

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `modelPath` | Đường dẫn model ONNX | `models/yolo11n-seg-version-1.0.1.onnx` |
| `isSegmentation` | Bật chế độ segmentation | `true` |
| `confidenceThreshold` | Ngưỡng confidence | `0.5` |
| `inputSize` | Kích thước đầu vào model | `640` |
| `maxCameraSearch` | Số camera tối đa tìm kiếm | `2` |

### Filter Settings (Lọc kết quả)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `filterSettings.maxAreaRatio` | Lọc đối tượng > X% diện tích ảnh | `0.15` |
| `filterSettings.topNDetections` | Số đối tượng tối đa hiển thị | `3` |

### Preprocessing Settings (Tiền xử lý ảnh)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `preprocessing.enabled` | Bật/tắt preprocessing | `true` |
| `preprocessing.forceLandscape` | Xoay ảnh về hướng ngang | `true` |
| `preprocessing.aiOrientationFix` | Sửa ảnh ngược 180° bằng AI | `true` |
| `preprocessing.aiConfidenceThreshold` | Ngưỡng confidence cho AI fix | `0.6` |
| `preprocessing.paddleModelPath` | Đường dẫn model PaddleOCR | `models/paddle/PP-LCNet_x1_0_doc_ori` |

### Enhancement Settings (Cải thiện chất lượng ảnh)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `preprocessing.enhancement.brightnessEnabled` | Bật tăng cường độ sáng | `true` |
| `preprocessing.enhancement.brightnessClipLimit` | CLAHE clip limit (1.0-5.0) | `2.5` |
| `preprocessing.enhancement.brightnessTileSize` | CLAHE tile grid size | `8` |
| `preprocessing.enhancement.sharpnessEnabled` | Bật làm sắc nét | `true` |
| `preprocessing.enhancement.sharpnessSigma` | Gaussian blur sigma | `1.0` |
| `preprocessing.enhancement.sharpnessAmount` | Sharpen amount (1.0-3.0) | `1.5` |

### OCR Pipeline Settings (Đọc nội dung nhãn)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `ocrPipeline.enabled` | Bật/tắt OCR pipeline | `true` |
| `ocrPipeline.debugEnabled` | Lưu debug output cho OCR | `true` |

#### QR Detector

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `ocrPipeline.qrDetector.symbolTypes` | Loại mã cần detect | `["QRCODE"]` |

#### Component Extractor

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `ocrPipeline.componentExtractor.aboveQrRatio` | Tỷ lệ vùng phía trên QR (% chiều cao ảnh) | `0.25` |
| `ocrPipeline.componentExtractor.belowQrRatio` | Tỷ lệ vùng phía dưới QR (% chiều cao ảnh) | `0.35` |
| `ocrPipeline.componentExtractor.paddingRatio` | Padding cho merged image | `0.02` |

#### OCR Extractor

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `ocrPipeline.ocr.language` | Ngôn ngữ OCR | `"en"` |
| `ocrPipeline.ocr.useGpu` | Sử dụng GPU (tắt để chạy CPU) | `false` |
| `ocrPipeline.ocr.showLog` | Hiển thị log PaddleOCR | `false` |
| `ocrPipeline.ocr.dropScore` | Ngưỡng lọc kết quả OCR | `0.5` |

#### Text Processor (Fuzzy Matching)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `ocrPipeline.textProcessor.colorsPath` | Đường dẫn database màu sắc | `"data/colors.json"` |
| `ocrPipeline.textProcessor.productsPath` | Đường dẫn database sản phẩm | `"data/products.json"` |
| `ocrPipeline.textProcessor.sizesPath` | Đường dẫn database kích thước | `"data/sizes.json"` |
| `ocrPipeline.textProcessor.matchThreshold` | Ngưỡng similarity tối thiểu (0.0-1.0) | `0.7` |
| `ocrPipeline.textProcessor.levenshteinWeight` | Trọng số Levenshtein (0.0-1.0) | `0.4` |
| `ocrPipeline.textProcessor.jaroWinklerWeight` | Trọng số Jaro-Winkler (0.0-1.0) | `0.6` |

### Visualization (Hiển thị)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `maskOpacity` | Độ trong suốt mask (0.0-1.0) | `0.4` |
| `maskColors` | Danh sách màu mask (BGR) | `[[128,0,128], ...]` |
| `boxColor` | Màu bounding box (BGR) | `[0, 255, 0]` |
| `textColor` | Màu text label (BGR) | `[0, 0, 0]` |

## Debug Mode Output

Khi bật Debug Mode và phát hiện đối tượng, ứng dụng tự động lưu (với cooldown 2 giây):

### Detection Debug

| Thư mục | Nội dung | Định dạng |
|---------|----------|----------|
| `debug/display/` | Ảnh có annotation (mask + bbox + label) | JPEG |
| `debug/original/` | Ảnh gốc không annotation | PNG |
| `debug/bbox/` | Crop theo bounding box | JPEG |
| `debug/mask/` | Crop theo mask với nền trong suốt | PNG (BGRA) |
| `debug/cropped/` | Ảnh sau crop, rotate, orientation fix | JPEG |
| `debug/preprocessing/` | Ảnh sau enhancement (kết quả cuối cùng) | JPEG |
| `debug/txt/` | Tọa độ contour của mask | TXT |

### OCR Pipeline Debug

| Thư mục | Nội dung | Định dạng |
|---------|----------|----------|
| `debug/qr-code/` | Ảnh QR code đã detect với bbox | JPEG |
| `debug/components/` | Vùng above/below QR đã cắt | JPEG |
| `debug/ocr-raw-text/` | Raw text từ PaddleOCR | TXT |
| `debug/result/` | Kết quả OCR pipeline cuối cùng | JSON |

## Performance Logging

Khi bật `performanceLogging.enabled`, ứng dụng hiển thị:
- FPS thời gian thực trên status bar
- Thời gian xử lý: preprocess, inference, postprocess, filter

## Tài liệu

- [SPECIFICATION.md](SPECIFICATION.md) - Đặc tả hệ thống
- [CHANGELOG.md](CHANGELOG.md) - Lịch sử thay đổi
- [VERSIONS.md](metadata/VERSIONS.md) - Thông tin phiên bản các tài nguyên ML (dataset, model, notebook, experiments)


