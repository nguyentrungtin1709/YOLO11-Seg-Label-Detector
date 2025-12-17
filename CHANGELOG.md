# Changelog

Tất cả các thay đổi đáng chú ý của dự án sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2025-12-18

### Tổng quan
Phiên bản bổ sung **OCR Pipeline** hoàn chỉnh - trích xuất thông tin từ nhãn sản phẩm bao gồm QR code detection, component extraction, OCR text extraction, và post-processing với fuzzy matching.

### Added
- **QR Code Detection**: Phát hiện QR code bằng pyzbar, parse định dạng `MMDDYY-FACILITY-TYPE-ORDER-POSITION`
- **Component Extraction**: Cắt vùng text dựa trên vị trí QR code (above QR = position/quantity, below QR = product/size/color)
- **OCR Text Extraction**: Trích xuất text bằng PaddleOCR (CPU-only mode)
- **Fuzzy Matching**: So khớp text với database bằng Levenshtein và Jaro-Winkler similarity
- **Validation**: Kiểm tra position từ QR code khớp với position từ OCR
- **OCR Result Widget**: Widget hiển thị kết quả OCR trong UI (QR info, extracted fields, validation status)
- **Debug Output**: 
  - `output/debug/qr-code/`: JSON kết quả QR detection
  - `output/debug/components/`: Ảnh các vùng đã cắt
  - `output/debug/ocr-raw-text/`: JSON kết quả OCR thô
  - `output/debug/result/`: JSON kết quả cuối cùng
- **Data Files**: Database colors.json (4904), products.json (2172), sizes.json (138)

### Technical
- Thêm Core Interfaces:
  - `IQrDetector`, `IComponentExtractor`, `IOcrExtractor`, `ITextProcessor`
  - Data classes: `QrDetectionResult`, `ComponentResult`, `OcrResult`, `TextBlock`, `LabelData`
- Thêm Core Implementations:
  - `core/qr/PyzbarQrDetector`
  - `core/extractor/LabelComponentExtractor`
  - `core/ocr/PaddleOcrExtractor`
  - `core/processor/FuzzyMatcher`, `LabelTextProcessor`
- Thêm Service layer: `OcrPipelineService` với `OcrPipelineResult`
- Thêm UI: `OcrResultWidget`
- Dependencies mới: `pyzbar>=0.1.9`
- Config mới: `ocrPipeline.enabled`, `debugEnabled`, `qrDetector`, `componentExtractor`, `ocr`, `textProcessor`

---

## [1.1.0] - 2025-12-17

### Tổng quan
Phiên bản bổ sung tính năng **Image Preprocessing** và **Image Enhancement** - xử lý ảnh nhãn sau khi phát hiện bao gồm crop, rotate, sửa hướng ảnh bằng AI, tăng cường độ sáng và độ sắc nét.

### Added
- **Preprocessing Pipeline**: Crop và xoay ảnh dựa trên segmentation mask
- **Force Landscape**: Tự động xoay ảnh về hướng ngang (width >= height)
- **AI Orientation Fix**: Sử dụng PaddleOCR (PP-LCNet_x1_0_doc_ori) để phát hiện và sửa ảnh bị ngược 180°
- **Brightness Enhancement**: Tăng cường độ sáng ảnh tối bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Sharpness Enhancement**: Làm sắc nét ảnh mờ bằng Unsharp Mask
- **Preprocessed Image Widget**: Hiển thị ảnh đã xử lý trong UI bên dưới Detection config panel
- **Debug Save**: 
  - `output/debug/cropped/`: Ảnh sau crop, rotate, orientation fix
  - `output/debug/preprocessing/`: Ảnh sau enhancement (kết quả cuối cùng)
- **Local Model Storage**: Model PP-LCNet được lưu trong `models/paddle/` để portable

### Technical
- Thêm Core layer: 
  - `IImagePreprocessor` interface, `GeometricTransformer`, `OrientationCorrector`, `DocumentPreprocessor`
  - `IImageEnhancer` interface, `BrightnessEnhancer`, `SharpnessEnhancer`, `ImageEnhancer`
- Thêm Service layer: `PreprocessingService` với `FullPreprocessingResult`
- Thêm UI layer: `PreprocessedImageWidget`
- Dependencies mới: `paddleocr>=2.7.0`, `paddlepaddle>=2.5.0`
- Config mới: `preprocessing.enabled`, `forceLandscape`, `aiOrientationFix`, `paddleModelPath`, `enhancement.*`

---

## [1.0.0] - 2025-12-13

### Tổng quan
Phiên bản đầu tiên của ứng dụng **Label Detector** - công cụ desktop phát hiện và phân đoạn nhãn sản phẩm trong thời gian thực sử dụng mô hình YOLO11n-seg (Instance Segmentation).

### Added
- Kiến trúc 3 tầng (Core → Service → UI) theo nguyên tắc SOLID
- Phát hiện và phân đoạn nhãn với YOLO11n-seg (ONNX)
- Hiển thị segmentation mask với opacity tùy chỉnh
- Lọc kết quả theo kích thước (maxAreaRatio) và Top N detection
- Chế độ Debug: lưu ảnh annotated, crop bbox, crop mask (transparent), tọa độ contour
- Performance logging với FPS display trên status bar
- Giao diện dark theme với PySide6
- Cấu hình linh hoạt qua file JSON

### Technical
- Python 3.12+
- PySide6, OpenCV, ONNX Runtime
- Model: yolo11n-seg-version-1.0.1.onnx

---
