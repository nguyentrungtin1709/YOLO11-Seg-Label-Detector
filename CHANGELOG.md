# Changelog

Tất cả các thay đổi đáng chú ý của dự án sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.4.0] - 2025-12-21

### Tổng quan
Phiên bản thêm hỗ trợ **OpenVINO Runtime** backend cho YOLO detector theo nguyên tắc OCP (Open/Closed Principle). Hệ thống giờ có thể chọn giữa ONNX Runtime (cross-platform) và OpenVINO Runtime (Intel-optimized) thông qua config.

### Added
- **OpenVINO Backend Support**: Hỗ trợ OpenVINO Runtime cho inference tốc độ cao trên Intel hardware
  - `core/detector/openvino_detector.py`: Implementation IDetector sử dụng OpenVINO Runtime
  - `core/detector/detector_factory.py`: Factory pattern để tạo detector dựa trên backend
  - INT8 quantization support (2-4x nhanh hơn ONNX FP32 trên Intel CPU)
- **Backend Selection**: Chọn backend qua config `s2_detection.backend` ("onnx" hoặc "openvino")
- **Factory Pattern**: Encapsulation việc tạo detector, dễ mở rộng cho backends mới
- **Documentation**: 
  - `config/YOLOv11-DOCS.md`: Hướng dẫn cấu hình backend ONNX/OpenVINO
  - `ARCHITECTURE.md`: Mô tả kiến trúc hệ thống hiện tại
  - `PLAN.md`: Kế hoạch triển khai với architecture diagrams
- **Test Suite**: `scripts/test_openvino.py` để verify implementation

### Changed
- **DetectorFactory**: S2DetectionService giờ sử dụng `createDetector()` thay vì hard-code `YOLODetector`
- **ConfigService**: Thêm method `getDetectionBackend()` để đọc backend từ config
- **PipelineOrchestrator**: Truyền backend parameter vào S2DetectionService
- **application_config.json**: Thêm field `backend` với default "onnx" (backward compatible)
- **requirements.txt**: Thêm OpenVINO Runtime như optional dependency với hướng dẫn

### Technical Details
- **OCP Compliance**: Mở rộng (thêm OpenVINO) mà không sửa code hiện tại
- **DIP Compliance**: Services phụ thuộc `IDetector` interface, không phụ thuộc concrete class
- **Backward Compatible**: Default backend = "onnx", không breaking changes
- **Model Format**: 
  - ONNX backend: `.onnx` file (FP32, ~6.5MB)
  - OpenVINO backend: `.xml` + `.bin` files (INT8, ~1.6MB)
- **Performance Benchmark** (7 samples, Intel CPU):
  - OpenVINO: 116.05ms avg detection (1.69 FPS)
  - ONNX: 188.96ms avg detection (1.57 FPS)
  - Improvement: 38.6% faster với OpenVINO

### Files Modified
- `core/detector/__init__.py`: Export factory functions
- `services/impl/config_service.py`: +1 method `getDetectionBackend()`
- `services/impl/s2_detection_service.py`: +1 parameter `backend`, sử dụng factory
- `ui/pipeline_orchestrator.py`: Truyền backend từ config
- `config/application_config.json`: +5 dòng (backend field + comments)
- `requirements.txt`: +8 dòng (OpenVINO documentation)

### Files Created
- `core/detector/openvino_detector.py` (~450 dòng)
- `core/detector/detector_factory.py` (~250 dòng)
- `config/YOLOv11-DOCS.md` (~200 dòng)
- `ARCHITECTURE.md` (~300 dòng)
- `scripts/test_openvino.py` (~350 dòng)
- `PLAN.md` (~700 dòng)

### Files Removed
- `docs/OPENVINO_GUIDE.md`: Replaced by `config/YOLOv11-DOCS.md`

### Testing
- All 6 tests passed trong test suite
- Factory pattern works correctly
- Backend availability detection
- ONNX detector creation
- OpenVINO detector creation (nếu installed)
- Error handling cho invalid backend
- Model loading và inference verification
- Tensor naming bug fix (support models without named tensors)

### Migration Guide
Xem `config/YOLOv11-DOCS.md` để biết hướng dẫn chi tiết về:
- Cấu hình backend trong application_config.json
- So sánh ONNX vs OpenVINO
- Cài đặt OpenVINO Runtime
- Performance comparison
- Troubleshooting

---

## [1.3.0] - 2025-12-19

### Tổng quan
Phiên bản tái cấu trúc **Services Layer** theo pattern DIP (Dependency Inversion Principle) và cải thiện **OCR Position/Quantity Recovery** với QR validation.

### Added
- **Pipeline Orchestrator**: Lớp điều phối trung tâm quản lý 8 services trong pipeline
- **Debug Cooldown**: Cơ chế giới hạn tần suất lưu debug (2 giây) để tránh quá tải
- **Pipeline Timing**: Lưu thời gian xử lý của từng service vào `output/debug/timing/`
- **Position/Quantity Recovery**: Phục hồi format `X/Y` khi OCR đọc sai "/" thành "1", "|", "l", "I", "!", "t", "i", "j"
  - Pattern 1: Format chuẩn với validation `quantity >= position`
  - Pattern 2: Recovery sử dụng QR position + separator detection

### Changed
- **Services Architecture**: Tái cấu trúc theo DIP pattern
  - Mỗi service tự khởi tạo core component thay vì nhận từ bên ngoài
  - Services nhận config parameters qua constructor
  - `PipelineOrchestrator` đọc config và tạo tất cả services
- **LabelTextProcessor**: Cập nhật `_tryParsePositionQuantity()` với qrResult parameter
  - Thêm validation: `quantity >= position` (e.g., "3/5" valid, "5/3" invalid)
  - Thêm validation: OCR position phải khớp QR position

### Removed
- **Old Service Files**: Xóa các file service cũ không còn sử dụng
  - `services/camera_service.py` → replaced by `services/impl/s1_camera_service.py`
  - `services/detection_service.py` → replaced by `services/impl/s2_detection_service.py`
  - `services/image_saver_service.py` → integrated into pipeline services
  - `services/ocr_pipeline_service.py` → replaced by S5-S8 services
  - `services/performance_logger.py` → replaced by PipelineOrchestrator timing
  - `services/preprocessing_service.py` → replaced by S3/S4 services
- **Unsafe OCR Pattern**: Loại bỏ Pattern 3 (fallback không có QR validation)

### Technical
- Cập nhật `services/__init__.py` để reference services mới trong `services/impl/`
- Thêm `POSITION_RECOVERY_SEPARATORS` trong `LabelTextProcessor`
- Cập nhật `PLAN.md` với kiến trúc v2.0 và service interfaces

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
