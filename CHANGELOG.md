# Changelog

Tất cả các thay đổi đáng chú ý của dự án sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
