# Changelog

Tất cả các thay đổi đáng chú ý của dự án sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án tuân theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-12-05

### Tổng quan
Phiên bản đầu tiên của ứng dụng **Label Detector** - công cụ desktop phát hiện nhãn sản phẩm trong thời gian thực sử dụng mô hình YOLO11n.

### Added

#### Kiến trúc
- Thiết kế kiến trúc 4 tầng theo nguyên tắc SOLID (UI → Service → Core → Infrastructure)
- Hệ thống Dependency Injection để quản lý các thành phần
- Cấu hình linh hoạt qua file JSON (`config/app_config.json`)

#### Tính năng Camera
- Tự động phát hiện và liệt kê camera trong hệ thống
- Chọn camera từ dropdown
- Bật/Tắt camera qua toggle switch
- Hiển thị video stream real-time

#### Tính năng Phát hiện
- Tích hợp model YOLO11n (định dạng ONNX) để phát hiện nhãn sản phẩm
- Bật/Tắt chức năng detection
- Điều chỉnh ngưỡng confidence (0.0 - 1.0)
- Hiển thị bounding box và label trên video

#### Tính năng Lưu ảnh
- Chụp và lưu ảnh gốc (không annotation) vào `output/captures/`
- Chế độ Debug: tự động lưu ảnh có annotation vào `output/debug/`
- Giới hạn tần suất lưu debug (1 giây/ảnh) để tránh quá tải

#### Giao diện
- Giao diện desktop với PySide6
- Dark theme
- Custom toggle switch widget kiểu iOS
- Status bar hiển thị thông báo

### Technical
- Python 3.8+
- PySide6 cho GUI
- OpenCV cho xử lý video
- ONNX Runtime cho inference model
- Kích thước frame: 640x640 (khớp với input model)

---
