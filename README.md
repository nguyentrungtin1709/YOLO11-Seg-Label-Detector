# Label Detector

Ứng dụng desktop phát hiện và phân đoạn nhãn sản phẩm (product labels) trong thời gian thực sử dụng mô hình YOLO11n-seg (Instance Segmentation).

![Label Detector UI](Template.png)

## Tính năng

- 📷 **Camera Management**: Tự động phát hiện và chọn camera, bật/tắt camera
- 🔍 **Instance Segmentation**: Phát hiện và phân đoạn nhãn với YOLO11n-seg (ONNX)
- 🎭 **Mask Visualization**: Hiển thị segmentation mask với màu sắc và opacity tùy chỉnh
- 🎯 **Adjustable Threshold**: Điều chỉnh ngưỡng confidence (0.0 - 1.0)
- 📐 **Size Filtering**: Lọc bỏ đối tượng quá lớn theo tỷ lệ diện tích
- 🏆 **Top N Selection**: Chỉ hiển thị N đối tượng có confidence cao nhất
- 📸 **Image Capture**: Chụp và lưu ảnh gốc
- 🐛 **Debug Mode**: Tự động lưu ảnh có annotation khi phát hiện đối tượng
- 🎨 **Dark Theme**: Giao diện tối, thân thiện với mắt
- ⚙️ **Configurable**: Tất cả màu sắc và tham số có thể cấu hình từ file JSON

## Yêu cầu hệ thống

- Python 3.8 trở lên
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
```
models/
└── yolo11n-seg_best.onnx
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
│   ├── interfaces/           # Abstraction layer
│   ├── camera/               # Camera implementation
│   ├── detector/             # YOLO detector implementation
│   └── writer/               # File writer implementation
├── services/                 # Service layer (business logic)
│   ├── camera_service.py
│   ├── detection_service.py  # Includes filtering logic
│   └── image_saver_service.py
├── ui/                       # UI layer (PySide6 widgets)
│   ├── main_window.py
│   └── widgets/
├── models/
│   └── yolo11n-seg_best.onnx # YOLO11n-seg model
├── output/
│   ├── captures/             # Ảnh chụp (raw)
│   └── debug/                # Ảnh debug (có annotation)
├── main.py                   # Entry point
├── requirements.txt          # Dependencies
└── README.md
```

## Cấu hình

File cấu hình: `config/app_config.json`

### Tham số cơ bản

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `modelPath` | Đường dẫn model ONNX | `models/yolo11n-seg_best.onnx` |
| `isSegmentation` | Bật chế độ segmentation | `true` |
| `confidenceThreshold` | Ngưỡng confidence | `0.5` |
| `inputSize` | Kích thước đầu vào model | `640` |
| `maxCameraSearch` | Số camera tối đa tìm kiếm | `2` |

### Filter Settings (Lọc kết quả)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `filterSettings.maxAreaRatio` | Lọc đối tượng > X% diện tích ảnh | `0.15` |
| `filterSettings.topNDetections` | Số đối tượng tối đa hiển thị | `3` |

### Visualization (Hiển thị)

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `maskOpacity` | Độ trong suốt mask (0.0-1.0) | `0.4` |
| `maskColors` | Danh sách màu mask (BGR) | `[[128,0,128], ...]` |
| `boxColor` | Màu bounding box (BGR) | `[0, 255, 0]` |
| `textColor` | Màu text label (BGR) | `[0, 0, 0]` |

## Tài liệu

- [SPECIFICATION.md](SPECIFICATION.md) - Đặc tả hệ thống
- [ARCHITECTURE.md](ARCHITECTURE.md) - Tài liệu kiến trúc
- [CHANGELOG.md](CHANGELOG.md) - Lịch sử thay đổi


