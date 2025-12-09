# Label Detector - Đặc tả Hệ thống

## 1. Tổng quan

**Label Detector** là ứng dụng desktop cho phép phát hiện nhãn sản phẩm (product labels) trong thời gian thực từ camera sử dụng **Instance Segmentation** với mô hình YOLO11n-seg. Ứng dụng không chỉ nhận diện và đánh dấu vị trí các nhãn bằng bounding box mà còn **vẽ mask pixel-level** với opacity có thể cấu hình lên từng đối tượng được phát hiện để người dùng có thể nhìn thấy rõ hơn hình dạng thực tế của nhãn.

### 1.1 Tính năng nổi bật

- **Instance Segmentation**: Phân đoạn pixel-level cho từng đối tượng
- **Lọc theo kích thước**: Chỉ hiển thị các đối tượng có diện tích nhỏ hơn ngưỡng
- **Top N Detection**: Chỉ hiển thị N đối tượng có confidence cao nhất
- **Màu sắc tùy chỉnh**: Màu mask và bounding box có thể cấu hình từ file config

## 2. Giao diện người dùng

### 2.1 Bố cục tổng thể

![Giao diện ứng dụng](Template.png)

Giao diện ứng dụng được thiết kế với bố cục ngang, gồm 2 phần chính:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Label Detector                                 │
├─────────────────────────────────────────────┬───────────────────────────────┤
│                                             │                               │
│                                             │  ┌─────────────────────────┐  │
│                                             │  │ Camera 0            ▼   │  │
│                                             │  │ Camera Power      [●━]  │  │
│                                             │  └─────────────────────────┘  │
│                                             │                               │
│          Vùng hiển thị Video/Camera         │  ┌─────────────────────────┐  │
│                 (Bên trái)                  │  │      Detection          │  │
│                                             │  ├─────────────────────────┤  │
│                                             │  │ Enable Detection [●━]   │  │
│           "Camera not connected"            │  │ Debug Mode       [●━]   │  │
│            (Khi camera chưa bật)            │  │ Confidence      [0.50]  │  │
│                                             │  └─────────────────────────┘  │
│                                             │                               │
│                                             │  ┌─────────────────────────┐  │
│                                             │  │      Capture Image      │  │
│                                             │  └─────────────────────────┘  │
│                                             │  ┌─────────────────────────┐  │
│                                             │  │          Close          │  │
│                                             │  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Chi tiết các thành phần

#### 2.2.1 Vùng hiển thị Video (Bên trái - chiếm 3/4 chiều ngang)

| Thuộc tính | Mô tả |
|------------|-------|
| Kích thước tối thiểu | 640 x 640 pixels |
| Nền mặc định | Màu tối (#1a1a1a) |
| Trạng thái mặc định | Hiển thị text "Camera not connected" |
| Khi camera bật | Hiển thị video stream real-time từ camera |
| Khi phát hiện đối tượng | Vẽ bounding box và label lên video |

#### 2.2.2 Panel cấu hình (Bên phải - chiều rộng cố định 250px)

Panel cấu hình chứa các nhóm điều khiển sau:

##### a) Nhóm Camera

| Thành phần | Loại | Mô tả |
|------------|------|-------|
| Camera Dropdown | ComboBox | Danh sách các camera khả dụng trong hệ thống |
| Camera Power | Toggle Switch (màu cam) | Bật/Tắt camera đã chọn |

##### b) Nhóm Detection

| Thành phần | Loại | Mô tả | Giá trị mặc định |
|------------|------|-------|------------------|
| Enable Detection | Toggle Switch (màu xanh lá) | Bật/Tắt chức năng phát hiện đối tượng | OFF |
| Debug Mode | Toggle Switch (màu xanh dương) | Bật/Tắt chế độ debug (tự động lưu ảnh) | OFF |
| Confidence | SpinBox (0.00 - 1.00) | Ngưỡng độ tin cậy để lọc kết quả | 0.50 |

##### c) Các nút hành động

| Thành phần | Màu sắc | Mô tả |
|------------|---------|-------|
| Capture Image | Xanh lá (#4CAF50) | Chụp và lưu ảnh thô từ camera |
| Close | Đỏ (#f44336) | Đóng ứng dụng |

## 3. Danh sách tính năng

### 3.1 Quản lý Camera

| ID | Tính năng | Mô tả |
|----|-----------|-------|
| F01 | Phát hiện camera | Tự động quét và liệt kê tất cả các camera khả dụng trong hệ thống khi khởi động |
| F02 | Chọn camera | Cho phép người dùng chọn camera muốn sử dụng từ danh sách dropdown |
| F03 | Bật camera | Kích hoạt camera đã chọn và bắt đầu hiển thị video stream lên màn hình |
| F04 | Tắt camera | Dừng video stream và giải phóng camera để các ứng dụng khác có thể sử dụng |
| F05 | Chuyển đổi camera | Khi đang bật camera, cho phép chuyển sang camera khác mà không cần tắt thủ công |

### 3.2 Phát hiện đối tượng

| ID | Tính năng | Mô tả |
|----|-----------|-------|
| F06 | Bật/Tắt phát hiện | Cho phép kích hoạt hoặc vô hiệu hóa chức năng phát hiện và phân đoạn nhãn |
| F07 | Phát hiện và phân đoạn nhãn | Nhận diện các nhãn sản phẩm xuất hiện trong từng frame video và tạo segmentation mask pixel-level |
| F08 | Hiển thị kết quả | Vẽ mask màu (opacity cấu hình được) lên các nhãn được phát hiện, đồng thời vẽ bounding box để tham chiếu |
| F09 | Hiển thị thông tin | Hiển thị tên class và độ tin cậy (confidence score) trên mỗi detection |
| F10 | Điều chỉnh ngưỡng | Cho phép thay đổi ngưỡng confidence để lọc các kết quả phát hiện |
| F11 | Lọc kết quả | Chỉ hiển thị các kết quả có confidence >= ngưỡng đã thiết lập |
| F12 | Lọc theo kích thước | Lọc bỏ các đối tượng có diện tích > maxAreaRatio (% diện tích ảnh) |
| F13 | Chọn Top N | Chỉ hiển thị N đối tượng có confidence cao nhất |

### 3.3 Chụp và lưu ảnh

| ID | Tính năng | Mô tả |
|----|-----------|-------|
| F14 | Chụp ảnh thủ công | Lưu frame hiện tại dưới dạng ảnh gốc (không có bounding box và mask) khi nhấn nút Capture |
| F15 | Chế độ Debug | Khi bật, tự động lưu frame có chứa đối tượng được phát hiện (có vẽ segmentation mask và bounding box) |
| F16 | Giới hạn tần suất lưu | Trong chế độ Debug, chỉ lưu tối đa 1 ảnh mỗi giây để tránh quá tải |

### 3.4 Điều khiển ứng dụng

| ID | Tính năng | Mô tả |
|----|-----------|-------|
| F17 | Đóng ứng dụng | Đóng ứng dụng an toàn, giải phóng camera và tài nguyên |
| F18 | Hiển thị trạng thái | Thông báo các sự kiện quan trọng qua status bar (kết nối camera, phát hiện đối tượng, lưu ảnh, ...) |

## 4. Định dạng lưu trữ

### 4.1 Cấu trúc thư mục

```
output/
├── captures/          # Ảnh chụp thủ công (ảnh gốc)
│   └── capture_YYYYMMDD_HHMMSS_mmm.jpg
└── debug/             # Ảnh debug tự động (có bounding box)
    └── debug_YYYYMMDD_HHMMSS_mmm.jpg
```

### 4.2 Quy tắc đặt tên file

| Loại | Thư mục | Định dạng tên | Nội dung |
|------|---------|---------------|----------|
| Capture | `output/captures/` | `capture_{timestamp}.jpg` | Ảnh gốc từ camera (không có annotation) |
| Debug | `output/debug/` | `debug_{timestamp}.jpg` | Ảnh có vẽ segmentation mask (opacity 40%) và bounding box với label |

*Timestamp format: YYYYMMDD_HHMMSS_mmm (năm-tháng-ngày_giờ-phút-giây_miligiây)*

## 5. Thông tin Model

| Thuộc tính | Giá trị |
|------------|---------|
| Kiến trúc | YOLO11n-seg (nano - Instance Segmentation) |
| Định dạng | ONNX |
| Kích thước input | 640 x 640 pixels |
| Số class | 1 (label) |
| File model | `models/yolo11n-seg_best.onnx` |
| **Output 1** | **Bounding boxes + Mask coefficients: [1, 116, 8400]** |
| **Output 2** | **Proto masks: [1, 32, 160, 160]** |
| **Visualization** | **Colored masks với opacity cấu hình + Bounding boxes** |

## 6. Cấu hình lọc kết quả (Filter Settings)

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `maxAreaRatio` | float | 0.15 | Lọc bỏ đối tượng có diện tích > 15% diện tích ảnh |
| `topNDetections` | int | 3 | Chỉ giữ lại N đối tượng có confidence cao nhất |

## 7. Cấu hình màu sắc

| Tham số | Kiểu | Mô tả |
|---------|------|-------|
| `boxColor` | [B, G, R] | Màu bounding box (BGR) |
| `textColor` | [B, G, R] | Màu text label (BGR) |
| `maskOpacity` | float | Độ trong suốt của mask (0.0 - 1.0) |
| `maskColors` | [[B,G,R], ...] | Danh sách màu cho mask theo thứ tự đối tượng |

## 8. Yêu cầu phi chức năng

### 8.1 Hiệu năng
- Xử lý real-time với FPS >= 15 trên CPU
- Độ trễ phát hiện < 100ms

### 8.2 Giao diện
- Giao diện tối (dark theme)
- Responsive khi resize cửa sổ
- Kích thước cửa sổ tối thiểu: 900 x 700 pixels

### 8.3 Tương thích
- Hệ điều hành: Windows, Linux, macOS
- Python 3.8+
- Hỗ trợ nhiều loại camera (USB, built-in)

## 9. Xử lý các trường hợp đặc biệt

| Tình huống | Xử lý |
|------------|-------|
| Không tìm thấy camera | Hiển thị "No cameras found" trong dropdown |
| Camera không mở được | Hiển thị thông báo lỗi, tắt toggle Camera Power |
| Không load được model | Hiển thị cảnh báo khi khởi động |
| Thư mục output không tồn tại | Tự động tạo thư mục khi lưu ảnh |
| Nút Capture khi camera tắt | Nút bị vô hiệu hóa (disabled) |

