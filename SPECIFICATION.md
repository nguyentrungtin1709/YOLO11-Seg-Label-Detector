# Label Detector - Đặc tả Hệ thống

## 1. Tổng quan

**Label Detector** là ứng dụng desktop cho phép phát hiện nhãn sản phẩm (product labels) trong thời gian thực từ camera sử dụng **Instance Segmentation** với mô hình YOLO11n-seg. Ứng dụng không chỉ nhận diện và đánh dấu vị trí các nhãn bằng bounding box mà còn **vẽ mask pixel-level** với opacity có thể cấu hình lên từng đối tượng được phát hiện để người dùng có thể nhìn thấy rõ hơn hình dạng thực tế của nhãn.

### 1.1 Tính năng nổi bật

- **Instance Segmentation**: Phân đoạn pixel-level cho từng đối tượng
- **Image Preprocessing**: Crop, rotate và sửa hướng ảnh nhãn tự động bằng AI
- **Image Enhancement**: Tăng cường độ sáng (CLAHE) và độ sắc nét (Unsharp Mask)
- **Lọc theo kích thước**: Chỉ hiển thị các đối tượng có diện tích nhỏ hơn ngưỡng
- **Top N Detection**: Chỉ hiển thị N đối tượng có confidence cao nhất
- **Màu sắc tùy chỉnh**: Màu mask và bounding box có thể cấu hình từ file config

## 2. Giao diện người dùng

### 2.1 Bố cục tổng thể

![Giao diện ứng dụng](assets/template.png)

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
│                                             │  │  Preprocessed Image     │  │
│                                             │  │  (Cropped & Rotated)    │  │
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

##### c) Vùng hiển thị ảnh đã xử lý (Preprocessed Image)

| Thuộc tính | Mô tả |
|------------|-------|
| Kích thước | 230 x 100 pixels (cấu hình được) |
| Nền mặc định | Màu tối với text "Preprocessed Label" |
| Khi có detection | Hiển thị ảnh nhãn đã được crop, rotate và sửa hướng |
| Scale | Tự động fit theo kích thước giữ tỷ lệ |

##### d) Các nút hành động

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

### 3.3 Tiền xử lý ảnh (Image Preprocessing)

| ID | Tính năng | Mô tả |
|----|-----------|-------|
| F14 | Crop theo mask | Crop ảnh nhãn dựa trên minimum area rectangle từ segmentation mask |
| F15 | Rotate | Xoay ảnh để nhãn nằm thẳng |
| F16 | Force Landscape | Tự động xoay ảnh về hướng ngang (width >= height) |
| F17 | AI Orientation Fix | Sử dụng PaddleOCR (PP-LCNet) để phát hiện và sửa ảnh bị ngược 180° |
| F18 | Brightness Enhancement | Tăng cường độ sáng ảnh tối bằng CLAHE (tùy chọn) |
| F19 | Sharpness Enhancement | Làm sắc nét ảnh mờ bằng Unsharp Mask (tùy chọn) |
| F20 | Hiển thị kết quả | Hiển thị ảnh đã xử lý trong widget bên dưới Detection config |

### 3.4 Chụp và lưu ảnh

| ID | Tính năng | Mô tả |
|----|-----------|-------|
| F21 | Chụp ảnh thủ công | Lưu frame hiện tại dưới dạng ảnh gốc (không có bounding box và mask) khi nhấn nút Capture |
| F22 | Chế độ Debug | Khi bật, tự động lưu đầy đủ: ảnh annotated, ảnh gốc, crop bbox, crop mask, ảnh cropped, ảnh enhanced, và tọa độ contour |
| F23 | Giới hạn tần suất lưu | Trong chế độ Debug, chỉ lưu tối đa 1 lần mỗi 2 giây để tránh quá tải |

### 3.5 Điều khiển ứng dụng

| ID | Tính năng | Mô tả |
|----|-----------|-------|
| F24 | Đóng ứng dụng | Đóng ứng dụng an toàn, giải phóng camera và tài nguyên |
| F25 | Hiển thị trạng thái | Thông báo các sự kiện quan trọng qua status bar (kết nối camera, phát hiện đối tượng, lưu ảnh, ...) |
| F26 | Performance Logging | Hiển thị FPS và thời gian xử lý (preprocess, inference, postprocess) trên status bar khi được bật |

## 4. Định dạng lưu trữ

### 4.1 Cấu trúc thư mục

```
output/
├── captures/              # Ảnh chụp thủ công (ảnh gốc)
│   └── capture_YYYYMMDD_HHMMSS_mmm.jpg
└── debug/                 # Ảnh debug tự động
    ├── display/           # Ảnh với annotation (mask + bbox + label)
    │   └── debug_YYYYMMDD_HHMMSS_mmm.jpg
    ├── original/          # Ảnh gốc không annotation
    │   └── frame_YYYYMMDD_HHMMSS_mmm.png
    ├── bbox/              # Crop theo bounding box
    │   └── bbox_YYYYMMDD_HHMMSS_mmm_{idx}.jpg
    ├── mask/              # Crop theo mask với alpha channel
    │   └── mask_YYYYMMDD_HHMMSS_mmm_{idx}.png
    ├── cropped/           # Ảnh sau crop, rotate, orientation fix
    │   └── cropped_YYYYMMDD_HHMMSS_mmm_{idx}.jpg
    ├── preprocessing/     # Ảnh sau enhancement (kết quả cuối cùng)
    │   └── preprocessed_YYYYMMDD_HHMMSS_mmm.jpg
    └── txt/               # Tọa độ contour của mask
        └── mask_YYYYMMDD_HHMMSS_mmm_{idx}.txt
```

### 4.2 Quy tắc đặt tên file

| Loại | Thư mục | Định dạng tên | Nội dung |
|------|---------|---------------|----------|
| Capture | `captures/` | `capture_{timestamp}.jpg` | Ảnh gốc từ camera (không có annotation) |
| Display | `debug/display/` | `debug_{timestamp}.jpg` | Ảnh có vẽ segmentation mask và bounding box |
| Original | `debug/original/` | `frame_{timestamp}.png` | Ảnh gốc tại thời điểm phát hiện |
| BBox Crop | `debug/bbox/` | `bbox_{timestamp}_{idx}.jpg` | Crop ảnh theo bounding box |
| Mask Crop | `debug/mask/` | `mask_{timestamp}_{idx}.png` | Crop ảnh theo mask với nền trong suốt (BGRA) |
| Cropped | `debug/cropped/` | `cropped_{timestamp}_{idx}.jpg` | Ảnh sau crop, rotate, orientation fix |
| Preprocessed | `debug/preprocessing/` | `preprocessed_{timestamp}.jpg` | Ảnh sau enhancement (kết quả cuối cùng) |
| Contour | `debug/txt/` | `mask_{timestamp}_{idx}.txt` | Tọa độ x,y của contour mask |

*Timestamp format: YYYYMMDD_HHMMSS_mmm (năm-tháng-ngày_giờ-phút-giây_miligiây)*
*idx: Index của detection trong frame (0, 1, 2, ...)*

### 4.3 Định dạng file TXT (Contour Coordinates)

```
# Class: label
# Confidence: 0.8765
# BBox: 100,50,300,250
# Format: x,y coordinates of contour points
#
# Contour 0 (150 points)
120,55
121,56
...
#
```

## 5. Thông tin Model

### 5.1 YOLO Model (Instance Segmentation)

| Thuộc tính | Giá trị |
|------------|----------|
| Kiến trúc | YOLO11n-seg (nano - Instance Segmentation) |
| Định dạng | ONNX |
| Kích thước input | 640 x 640 pixels |
| Số class | 1 (label) |
| File model | `models/yolo11n-seg-version-x-x-x.onnx` |
| **Output 1** | **Bounding boxes + Mask coefficients: [1, 116, 8400]** |
| **Output 2** | **Proto masks: [1, 32, 160, 160]** |
| **Visualization** | **Colored masks với opacity cấu hình + Bounding boxes** |

### 5.2 PaddleOCR Model (Orientation Classification)

| Thuộc tính | Giá trị |
|------------|----------|
| Model | PP-LCNet_x1_0_doc_ori |
| Framework | PaddlePaddle |
| Chức năng | Phân loại hướng ảnh tài liệu (0°, 90°, 180°, 270°) |
| Sử dụng | Phát hiện và sửa ảnh bị ngược 180° |
| Đường dẫn | `models/paddle/PP-LCNet_x1_0_doc_ori/` |

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

## 8. Cấu hình Preprocessing

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `preprocessing.enabled` | bool | true | Bật/tắt chức năng preprocessing |
| `preprocessing.forceLandscape` | bool | true | Tự động xoay ảnh về hướng ngang |
| `preprocessing.aiOrientationFix` | bool | true | Sử dụng AI để sửa ảnh ngược 180° |
| `preprocessing.aiConfidenceThreshold` | float | 0.6 | Ngưỡng confidence cho AI orientation fix |
| `preprocessing.paddleModelPath` | string | models/paddle/PP-LCNet_x1_0_doc_ori | Đường dẫn model PaddleOCR |
| `preprocessing.displayWidth` | int | 230 | Chiều rộng widget hiển thị |
| `preprocessing.displayHeight` | int | 100 | Chiều cao widget hiển thị |

## 8.1 Cấu hình Enhancement

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `preprocessing.enhancement.brightnessEnabled` | bool | true | Bật tăng cường độ sáng (CLAHE) |
| `preprocessing.enhancement.brightnessClipLimit` | float | 2.5 | CLAHE clip limit (1.0 - 5.0) |
| `preprocessing.enhancement.brightnessTileSize` | int | 8 | CLAHE tile grid size |
| `preprocessing.enhancement.sharpnessEnabled` | bool | true | Bật làm sắc nét (Unsharp Mask) |
| `preprocessing.enhancement.sharpnessSigma` | float | 1.0 | Gaussian blur sigma |
| `preprocessing.enhancement.sharpnessAmount` | float | 1.5 | Sharpen amount (1.0 - 3.0) |

## 9. Yêu cầu phi chức năng

### 9.1 Hiệu năng
- Xử lý real-time với FPS >= 15 trên CPU
- Độ trễ phát hiện < 100ms
- Target FPS có thể cấu hình (mặc định: 60 FPS)
- Performance logging tùy chọn với FPS display trên status bar

### 9.2 Giao diện
- Giao diện tối (dark theme)
- Responsive khi resize cửa sổ
- Kích thước cửa sổ tối thiểu: 900 x 700 pixels (cấu hình được)

### 9.3 Tương thích
- Hệ điều hành: Windows, Linux, macOS
- Python 3.12+
- Hỗ trợ nhiều loại camera (USB, built-in)

## 10. Xử lý các trường hợp đặc biệt

| Tình huống | Xử lý |
|------------|-------|
| Không tìm thấy camera | Hiển thị "No cameras found" trong dropdown |
| Camera không mở được | Hiển thị thông báo lỗi, tắt toggle Camera Power |
| Không load được YOLO model | Hiển thị cảnh báo khi khởi động |
| Không load được PaddleOCR model | Vô hiệu hóa AI orientation fix, sử dụng các bước khác |
| Thư mục output không tồn tại | Tự động tạo thư mục khi lưu ảnh |
| Nút Capture khi camera tắt | Nút bị vô hiệu hóa (disabled) |

