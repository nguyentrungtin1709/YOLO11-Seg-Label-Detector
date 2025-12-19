# Label Detection Pipeline - Tóm tắt kỹ thuật

## Tổng quan

Pipeline trích xuất thông tin từ nhãn sản phẩm gồm 8 bước tuần tự:

1. **S1 - Camera**: Thu nhận hình ảnh từ camera
2. **S2 - Detection**: Phát hiện vùng nhãn bằng YOLO
3. **S3 - Preprocessing**: Cắt, xoay và sửa hướng nhãn
4. **S4 - Enhancement**: Tăng cường độ sáng và độ nét
5. **S5 - QR Detection**: Phát hiện và giải mã mã QR
6. **S6 - Component Extraction**: Trích xuất vùng chứa text
7. **S7 - OCR**: Nhận dạng ký tự từ ảnh
8. **S8 - Postprocessing**: Fuzzy matching và xác thực kết quả

---

## Chi tiết từng bước

### S1 - Camera Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Thu nhận khung hình từ camera, tạo Frame ID duy nhất |
| Kỹ thuật | OpenCV VideoCapture |
| Output | Ảnh gốc + Frame ID + Timestamp |

---

### S2 - Detection Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Phát hiện vùng chứa nhãn trong ảnh |
| Kỹ thuật | YOLO11 Instance Segmentation (ONNX Runtime) |
| Lọc kết quả | Ngưỡng độ tin cậy, Tỷ lệ diện tích tối đa, Giữ lại N detection tốt nhất |
| Output | Bounding box + Segmentation mask |

---

### S3 - Preprocessing Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Cắt vùng nhãn, xoay về hướng chuẩn, sửa lỗi ngược 180 độ |
| Kỹ thuật | Geometric Transform (OpenCV), PaddlePaddle Orientation Classification |
| Output | Ảnh nhãn đã crop và căn chỉnh đúng hướng |

---

### S4 - Enhancement Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Tăng cường độ sáng và độ nét ảnh |
| Kỹ thuật | CLAHE trên không gian màu LAB (brightness), Unsharp Masking (sharpness) |
| Output | Ảnh đã tăng cường chất lượng |

---

### S5 - QR Detection Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Phát hiện và giải mã mã QR trên nhãn |
| Kỹ thuật | zxing-cpp (C++ barcode library) |
| Format QR | MMDDYY-FACILITY-TYPE-ORDER-POSITION |
| Output | QR data (dateCode, facility, orderType, orderNumber, position) + Polygon 4 góc |

---

### S6 - Component Extraction Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Trích xuất vùng text dựa trên vị trí mã QR |
| Kỹ thuật | Region-based extraction với tỷ lệ cấu hình, Image concatenation |
| Vùng trích xuất | Above QR (Product Code), Below QR (Size, Color) |
| Output | Ảnh merged chứa các vùng text |

---

### S7 - OCR Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Nhận dạng ký tự từ ảnh text |
| Kỹ thuật | PaddleOCR (Text Detection + Text Recognition) |
| Output | Danh sách Text Block (text, position, confidence) |

---

### S8 - Postprocessing Service

| Hạng mục | Chi tiết |
|----------|----------|
| Xử lý | Sửa lỗi OCR và xác thực kết quả với QR data |
| Kỹ thuật | Fuzzy Matching (Levenshtein distance) với database Products/Sizes/Colors |
| Validation | So sánh OCR result với QR data |
| Output | LabelData (productCode, size, color, isValid) |

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Object Detection | YOLO11 Instance Segmentation |
| Inference Runtime | ONNX Runtime |
| Image Processing | OpenCV |
| Orientation Fix | PaddlePaddle |
| QR Detection | zxing-cpp |
| OCR | PaddleOCR |
| Text Matching | RapidFuzz (Fuzzy Matching) |
| Configuration | JSON-based config |
