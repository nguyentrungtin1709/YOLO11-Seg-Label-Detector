# ML Pipelines - Label Detection Pipeline

## Tổng quan

Hệ thống trích xuất thông tin từ nhãn sản phẩm được thiết kế theo kiến trúc pipeline gồm 8 bước tuần tự. Mỗi bước đảm nhận một nhiệm vụ cụ thể, nhận đầu vào từ bước trước và truyền kết quả cho bước tiếp theo.

Luồng xử lý tổng quát:

1. **S1 - Camera**: Thu nhận hình ảnh từ camera
2. **S2 - Detection**: Phát hiện vùng nhãn bằng YOLO
3. **S3 - Preprocessing**: Cắt, xoay và sửa hướng nhãn
4. **S4 - Enhancement**: Tăng cường độ sáng và độ nét
5. **S5 - QR Detection**: Phát hiện và giải mã mã QR
6. **S6 - Component Extraction**: Trích xuất vùng chứa text
7. **S7 - OCR**: Nhận dạng ký tự từ ảnh
8. **S8 - Postprocessing**: Fuzzy matching và xác thực kết quả

---

## S1 - Camera Service

### Mục đích

Thu nhận hình ảnh từ camera và tạo định danh duy nhất cho mỗi khung hình.

### Xử lý

1. Thiết lập kết nối với camera thông qua OpenCV.
2. Tìm kiếm và mở camera khả dụng trong hệ thống (quét tối đa N camera).
3. Chụp khung hình với độ phân giải được cấu hình (width x height).
4. Tạo Frame ID duy nhất dựa trên timestamp theo format `frame_YYYYMMDD_HHMMSS_milliseconds`.
5. Đóng gói ảnh, Frame ID và timestamp thành đối tượng kết quả.

### Đầu ra

- Ảnh gốc từ camera
- Frame ID để theo dõi xuyên suốt pipeline
- Timestamp ghi nhận thời điểm chụp

---

## S2 - Detection Service

### Mục đích

Phát hiện vùng chứa nhãn trong ảnh bằng mô hình YOLO Instance Segmentation.

### Xử lý

1. Nạp mô hình YOLO11 Segmentation từ file ONNX.
2. Tiền xử lý ảnh đầu vào: resize về kích thước input của model (640x640).
3. Chạy inference để phát hiện các vùng nhãn trong ảnh.
4. Lọc kết quả phát hiện theo các tiêu chí:
   - Confidence threshold: Loại bỏ detection có độ tin cậy thấp.
   - Max area ratio: Loại bỏ detection chiếm diện tích quá lớn (có thể là false positive).
   - Top N detections: Chỉ giữ lại N detection có confidence cao nhất.
5. Trích xuất bounding box và segmentation mask cho mỗi detection.

### Đầu ra

- Danh sách detection với bounding box (x, y, width, height)
- Segmentation mask (polygon) để crop chính xác vùng nhãn
- Confidence score của mỗi detection

---

## S3 - Preprocessing Service

### Mục đích

Cắt vùng nhãn từ ảnh gốc, xoay về hướng chuẩn và sửa lỗi hướng ngược.

### Xử lý

1. **Crop bằng Segmentation Mask**
   - Sử dụng polygon từ S2 để tạo mask.
   - Áp dụng phép biến đổi hình học để cắt chính xác vùng nhãn.
   - Loại bỏ nền và chỉ giữ lại vùng nhãn.

2. **Xoay về hướng Landscape**
   - Phát hiện hướng của nhãn (dọc hay ngang).
   - Nếu nhãn đang ở hướng dọc (portrait), xoay 90 độ về hướng ngang (landscape).
   - Đảm bảo nhãn luôn ở định dạng ngang để OCR hoạt động tốt hơn.

3. **AI Orientation Fix (Sửa lỗi ngược 180 độ)**
   - Sử dụng mô hình PaddlePaddle để phân loại hướng của document.
   - Phát hiện nếu nhãn bị ngược 180 độ.
   - Tự động xoay 180 độ nếu phát hiện nhãn bị ngược.
   - Áp dụng ngưỡng confidence để tránh sửa sai.

### Đầu ra

- Ảnh nhãn đã được crop và căn chỉnh đúng hướng
- Thông tin về các phép biến đổi đã áp dụng

---

## S4 - Enhancement Service

### Mục đích

Tăng cường chất lượng ảnh để cải thiện độ chính xác OCR.

### Xử lý

1. **Brightness Enhancement (Cân bằng độ sáng)**
   - Chuyển ảnh từ không gian màu BGR sang LAB.
   - Áp dụng thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L (lightness).
   - CLAHE chia ảnh thành các ô nhỏ (tile) và cân bằng histogram cục bộ.
   - Giới hạn contrast (clip limit) để tránh khuếch đại nhiễu.
   - Chuyển ngược về không gian BGR.

2. **Sharpness Enhancement (Làm nét)**
   - Áp dụng kỹ thuật Unsharp Masking.
   - Tạo phiên bản mờ của ảnh bằng Gaussian Blur với sigma cấu hình được.
   - Trừ ảnh mờ khỏi ảnh gốc để tạo detail mask.
   - Cộng detail mask vào ảnh gốc với hệ số amount để làm nét các cạnh và text.

### Đầu ra

- Ảnh đã được tăng cường độ sáng và độ nét
- Chất lượng text rõ ràng hơn cho bước OCR

---

## S5 - QR Detection Service

### Mục đích

Phát hiện và giải mã mã QR trên nhãn để lấy thông tin đơn hàng.

### Xử lý

1. **Tiền xử lý ảnh**
   - Chuyển ảnh từ BGR sang Grayscale để tăng tốc độ xử lý.

2. **Phát hiện mã QR**
   - Sử dụng thư viện zxing-cpp (C++ binding) để quét mã QR.
   - Thử nhiều hướng xoay (0, 90, 180, 270 độ) nếu cần.
   - Thử các phiên bản downscale để phát hiện mã QR nhỏ hoặc bị nhiễu.

3. **Giải mã nội dung**
   - Đọc nội dung text từ mã QR.
   - Parse theo format chuẩn: `MMDDYY-FACILITY-TYPE-ORDER-POSITION`
   - Trích xuất các trường:
     - Date Code: Ngày tạo đơn hàng (MMDDYY)
     - Facility: Mã cơ sở sản xuất (VD: VA)
     - Order Type: Loại đơn hàng (M: Multi, S: Single)
     - Order Number: Số đơn hàng
     - Position: Vị trí sản phẩm trong đơn

4. **Trích xuất vị trí**
   - Lấy tọa độ 4 góc của mã QR (polygon).
   - Tính bounding box từ polygon.

### Đầu ra

- Nội dung text của mã QR
- Dữ liệu đã parse (dateCode, facility, orderType, orderNumber, position)
- Polygon 4 góc của mã QR (dùng cho S6)

---

## S6 - Component Extraction Service

### Mục đích

Trích xuất các vùng chứa text dựa trên vị trí tương đối với mã QR.

### Xử lý

1. **Xác định vùng trích xuất**
   - Sử dụng polygon của mã QR làm điểm tham chiếu.
   - Tính toán vùng "Above QR" (phía trên mã QR):
     - Thường chứa Product Code
     - Kích thước theo tỷ lệ cấu hình (VD: 35% width, 20% height)
   - Tính toán vùng "Below QR" (phía dưới mã QR):
     - Thường chứa Size và Color
     - Kích thước theo tỷ lệ cấu hình (VD: 65% width, 45% height)

2. **Crop các vùng**
   - Cắt vùng Above QR từ ảnh.
   - Cắt vùng Below QR từ ảnh.
   - Thêm padding xung quanh để đảm bảo không cắt mất text.

3. **Merge vùng text**
   - Ghép các vùng đã crop thành một ảnh duy nhất.
   - Việc merge giúp OCR xử lý một lần thay vì nhiều lần, tăng hiệu suất.

### Đầu ra

- Ảnh merged chứa tất cả vùng text cần OCR
- Thông tin về vị trí các vùng trong ảnh merged

---

## S7 - OCR Service

### Mục đích

Nhận dạng và trích xuất text từ các vùng ảnh đã được tách.

### Xử lý

1. **Khởi tạo PaddleOCR**
   - Nạp mô hình OCR với ngôn ngữ tiếng Anh.
   - Cấu hình thiết bị xử lý (CPU hoặc GPU).

2. **Text Detection**
   - Phát hiện vùng chứa text trong ảnh.
   - Sử dụng ngưỡng detection threshold để lọc vùng có độ tin cậy thấp.
   - Trả về bounding box của từng dòng text.

3. **Text Recognition**
   - Nhận dạng ký tự trong từng vùng text đã phát hiện.
   - Áp dụng ngưỡng recognition score để lọc kết quả kém.
   - Trả về text đã nhận dạng kèm confidence score.

4. **Tổng hợp kết quả**
   - Gom các text block thành danh sách.
   - Mỗi block chứa: text content, bounding box, confidence score.

### Đầu ra

- Danh sách Text Block với nội dung và vị trí
- Confidence score cho mỗi text block

---

## S8 - Postprocessing Service

### Mục đích

Sửa lỗi OCR bằng fuzzy matching và xác thực kết quả với dữ liệu từ mã QR.

### Xử lý

1. **Fuzzy Matching với Database**
   - So khớp mờ kết quả OCR với các database:
     - Products Database: Danh sách mã sản phẩm hợp lệ
     - Sizes Database: Danh sách kích thước hợp lệ
     - Colors Database: Danh sách màu sắc hợp lệ
   - Sử dụng thuật toán fuzzy matching (VD: Levenshtein distance, ratio matching).
   - Áp dụng ngưỡng minimum score để chấp nhận kết quả sửa lỗi.
   - Tự động sửa lỗi OCR phổ biến (VD: "0" thành "O", "1" thành "I").

2. **Trích xuất thông tin nhãn**
   - Xác định Product Code từ text blocks.
   - Xác định Size từ text blocks.
   - Xác định Color từ text blocks.
   - Map về giá trị chuẩn trong database.

3. **Validation với QR Data**
   - So sánh thông tin trích xuất được với dữ liệu từ mã QR.
   - Kiểm tra tính nhất quán giữa các trường.
   - Đánh dấu kết quả là Valid hoặc Invalid.

4. **Tổng hợp Label Data**
   - Đóng gói tất cả thông tin vào đối tượng LabelData.
   - Bao gồm: productCode, size, color, validation status.

### Đầu ra

- LabelData chứa thông tin nhãn đã được xác thực
- Trạng thái Valid/Invalid
- Chi tiết về quá trình matching và validation

---

## Luồng dữ liệu

```
+--------+     +--------+     +--------+     +--------+
|   S1   |---->|   S2   |---->|   S3   |---->|   S4   |
| Camera |     | Detect |     |  Prep  |     |Enhance |
+--------+     +--------+     +--------+     +--------+
    |              |              |              |
  Frame        Detection       Cropped       Enhanced
  Image        + Mask         Oriented        Image
                                Image
                                                 |
                                                 v
+--------+     +--------+     +--------+     +--------+
|   S8   |<----|   S7   |<----|   S6   |<----|   S5   |
|  Post  |     |  OCR   |     |Extract |     |   QR   |
+--------+     +--------+     +--------+     +--------+
    |              |              |              |
  Label         Text           Merged          QR Data
  Data         Blocks          Regions        + Polygon
```

---

## Nguyên tắc thiết kế

### Single Responsibility Principle (SRP)

Mỗi service chỉ đảm nhận một nhiệm vụ duy nhất:
- S1 chỉ xử lý camera
- S2 chỉ xử lý detection
- S3 chỉ xử lý preprocessing
- ...

### Dependency Inversion Principle (DIP)

- Các service phụ thuộc vào abstraction (interface) thay vì implementation cụ thể.
- VD: S2 phụ thuộc vào IDetector, không phụ thuộc trực tiếp vào YOLODetector.

### Open/Closed Principle (OCP)

- Dễ dàng mở rộng thêm service mới hoặc thay đổi implementation.
- VD: Thay ZxingQrDetector bằng PyzbarQrDetector mà không cần sửa S5.

### Configurability

- Tất cả tham số được đọc từ file cấu hình JSON.
- Không hard-code threshold, path, hay kích thước.
