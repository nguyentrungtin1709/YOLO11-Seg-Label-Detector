# Scripts - Hướng dẫn sử dụng

Tài liệu hướng dẫn chạy các script tiện ích trong dự án Label Detector.

---

## Danh sách Scripts

| Script | Mô tả |
|--------|-------|
| `detection.py` | Xử lý batch ảnh qua pipeline phát hiện nhãn (S2-S8) |
| `compare_backends.py` | So sánh hiệu năng giữa OpenVINO và ONNX backends |
| `filters/qr-errors-filter.py` | Lọc và sao chép ảnh bị lỗi QR detection |

---

## 1. detection.py

### Mô tả

Script xử lý hàng loạt ảnh từ thư mục qua pipeline phát hiện nhãn. Thay thế camera real-time bằng batch processing. Hỗ trợ các bước S2-S8 trong pipeline:

- S2: Detection - Phát hiện nhãn bằng YOLO
- S3: Preprocessing - Crop, rotate, fix orientation
- S4: Enhancement - Tăng cường độ sáng, độ nét
- S5: QR Detection - Phát hiện và giải mã QR code
- S6: Component Extraction - Trích xuất vùng text
- S7: OCR - Nhận dạng văn bản
- S8: Postprocessing - Fuzzy matching và validation

### Tham số

| Tham số | Viết tắt | Mô tả | Mặc định |
|---------|----------|-------|----------|
| `--input` | `-i` | Thư mục chứa ảnh đầu vào | `samples` |
| `--config` | `-c` | Đường dẫn file cấu hình JSON | `config/application_config.json` |
| `--limit` | `-n` | Số lượng ảnh tối đa xử lý | Không giới hạn |
| `--debug` | `-d` | Bật chế độ debug (lưu output vào `output/debug/`) | Tắt |

### Ví dụ

```bash
python scripts/detection.py --input samples/set1 --debug --limit 50
```

Lệnh trên sẽ:
- Đọc ảnh từ thư mục `samples/set1`
- Bật chế độ debug để lưu kết quả vào `output/debug/`
- Chỉ xử lý tối đa 50 ảnh

### Output

Khi bật `--debug`, kết quả được lưu tự động vào các thư mục trong `output/debug/`:
- `s2_detection/` - Kết quả phát hiện nhãn
- `s3_preprocessing/` - Ảnh sau preprocessing
- `s4_enhancement/` - Ảnh sau enhancement
- `s5_qr_detection/` - Kết quả QR detection
- `s6_component_extraction/` - Vùng text đã trích xuất
- `s7_ocr/` - Kết quả OCR
- `s8_postprocessing/` - Kết quả fuzzy matching
- `timing/` - Thông tin thời gian xử lý

---

## 2. compare_backends.py

### Mô tả

Script so sánh hiệu năng giữa hai backends: OpenVINO và ONNX. Đọc các file JSON detection từ thư mục output, tính toán thống kê và tạo báo cáo với biểu đồ.

### Tham số

Script này không có tham số dòng lệnh. Các đường dẫn được cấu hình trong code:
- Input OpenVINO: `output/debug/s2_detection/OpenVINO/`
- Input ONNX: `output/debug/s2_detection/ONNX/`
- Output report: `output/debug/s2_detection/backend_comparison_report.txt`
- Output chart: `output/debug/s2_detection/backend_comparison_chart.png`

### Yêu cầu

Trước khi chạy, cần có dữ liệu detection từ cả hai backends:
1. Chạy `detection.py` với backend OpenVINO, lưu output vào `OpenVINO/`
2. Chạy `detection.py` với backend ONNX, lưu output vào `ONNX/`

### Ví dụ

```bash
python scripts/compare_backends.py
```

### Output

- **Report**: File text chứa thống kê chi tiết (average, median, min, max, std)
- **Chart**: Biểu đồ bar so sánh thời gian xử lý trung bình

Thống kê được tính toán:
- Số mẫu xử lý
- Thời gian xử lý trung bình (ms)
- Thời gian median, min, max
- Độ lệch chuẩn
- FPS trung bình
- Phần trăm cải thiện hiệu năng

---

## 3. filters/qr-errors-filter.py

### Mô tả

Script lọc và sao chép các ảnh bị lỗi QR detection. Phân tích file `batch_summary_*.json` trong thư mục timing, tìm các frame có lỗi "No QR code detected" và sao chép ảnh enhancement tương ứng vào thư mục riêng để phân tích.

### Tham số

| Tham số | Viết tắt | Mô tả | Mặc định |
|---------|----------|-------|----------|
| `--debug-dir` | | Đường dẫn thư mục debug output | `output/debug` |
| `--verbose` | `-v` | Bật logging chi tiết | Tắt |

### Yêu cầu

Trước khi chạy, cần có:
- Thư mục `output/debug/timing/` chứa file `batch_summary_*.json`
- Thư mục `output/debug/s4_enhancement/` chứa ảnh enhancement

### Ví dụ

```bash
python scripts/filters/qr-errors-filter.py --debug-dir output/debug --verbose
```

### Output

- Ảnh lỗi được sao chép vào: `output/debug/s5_qr_detection/errors/{batch_name}/`
- Báo cáo tổng hợp hiển thị trên console:
  - Tên batch
  - Tổng số frame
  - Số lượng lỗi
  - Tỷ lệ lỗi (%)
  - Tổng số file đã sao chép

---

## Lưu ý chung

1. **Working Directory**: Tất cả scripts nên được chạy từ thư mục gốc của dự án:
   ```bash
   cd /path/to/YOLO11-Seg-Label-Detector
   python scripts/detection.py
   ```

2. **Virtual Environment**: Đảm bảo đã kích hoạt môi trường ảo trước khi chạy:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. **Dependencies**: Các script sử dụng các thư viện trong `requirements.txt`. Đảm bảo đã cài đặt đầy đủ.
