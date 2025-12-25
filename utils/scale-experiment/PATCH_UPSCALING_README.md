# Hướng dẫn: Thêm tính năng Image Upscaling cho QR Detection

Tính năng này giúp **upscale ảnh nhỏ** sau bước preprocessing để cải thiện độ chính xác của QR code detection.

## 📋 Yêu cầu

- Repository: [YOLO11-Seg-Label-Detector](https://github.com/nguyentrungtin1709/YOLO11-Seg-Label-Detector)
- Branch: `dev`
- Không cần cài đặt thêm dependencies

## 🚀 Cách cài đặt

### Bước 1: Clone repo gốc (nếu chưa có)

```bash
git clone https://github.com/nguyentrungtin1709/YOLO11-Seg-Label-Detector.git
cd YOLO11-Seg-Label-Detector
git checkout dev
```

### Bước 2: Thay thế 4 files sau

Copy và ghi đè các files từ thư mục patch vào repo:

| File nguồn | Đích |
|------------|------|
| `s3_preprocessing_service.py` | `services/impl/s3_preprocessing_service.py` |
| `config_service.py` | `services/impl/config_service.py` |
| `pipeline_orchestrator.py` | `ui/pipeline_orchestrator.py` |
| `application_config.json` | `config/application_config.json` |

### Bước 3: Chạy ứng dụng

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## ⚙️ Cấu hình

Trong `config/application_config.json`, section `s3_preprocessing`:

```json
{
    "s3_preprocessing": {
        "minWidth": 300,
        "minHeight": 200
    }
}
```

- `minWidth`: Chiều rộng tối thiểu (pixels)
- `minHeight`: Chiều cao tối thiểu (pixels)

Ảnh nhỏ hơn kích thước này sẽ được **tự động upscale** bằng `cv2.INTER_CUBIC`.

## 📝 Log output

Khi upscaling xảy ra, bạn sẽ thấy log:

```
[frame_xxx] Upscaled image from 150x100 to 300x200 (scale=2.00x) for better QR detection
```

## 🔧 Thay đổi chi tiết

| File | Thay đổi |
|------|----------|
| `s3_preprocessing_service.py` | Thêm method `_upscaleIfNeeded()`, params `minWidth`, `minHeight` |
| `config_service.py` | Thêm `getPreprocessingMinWidth()`, `getPreprocessingMinHeight()` |
| `pipeline_orchestrator.py` | Truyền `minWidth`, `minHeight` vào S3PreprocessingService |
| `application_config.json` | Thêm config `minWidth`, `minHeight` |
