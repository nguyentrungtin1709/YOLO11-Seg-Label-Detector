# QR Detection Backend Configuration Guide

## Overview

This document explains how to configure the QR code detector to switch between two detection backends: ZXing-cpp and OpenCV WeChat QRCode. It also covers the optional preprocessing pipeline.

## QR Code Format

The system expects QR codes in the following format:

```
MMDDYY-FACILITY-TYPE-ORDER-POSITION[/REVISION]
```

**Format Details:**
- `MMDDYY`: Date code (6 digits)
- `FACILITY`: Facility code (2 uppercase letters)
- `TYPE`: Order type (1 uppercase letter)
- `ORDER`: Order number (numeric, variable length)
- `POSITION`: Position number (numeric, variable length)
- `/REVISION`: Optional revision count (numeric)

**Examples:**
- `110125-VA-M-000002-2` (no revision)
- `110125-VA-M-000002-2/1` (revised once)
- `110125-VA-M-000002-2/2` (revised twice)

The detection system automatically parses this format and validates the structure using regex pattern matching.

## Backend Options

### ZXing-cpp
- **Library**: `zxing-cpp` (C++ library with Python bindings)
- **Type**: Traditional image processing
- **Speed**: Very fast (~5-15ms per detection)
- **Accuracy**: Good for standard QR codes
- **Dependencies**: `zxingcpp` package
- **Use case**: General purpose, real-time detection

### OpenCV WeChat QRCode
- **Library**: OpenCV's `wechat_qrcode` module
- **Type**: Deep learning based (CNN with super-resolution)
- **Speed**: Moderate (~20-50ms per detection)
- **Accuracy**: Better for difficult/damaged QR codes
- **Dependencies**: OpenCV with contrib modules + model files
- **Use case**: Challenging conditions, low quality images, small QR codes

## Configuration File

Edit the file: `config/application_config.json`

### Switching to ZXing Backend

```json
{
    "s5_qr_detection": {
        "enabled": true,
        "backend": "zxing",
        "zxing": {
            "tryRotate": true,
            "tryDownscale": true
        }
    }
}
```

### Switching to WeChat Backend

```json
{
    "s5_qr_detection": {
        "enabled": true,
        "backend": "wechat",
        "wechat": {
            "modelDir": "models/wechat"
        }
    }
}
```

## Configuration Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `true` | Enable/disable QR detection |
| `backend` | String | `"zxing"` | Backend selection: `"zxing"` or `"wechat"` |

### ZXing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tryRotate` | Boolean | `true` | Try rotated barcodes (90°/270°) |
| `tryDownscale` | Boolean | `true` | Try downscaled versions for better detection |

### WeChat Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelDir` | String | `"models/wechat"` | Directory containing model files |

## Preprocessing Pipeline

The preprocessing pipeline improves detection rate on difficult images. It runs **before** QR detection and applies image processing techniques.

**Important Notes:**
- Input from S4 Enhancement Service is already grayscale with CLAHE brightness enhancement applied
- Preprocessing here focuses on scaling and denoising only
- Scale factor is computed dynamically based on input image width and target width
- QR coordinates detected on scaled images are automatically scaled back to original size for S6 Component Extraction

### Preprocessing Modes

#### Disabled (Default)
```json
"preprocessing": {
    "enabled": false
}
```
No preprocessing - uses image directly from S4 Enhancement (grayscale with CLAHE).

#### Minimal Mode
```json
"preprocessing": {
    "enabled": true,
    "mode": "minimal",
    "targetWidth": 640
}
```
**Pipeline**: Scale only (fast)

- Scales image to target width (maintains aspect ratio)
- Scale factor computed dynamically: `targetWidth / originalWidth`
- Useful for small QR codes without adding extra processing overhead

#### Full Mode
```json
"preprocessing": {
    "enabled": true,
    "mode": "full",
    "targetWidth": 640
}
```
**Pipeline**: Scale → Denoise

| Step | Operation | Purpose |
|------|-----------|---------|
| 1. Scale | Resize to target width | Enlarge small QR codes for better detection |
| 2. Denoise | Median blur (3x3) | Remove salt-and-pepper noise |

### Preprocessing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable/disable preprocessing |
| `mode` | String | `"full"` | Mode: `"minimal"` or `"full"` |
| `targetWidth` | Integer | `640` | Target width for scaling (pixels). Scale factor computed dynamically as `targetWidth / originalWidth` |

## Requirements

### ZXing-cpp
Already included in `requirements.txt`:
```
zxingcpp>=2.2.0
```

### OpenCV WeChat QRCode
OpenCV with contrib modules (already included):
```
opencv-contrib-python>=4.8.0
```

### WeChat Model Files
Download and place in `models/wechat/`:
```
models/wechat/
├── detect.prototxt      # Detection network architecture
├── detect.caffemodel    # Detection network weights
├── sr.prototxt          # Super-resolution architecture
└── sr.caffemodel        # Super-resolution weights
```

**Download links**:
- [detect.prototxt](https://github.com/WeChatCV/opencv_3rdparty/raw/wechat_qrcode/detect.prototxt)
- [detect.caffemodel](https://github.com/WeChatCV/opencv_3rdparty/raw/wechat_qrcode/detect.caffemodel)
- [sr.prototxt](https://github.com/WeChatCV/opencv_3rdparty/raw/wechat_qrcode/sr.prototxt)
- [sr.caffemodel](https://github.com/WeChatCV/opencv_3rdparty/raw/wechat_qrcode/sr.caffemodel)

## Performance Comparison

| Metric | ZXing-cpp | WeChat QRCode |
|--------|-----------|---------------|
| Average detection time | 5-15 ms | 20-50 ms |
| First detection (cold) | 10-20 ms | 100-200 ms |
| Standard QR accuracy | Excellent | Excellent |
| Damaged QR accuracy | Good | Better |
| Low light accuracy | Good | Better |
| Small QR accuracy | Good | Better |
| Memory usage | Low | Higher |
| Model files required | No | Yes (4 files) |

## Usage Examples

### Example 1: Fast Real-time Detection
Best for good quality images, real-time applications:
```json
{
    "s5_qr_detection": {
        "enabled": true,
        "backend": "zxing",
        "zxing": {
            "tryRotate": true,
            "tryDownscale": true
        },
        "preprocessing": {
            "enabled": false
        }
    }
}
```

### Example 2: Challenging Conditions
Best for difficult images, damaged/small QR codes:
```json
{
    "s5_qr_detection": {
        "enabled": true,
        "backend": "wechat",
        "wechat": {
            "modelDir": "models/wechat"
        },
        "preprocessing": {
            "enabled": true,
            "mode": "full",
            "targetWidth": 640
        }
    }
}
```

### Example 3: Balanced Approach
Good balance between speed and accuracy:
```json
{
    "s5_qr_detection": {
        "enabled": true,
        "backend": "zxing",
        "zxing": {
            "tryRotate": true,
            "tryDownscale": true
        },
        "preprocessing": {
            "enabled": true,
            "mode": "minimal",
            "targetWidth": 640
        }
    }
}
```

## Debug Output

When debug is enabled (`debug.enabled: true`), QR detection saves:

### Debug Files Location
```
output/debug/s5_qr_detection/
├── inputs/                    # Preprocessed images before detection
│   ├── frame_001_full.png     # Full mode preprocessing
│   ├── frame_002_minimal.png  # Minimal mode preprocessing
│   └── frame_003_raw.png      # No preprocessing
└── qr/                        # Detection results
    ├── frame_001_qr.json      # QR data + parsed fields
    └── frame_002_qr.json
```

### Debug JSON Content
```json
{
    "frameId": "frame_001",
    "text": "110125-VA-M-000002-2",
    "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "rect": [x, y, width, height],
    "confidence": 1.0,
    "parsed": {
        "dateCode": "110125",
        "facility": "VA",
        "orderType": "M",
        "orderNumber": "000002",
        "position": "2",
        "revisionCount": null
    }
}
```

## Troubleshooting

### Backend not found error
- Check backend name spelling: `"zxing"` or `"wechat"`
- For ZXing: Ensure `zxingcpp` is installed: `pip install zxingcpp`
- For WeChat: Ensure `opencv-contrib-python` is installed

### WeChat model loading error
```
Error: Could not load WeChat QR model from models/wechat
```
- Verify all 4 model files exist in `models/wechat/`
- Check file names match exactly (case-sensitive)
- Re-download model files if corrupted

### No QR detected
1. Check if QR is visible and not too small
2. Try enabling preprocessing with `"mode": "full"`
3. Increase `targetWidth` to 800 or 1024 for very small QR codes
4. Try switching to WeChat backend (better for difficult conditions)
5. Enable debug to inspect preprocessed images

### Slow detection
- Switch from WeChat to ZXing backend (3-5x faster)
- Disable preprocessing or use `"mode": "minimal"`
- Reduce `targetWidth` if using preprocessing (e.g., 480 instead of 640)

## Best Practices

1. **Development**: Use ZXing backend for faster iteration
2. **Production**: Choose based on image quality:
   - Good quality → ZXing (faster)
   - Poor quality → WeChat with preprocessing (more robust)
3. **Testing**: Enable debug to inspect preprocessing results
4. **Optimization**: Start with preprocessing disabled, enable only if needed
5. **Small QR codes**: Use preprocessing with higher `targetWidth` (800 or 1024)

## Pipeline Integration

QR detection is Step 5 in the processing pipeline:

```
S4 Enhancement (Grayscale + CLAHE)
    ↓
S5 QR Detection
    ├── Preprocessing (optional)
    │   ├── Minimal: Scale
    │   └── Full: Scale → Denoise
    └── Detection (ZXing or WeChat)
        ↓
        Coordinates scaled back to original size
    ↓
S6 Component Extraction
```

**Note**: S5 receives **grayscale** image from S4 Enhancement. The preprocessing pipeline is designed to work with grayscale input. Enhancement (CLAHE + Sharpen) is already applied in S4, so S5 preprocessing focuses on other techniques.

## Additional Notes

- Both backends use the same QR content parsing (regex-based)
- Detection results are consistent across backends
- Backend selection does not affect other pipeline steps
- Configuration changes take effect immediately on application restart
