# QR Detection Backend Configuration Guide

## Overview

This document explains how to configure the QR code detector to switch between two detection backends: ZXing-cpp and OpenCV WeChat QRCode. It also covers the optional preprocessing pipeline.

## Backend Options

### ZXing-cpp (Default)
- **Library**: `zxing-cpp` (C++ library with Python bindings)
- **Type**: Traditional image processing
- **Speed**: Very fast (~5-15ms per detection)
- **Accuracy**: Good for standard QR codes
- **Dependencies**: `zxingcpp` package
- **Use case**: General purpose, real-time detection

### OpenCV WeChat QRCode
- **Library**: OpenCV's `wechat_qrcode` module
- **Type**: Deep learning based (CNN)
- **Speed**: Moderate (~20-50ms per detection)
- **Accuracy**: Better for difficult/damaged QR codes
- **Dependencies**: OpenCV with contrib modules + model files
- **Use case**: Challenging conditions, low quality images

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

### Preprocessing Modes

#### Disabled (Default)
```json
"preprocessing": {
    "enabled": false
}
```
No preprocessing - uses image directly from S4 Enhancement.

#### Minimal Mode
```json
"preprocessing": {
    "enabled": true,
    "mode": "minimal",
    "scaleFactor": 1.5
}
```
**Pipeline**: Scale only (fast)

- Scales image by `scaleFactor` (default 1.5x)
- Useful for small QR codes

#### Full Mode
```json
"preprocessing": {
    "enabled": true,
    "mode": "full",
    "scaleFactor": 1.5
}
```
**Pipeline**: Scale → Denoise → Binary → Morph → Invert

| Step | Operation | Purpose |
|------|-----------|---------|
| 1. Scale | Resize by factor | Enlarge small QR codes |
| 2. Denoise | Median blur (3x3) | Remove salt-and-pepper noise |
| 3. Binary | Adaptive threshold | Convert to black/white |
| 4. Morph | Morphological closing | Fill small holes |
| 5. Invert | Bitwise NOT | Handle white-on-black QR |

### Preprocessing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Boolean | `false` | Enable/disable preprocessing |
| `mode` | String | `"full"` | Mode: `"minimal"` or `"full"` |
| `scaleFactor` | Float | `1.5` | Scale factor for image resize |

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
| Standard QR accuracy | ✅ Excellent | ✅ Excellent |
| Damaged QR accuracy | ⚠️ Good | ✅ Better |
| Low light accuracy | ⚠️ Good | ✅ Better |
| Small QR accuracy | ⚠️ Good | ✅ Better |
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
            "scaleFactor": 1.5
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
            "scaleFactor": 1.5
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
    "text": "241023-V-BULK-123456-001-R0",
    "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "rect": [x, y, w, h],
    "confidence": 1.0,
    "backend": "zxing",
    "preprocessingEnabled": true,
    "preprocessingMode": "full",
    "parsed": {
        "dateCode": "241023",
        "facility": "V",
        "orderType": "BULK",
        "orderNumber": "123456",
        "position": "001",
        "revisionCount": "R0"
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
3. Try switching to WeChat backend
4. Enable debug to inspect preprocessed images

### Slow detection
- Switch from WeChat to ZXing backend
- Disable preprocessing or use `"mode": "minimal"`
- Reduce `scaleFactor` if using preprocessing

## Best Practices

1. **Development**: Use ZXing backend for faster iteration
2. **Production**: Choose based on image quality:
   - Good quality → ZXing (faster)
   - Poor quality → WeChat with preprocessing (more robust)
3. **Testing**: Enable debug to inspect preprocessing results
4. **Optimization**: Start with preprocessing disabled, enable only if needed

## Pipeline Integration

QR detection is Step 5 in the processing pipeline:

```
S4 Enhancement (Grayscale)
    ↓
S5 QR Detection
    ├── Preprocessing (optional)
    │   └── Scale → Denoise → Binary → Morph → Invert
    └── Detection (ZXing or WeChat)
    ↓
S6 Component Extraction
```

**Note**: S5 receives **grayscale** image from S4 Enhancement. The preprocessing pipeline is designed to work with grayscale input. Enhancement (CLAHE + Sharpen) is already applied in S4, so S5 preprocessing focuses on other techniques.

## Additional Notes

- Both backends use the same QR content parsing (regex-based)
- Detection results are consistent across backends
- Backend selection does not affect other pipeline steps
- Configuration changes take effect immediately on application restart
