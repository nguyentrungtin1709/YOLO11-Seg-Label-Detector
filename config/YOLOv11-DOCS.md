# YOLOv11 Backend Configuration Guide

## Overview

This document explains how to configure the YOLO11 detector to switch between two inference backends: ONNX Runtime and OpenVINO Runtime.

## Backend Options

### ONNX Runtime (Default)
- **Cross-platform**: Works on Windows, Linux, macOS
- **Model format**: `.onnx` file (single file)
- **Precision**: FP32 (32-bit floating point)
- **Model size**: ~6.5 MB
- **Performance**: Standard inference speed
- **Use case**: General purpose, development, testing

### OpenVINO Runtime
- **Platform**: Optimized for Intel CPUs (Windows, Linux)
- **Model format**: `.xml` + `.bin` files (IR format)
- **Precision**: INT8 (8-bit integer quantization)
- **Model size**: ~1.6 MB
- **Performance**: 2-4x faster on Intel CPUs
- **Use case**: Production deployment on Intel hardware

## Configuration File

Edit the file: `config/application_config.json`

### Switching to ONNX Backend

```json
{
    "s2_detection": {
        "backend": "onnx",
        "modelPath": "models/yolo11n-seg-version-1-0-0.onnx",
        "isSegmentation": true,
        "inputSize": 640,
        "confidenceThreshold": 0.5
    }
}
```

### Switching to OpenVINO Backend

```json
{
    "s2_detection": {
        "backend": "openvino",
        "modelPath": "models/yolo11n-seg-version-1-0-0_int8_openvino_model/yolo11n-seg-version-1-0-0.xml",
        "isSegmentation": true,
        "inputSize": 640,
        "confidenceThreshold": 0.5
    }
}
```

## Configuration Parameters

### backend
- **Type**: String
- **Values**: `"onnx"` or `"openvino"`
- **Required**: Yes
- **Description**: Selects the inference backend

### modelPath
- **Type**: String
- **Required**: Yes
- **Description**: Path to the model file
  - For ONNX: Path to `.onnx` file
  - For OpenVINO: Path to `.xml` file (`.bin` must be in same directory)

### Other Parameters
- `isSegmentation`: Enable instance segmentation (default: true)
- `inputSize`: Input image size for model (default: 640)
- `confidenceThreshold`: Minimum confidence score (default: 0.5)
- `maxAreaRatio`: Maximum detection area ratio (default: 0.40)
- `topNDetections`: Maximum number of detections (default: 2)

## Requirements

### ONNX Runtime
Already included in `requirements.txt`:
```
onnxruntime>=1.16.0
```

### OpenVINO Runtime
Install OpenVINO for production use:
```bash
pip install openvino>=2024.0.0
```

Or enable in `requirements.txt` by uncommenting:
```
# OpenVINO Runtime (optional, Intel-optimized inference)
openvino>=2024.0.0
```

## Model Files

### ONNX Model
Located at: `models/yolo11n-seg-version-1-0-0.onnx`
- Single file format
- Ready to use with ONNX Runtime

### OpenVINO Model
Located at: `models/yolo11n-seg-version-1-0-0_int8_openvino_model/`
- Two files: `.xml` (architecture) + `.bin` (weights)
- Optimized for Intel CPUs with INT8 quantization

## Performance Comparison

Based on testing with 7 sample images on Intel CPU:

| Metric | ONNX Runtime | OpenVINO Runtime | Improvement |
|--------|-------------|------------------|-------------|
| Average detection time | 188.96 ms | 116.05 ms | 38.6% faster |
| Batch total time | 4545.57 ms | 4287.30 ms | 5.7% faster |
| Average FPS | 1.57 | 1.69 | +7.6% |
| Model size | 6.5 MB | 1.6 MB | 75% smaller |

## Troubleshooting

### Backend not found error
If you see an error about backend not available:
- Check that the backend name is spelled correctly: `"onnx"` or `"openvino"`
- For OpenVINO: Ensure it is installed with `pip install openvino>=2024.0.0`

### Model loading error
- Verify the model file exists at the specified path
- For ONNX: Check that `.onnx` file is present
- For OpenVINO: Check that both `.xml` and `.bin` files are present

### Performance issues
- ONNX: Standard performance, no optimization needed
- OpenVINO: Ensure you're running on Intel CPU for best results
- OpenVINO: First inference may be slower due to model compilation

## Best Practices

1. **Development**: Use ONNX backend for easier debugging and cross-platform compatibility
2. **Production**: Use OpenVINO backend for faster inference on Intel hardware
3. **Testing**: Switch backends to compare accuracy and performance
4. **Deployment**: Choose backend based on target hardware and performance requirements

## Additional Notes

- Both backends use the same preprocessing and postprocessing pipeline
- Detection results should be consistent across backends
- Backend selection does not affect other pipeline steps (S3-S8)
- Configuration changes take effect immediately on application restart
