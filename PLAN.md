# Kế hoạch triển khai OpenVINO Backend

**Ngày tạo:** 21/12/2025  
**Mục tiêu:** Hỗ trợ cả ONNX Runtime và OpenVINO Runtime cho YOLO detector theo nguyên tắc Open/Closed Principle (OCP)

---

## 📋 Phân tích kiến trúc hiện tại

### Cấu trúc tầng (Layers)

```
┌──────────────────────────────────────────────────────────┐
│  UI Layer (pipeline_orchestrator.py)                     │
│  - Khởi tạo ConfigService                                │
│  - Truyền tham số từ config vào services                 │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  Service Layer (services/impl/)                          │
│  - S1-S8: Pipeline services                              │
│  - Khởi tạo core components                              │
│  - Quản lý logging, debug, timing                        │
│  - KHÔNG phụ thuộc trực tiếp vào ConfigService           │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  Core Layer (core/)                                      │
│  - Triển khai logic chính (detector, preprocessor, etc.) │
│  - Implements interfaces từ core/interfaces/             │
│  - Không phụ thuộc vào service layer                     │
└──────────────────────────────────────────────────────────┘
```

### Luồng khởi tạo hiện tại (S2 Detection)

```
PipelineOrchestrator.__init__()
    │
    ├─> ConfigService.loadConfig("application_config.json")
    │       └─> Đọc s2_detection.modelPath, inputSize, etc.
    │
    └─> _initializeServices()
            └─> S2DetectionService(
                    modelPath=config.getModelPath(),        # ← Tham số
                    inputSize=config.getInputSize(),        # ← Tham số
                    isSegmentation=config.isSegmentation(), # ← Tham số
                    ...
                )
                    └─> YOLODetector(inputSize, classNames, isSegmentation)
                            └─> detector.loadModel(modelPath)
```

**Điểm mạnh:**
- ✅ DIP: Services nhận tham số, không phụ thuộc vào ConfigService
- ✅ SRP: Mỗi service có trách nhiệm rõ ràng
- ✅ Dễ test: Mock parameters thay vì mock ConfigService

**Vấn đề cần giải quyết:**
- ⚠️ S2DetectionService hard-code `YOLODetector` (ONNX Runtime)
- ⚠️ Không có cơ chế chọn backend (ONNX vs OpenVINO)

---

## 🎯 Thiết kế giải pháp (OCP Compliant)

### Nguyên tắc thiết kế

1. **Open/Closed Principle (OCP)**
   - Mở rộng: Thêm OpenVINO detector mà không sửa YOLODetector
   - Đóng: Không thay đổi IDetector interface và logic xử lý

2. **Dependency Inversion Principle (DIP)**
   - S2DetectionService phụ thuộc vào `IDetector` (abstraction)
   - Không phụ thuộc vào concrete implementation (YOLODetector, OpenVINODetector)

3. **Factory Pattern**
   - Sử dụng Factory để chọn backend dựa trên tham số
   - Factory ẩn chi tiết khởi tạo concrete detector

### Kiến trúc mới

```
┌─────────────────────────────────────────────────────────────┐
│  IDetector (Interface)                                      │
│  - loadModel(modelPath)                                     │
│  - detect(image, confidenceThreshold)                       │
│  - getClassNames()                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │ implements
      ┌────────────┴────────────┐
      │                         │
┌─────▼─────────────┐   ┌──────▼──────────────┐
│  YOLODetector     │   │ OpenVINODetector    │
│  (ONNX Runtime)   │   │ (OpenVINO Runtime)  │
└───────────────────┘   └─────────────────────┘
      ▲                         ▲
      │                         │
      └────────┬────────────────┘
               │ creates
       ┌───────▼────────┐
       │ DetectorFactory│
       │ - create(...)  │
       └────────────────┘
               ▲
               │ uses
       ┌───────┴────────┐
       │ S2Detection    │
       │ Service        │
       └────────────────┘
```

### Configuration schema

```json
{
  "s2_detection": {
    "_description": "Step 2: YOLO detection settings",
    "backend": "openvino",  // ← MỚI: "onnx" hoặc "openvino"
    "modelPath": "models/yolo11n-seg-version-1-0-0_int8_openvino_model/yolo11n-seg-version-1-0-0.xml",
    "isSegmentation": true,
    "inputSize": 640,
    ...
  }
}
```

---

## 📝 Danh sách file cần tạo/sửa

### ✨ Files mới (Total: 4 files)

#### 1. `core/detector/openvino_detector.py` (NEW)
**Mô tả:** Implementation của IDetector sử dụng OpenVINO Runtime  
**Nhiệm vụ:**
- Load model từ `.xml` file (OpenVINO IR format)
- Preprocessing/postprocessing giống hệt YOLODetector
- Hỗ trợ segmentation với proto masks

**Dependencies:**
```python
from openvino.runtime import Core
import numpy as np
import cv2
from core.interfaces.detector_interface import IDetector, Detection
```

**Methods chính:**
- `loadModel(modelPath: str) -> bool`
  - Sử dụng `Core().read_model(model=xmlPath)`
  - Compile model cho CPU device
  - Get input/output tensor names
  
- `detect(image: np.ndarray, confidenceThreshold: float) -> List[Detection]`
  - Reuse `_preprocess()` và `_postprocess()` từ YOLODetector
  - Inference: `compiled_model([input_tensor])`

**Ước lượng:** ~400 dòng (clone YOLODetector và thay runtime)

---

#### 2. `core/detector/detector_factory.py` (NEW)
**Mô tả:** Factory class để tạo detector dựa trên backend  
**Nhiệm vụ:**
- Kiểm tra backend parameter ("onnx" hoặc "openvino")
- Validate dependencies (onnxruntime/openvino installed?)
- Tạo instance phù hợp với error handling

**Code structure:**
```python
from typing import Optional, List
from core.interfaces.detector_interface import IDetector

def createDetector(
    backend: str = "onnx",
    modelPath: str = "",
    inputSize: int = 640,
    classNames: Optional[List[str]] = None,
    isSegmentation: bool = False
) -> IDetector:
    """
    Factory function to create detector based on backend.
    
    Args:
        backend: "onnx" or "openvino"
        modelPath: Path to model file (.onnx or .xml)
        inputSize: Model input size
        classNames: List of class names
        isSegmentation: Whether model supports segmentation
        
    Returns:
        IDetector: Detector instance
        
    Raises:
        ValueError: If backend is invalid
        ImportError: If required library not installed
    """
    backend = backend.lower()
    
    if backend == "openvino":
        try:
            from core.detector.openvino_detector import OpenVINODetector
            return OpenVINODetector(inputSize, classNames, isSegmentation)
        except ImportError:
            raise ImportError(
                "OpenVINO Runtime not installed. "
                "Install with: pip install openvino>=2024.0.0"
            )
    
    elif backend == "onnx":
        try:
            from core.detector.yolo_detector import YOLODetector
            return YOLODetector(inputSize, classNames, isSegmentation)
        except ImportError:
            raise ImportError(
                "ONNX Runtime not installed. "
                "Install with: pip install onnxruntime>=1.16.0"
            )
    
    else:
        raise ValueError(
            f"Invalid backend: '{backend}'. "
            f"Must be 'onnx' or 'openvino'."
        )
```

**Ước lượng:** ~80 dòng

---

#### 3. `core/detector/__init__.py` (UPDATE)
**Mô tả:** Export detector classes và factory  
**Nhiệm vụ:**
- Export `createDetector` để dễ import
- Export concrete classes cho direct usage nếu cần

**Code:**
```python
"""
Detector module for object detection and instance segmentation.

Provides:
- YOLODetector: ONNX Runtime implementation
- OpenVINODetector: OpenVINO Runtime implementation
- createDetector: Factory function to create detector based on backend
"""

from core.detector.detector_factory import createDetector

__all__ = [
    'createDetector',
]
```

**Ước lượng:** ~15 dòng

---

#### 4. `docs/OPENVINO_MIGRATION.md` (NEW - Optional)
**Mô tả:** Documentation cho việc migrate từ ONNX sang OpenVINO  
**Nhiệm vụ:**
- Hướng dẫn cài đặt OpenVINO
- So sánh performance ONNX vs OpenVINO
- Troubleshooting common issues

**Ước lượng:** ~150 dòng

---

### 🔧 Files cần sửa (Total: 4 files)

#### 1. `services/impl/config_service.py` (MODIFY)
**Thay đổi:** Thêm method để đọc backend từ config

**Location:** Line ~206 (trong S2 Detection Settings section)

**Thêm:**
```python
def getDetectionBackend(self) -> str:
    """
    Get detection backend (onnx or openvino).
    
    Returns:
        str: Backend name ("onnx" or "openvino"), default "onnx"
    """
    backend = self.get("s2_detection.backend", "onnx")
    return backend.lower()
```

**Ước lượng:** +10 dòng

---

#### 2. `services/impl/s2_detection_service.py` (MODIFY)
**Thay đổi:** Sử dụng DetectorFactory thay vì hard-code YOLODetector

**Location:** Line ~24-26 (imports) và Line ~112-114 (__init__)

**Before:**
```python
from core.detector.yolo_detector import YOLODetector

# In __init__:
self._detector: IDetector = YOLODetector(
    inputSize=inputSize,
    classNames=classNames or ["label"],
    isSegmentation=isSegmentation
)
```

**After:**
```python
from core.detector import createDetector

# In __init__:
self._detector: IDetector = createDetector(
    backend=backend,  # ← NEW parameter
    modelPath=modelPath,
    inputSize=inputSize,
    classNames=classNames or ["label"],
    isSegmentation=isSegmentation
)
```

**Constructor signature change:**
```python
def __init__(
    self,
    backend: str = "onnx",  # ← NEW parameter
    modelPath: str = "",
    inputSize: int = 640,
    ...
):
```

**Ước lượng:** ~5 dòng thay đổi, +1 parameter

---

#### 3. `ui/pipeline_orchestrator.py` (MODIFY)
**Thay đổi:** Truyền thêm parameter `backend` vào S2DetectionService

**Location:** Line ~104-113 (S2 Detection Service initialization)

**Before:**
```python
self._s2DetectionService = S2DetectionService(
    modelPath=self._configService.getModelPath(),
    inputSize=self._configService.getInputSize(),
    isSegmentation=self._configService.isSegmentation(),
    classNames=classNames,
    ...
)
```

**After:**
```python
self._s2DetectionService = S2DetectionService(
    backend=self._configService.getDetectionBackend(),  # ← NEW
    modelPath=self._configService.getModelPath(),
    inputSize=self._configService.getInputSize(),
    isSegmentation=self._configService.isSegmentation(),
    classNames=classNames,
    ...
)
```

**Ước lượng:** +1 dòng

---

#### 4. `config/application_config.json` (MODIFY)
**Thay đổi:** Thêm field `backend` vào s2_detection section

**Location:** Line ~33-35 (s2_detection section)

**Before:**
```json
{
  "s2_detection": {
    "_description": "Step 2: YOLO detection settings",
    "modelPath": "models/yolo11n-seg-version-1-0-0.onnx",
    "isSegmentation": true,
    ...
  }
}
```

**After:**
```json
{
  "s2_detection": {
    "_description": "Step 2: YOLO detection settings",
    "backend": "onnx",
    "modelPath": "models/yolo11n-seg-version-1-0-0.onnx",
    "_comment_backend": "Backend for inference: 'onnx' (ONNX Runtime) or 'openvino' (OpenVINO Runtime)",
    "_comment_modelPath_onnx": "For ONNX: models/yolo11n-seg-version-1-0-0.onnx",
    "_comment_modelPath_openvino": "For OpenVINO: models/yolo11n-seg-version-1-0-0_int8_openvino_model/yolo11n-seg-version-1-0-0.xml",
    "isSegmentation": true,
    ...
  }
}
```

**Ước lượng:** +5 dòng

---

#### 5. `requirements.txt` (MODIFY)
**Thay đổi:** Thêm OpenVINO Runtime (optional dependency)

**Location:** Line ~32-35 (ML Inference section)

**After:**
```txt
# ========== ML Inference ==========
# ONNX Runtime: High-performance inference engine
# - Hỗ trợ CPU/GPU acceleration
# - Nhẹ hơn PyTorch/TensorFlow
onnxruntime>=1.16.0

# OpenVINO Runtime: Intel-optimized inference engine (OPTIONAL)
# - Tối ưu cho CPU/GPU/VPU Intel
# - Hỗ trợ INT8 quantization cho tốc độ cao
# - Nhanh hơn 2-4x so với ONNX Runtime trên Intel hardware
# - Uncomment dòng dưới nếu muốn sử dụng OpenVINO backend
# openvino>=2024.0.0
```

**Ước lượng:** +8 dòng

---

## 🔄 Quy trình triển khai (Step-by-step)

### Phase 1: Core Layer Implementation

**Step 1.1:** Tạo `core/detector/openvino_detector.py`
- Clone YOLODetector structure
- Thay ONNX Runtime → OpenVINO Runtime
- Test độc lập với sample image

**Step 1.2:** Tạo `core/detector/detector_factory.py`
- Implement createDetector function
- Add validation và error handling
- Unit test với cả 2 backends

**Step 1.3:** Update `core/detector/__init__.py`
- Export createDetector
- Verify import paths

**Acceptance Criteria:**
```python
# Test script
from core.detector import createDetector

# Test ONNX backend
detector_onnx = createDetector(
    backend="onnx",
    modelPath="models/yolo11n-seg-version-1-0-0.onnx",
    inputSize=640,
    isSegmentation=True
)
assert detector_onnx.loadModel(modelPath)

# Test OpenVINO backend
detector_openvino = createDetector(
    backend="openvino",
    modelPath="models/yolo11n-seg-version-1-0-0_int8_openvino_model/yolo11n-seg-version-1-0-0.xml",
    inputSize=640,
    isSegmentation=True
)
assert detector_openvino.loadModel(modelPath)
```

---

### Phase 2: Service Layer Integration

**Step 2.1:** Update `services/impl/config_service.py`
- Thêm `getDetectionBackend()` method
- Test đọc config mới

**Step 2.2:** Update `services/impl/s2_detection_service.py`
- Thêm parameter `backend` vào constructor
- Thay `YOLODetector()` → `createDetector()`
- Update docstring

**Acceptance Criteria:**
```python
# Test service với ONNX backend
service_onnx = S2DetectionService(
    backend="onnx",
    modelPath="models/yolo11n-seg-version-1-0-0.onnx",
    inputSize=640,
    isSegmentation=True,
    ...
)

# Test service với OpenVINO backend
service_openvino = S2DetectionService(
    backend="openvino",
    modelPath="models/yolo11n-seg-version-1-0-0_int8_openvino_model/yolo11n-seg-version-1-0-0.xml",
    inputSize=640,
    isSegmentation=True,
    ...
)
```

---

### Phase 3: Orchestrator Integration

**Step 3.1:** Update `ui/pipeline_orchestrator.py`
- Truyền `backend` parameter từ config
- Verify initialization không break

**Step 3.2:** Update `config/application_config.json`
- Thêm field `backend` với default "onnx"
- Thêm comments hướng dẫn sử dụng

**Acceptance Criteria:**
- Application khởi động bình thường với `backend: "onnx"`
- Application khởi động bình thường với `backend: "openvino"`
- Switching backend bằng cách sửa config file

---

### Phase 4: Testing & Validation

**Step 4.1:** Functional Testing
- Test detection với ONNX backend (baseline)
- Test detection với OpenVINO backend
- So sánh output (bbox, masks, confidence)

**Step 4.2:** Performance Testing
- Benchmark inference time: ONNX vs OpenVINO
- Memory usage comparison
- Document kết quả trong CHANGELOG.md

**Step 4.3:** Integration Testing
- Run `scripts/detection.py` với cả 2 backends
- Verify debug output consistency
- Test với UI application (main.py)

**Step 4.4:** Error Handling Testing
- Test khi OpenVINO chưa cài đặt
- Test khi model path sai
- Test khi backend không hợp lệ

**Acceptance Criteria:**
- Tất cả tests pass
- OpenVINO inference time < ONNX inference time (trên Intel CPU)
- Output quality tương đương (±2% confidence)

---

### Phase 5: Documentation & Cleanup

**Step 5.1:** Update requirements.txt
- Thêm openvino dependency (commented)
- Hướng dẫn cài đặt

**Step 5.2:** Create OPENVINO_MIGRATION.md (optional)
- Installation guide
- Configuration guide
- Performance comparison
- Troubleshooting

**Step 5.3:** Update CHANGELOG.md
- Document new feature
- List breaking changes (nếu có)
- Performance improvements

---

## 📊 Effort Estimation

| Phase | Tasks | Estimated Time | Complexity |
|-------|-------|----------------|------------|
| **Phase 1** | Core Layer | 4-6 hours | Medium |
| - OpenVINODetector | | 3-4 hours | Medium |
| - DetectorFactory | | 1 hour | Low |
| - Testing | | 1 hour | Low |
| **Phase 2** | Service Layer | 2-3 hours | Low |
| - ConfigService update | | 30 min | Low |
| - S2DetectionService update | | 1 hour | Low |
| - Testing | | 1 hour | Low |
| **Phase 3** | Orchestrator | 1-2 hours | Low |
| - PipelineOrchestrator update | | 30 min | Low |
| - Config update | | 30 min | Low |
| - Testing | | 1 hour | Low |
| **Phase 4** | Testing & Validation | 3-4 hours | Medium |
| - Functional testing | | 1 hour | Medium |
| - Performance testing | | 1 hour | Medium |
| - Integration testing | | 1-2 hours | Medium |
| **Phase 5** | Documentation | 1-2 hours | Low |
| **Total** | | **11-17 hours** | **Medium** |

---

## ⚠️ Risks & Mitigation

### Risk 1: OpenVINO API khác biệt với ONNX Runtime
**Impact:** High  
**Probability:** Medium  
**Mitigation:**
- Reuse preprocessing/postprocessing logic từ YOLODetector
- Chỉ thay đổi phần load model và inference
- Extensive testing với sample images

### Risk 2: OpenVINO model format không tương thích
**Impact:** High  
**Probability:** Low  
**Mitigation:**
- Verify `.xml` file tồn tại và đúng format
- Test load model trước khi integrate
- Fallback về ONNX nếu OpenVINO fail

### Risk 3: Performance không cải thiện như mong đợi
**Impact:** Medium  
**Probability:** Medium  
**Mitigation:**
- Benchmark trên Intel hardware (not AMD/ARM)
- So sánh INT8 OpenVINO với FP32 ONNX (fair comparison)
- Document kết quả để user biết trước

### Risk 4: Breaking changes ảnh hưởng existing code
**Impact:** High  
**Probability:** Low  
**Mitigation:**
- Default backend = "onnx" (backward compatible)
- Không thay đổi IDetector interface
- Extensive regression testing

---

## ✅ Success Criteria

### Functional Requirements
- ✅ Hỗ trợ cả ONNX và OpenVINO backend
- ✅ Switch backend bằng config file
- ✅ Output consistency giữa 2 backends
- ✅ Backward compatible (default ONNX)

### Non-Functional Requirements
- ✅ OpenVINO inference nhanh hơn ONNX (trên Intel CPU)
- ✅ Code tuân thủ OCP (mở rộng, không sửa)
- ✅ Không phụ thuộc hard vào ConfigService
- ✅ Error handling graceful khi library thiếu

### Code Quality
- ✅ Type hints đầy đủ
- ✅ Docstrings chi tiết
- ✅ Logging phù hợp
- ✅ Tuân thủ coding guidelines (SOLID, naming conventions)

---

## 🚀 Deployment Strategy

### Development Environment
```bash
# Install OpenVINO
pip install openvino>=2024.0.0

# Verify installation
python -c "from openvino.runtime import Core; print('OpenVINO OK')"
```

### Configuration
```json
// config/application_config.json
{
  "s2_detection": {
    "backend": "openvino",
    "modelPath": "models/yolo11n-seg-version-1-0-0_int8_openvino_model/yolo11n-seg-version-1-0-0.xml",
    ...
  }
}
```

### Rollback Plan
Nếu OpenVINO gặp vấn đề:
1. Đổi `backend: "openvino"` → `backend: "onnx"`
2. Đổi `modelPath` về `.onnx` file
3. Restart application

---

## 📚 References

### OpenVINO Documentation
- [OpenVINO Runtime API](https://docs.openvino.ai/latest/api/api_reference.html)
- [Model Inference](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Integrate_OV_with_your_application.html)
- [Supported Devices](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)

### YOLO + OpenVINO Examples
- [Ultralytics YOLOv8 OpenVINO](https://docs.ultralytics.com/integrations/openvino/)
- [OpenVINO YOLOv5 Demo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/object_detection_demo/python)

---

## 💡 Future Enhancements

### Phase 2 (Post-MVP)
- [ ] Support GPU backend cho OpenVINO (Intel GPU, integrated graphics)
- [ ] Auto-select backend dựa trên hardware detection
- [ ] Performance monitoring dashboard
- [ ] Model benchmarking tool

### Phase 3 (Advanced)
- [ ] Support TensorRT backend (NVIDIA GPU)
- [ ] Dynamic backend switching (runtime)
- [ ] Multi-model ensemble (ONNX + OpenVINO parallel)
- [ ] Cloud inference với OpenVINO Model Server

---

## 📝 Notes

- Tài liệu này tuân thủ quy tắc thiết kế trong `.github/copilot-instructions.md`
- Tất cả thay đổi đều additive (không breaking changes)
- Backward compatible với codebase hiện tại
- Focus vào maintainability và extensibility

---

**Chờ xác nhận từ user trước khi bắt đầu implementation.**
