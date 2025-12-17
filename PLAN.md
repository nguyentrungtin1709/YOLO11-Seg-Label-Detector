# PLAN.md - Kế hoạch hoàn thiện Label OCR Pipeline

## 1. Tổng quan quy trình

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           LABEL OCR EXTRACTION PIPELINE                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
    ╔════════════════════════════════════╩════════════════════════════════════╗
    ║                    ĐÃ HOÀN THÀNH (Bước 1-4)                            ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Bước 1:       │───▶│   Bước 2:       │───▶│   Bước 3:       │───▶│   Bước 4:       │
│  Camera/OpenCV  │    │ YOLO11-Seg      │    │ Crop/Rotate/    │    │ CLAHE +         │
│  (Lấy frame)    │    │ Detection       │    │ Orientation Fix │    │ Unsharp Mask    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
    ╔═════════════════════════════════════════════════════════════════════════╩═════════╗
    ║                         CẦN TRIỂN KHAI (Bước 5-8)                                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Bước 5:       │───▶│   Bước 6:       │───▶│   Bước 7:       │───▶│   Bước 8:       │
│  QR Detection   │    │ Component       │    │ OCR Text        │    │ Post-Process    │
│  (pyzbar)       │    │ Extraction      │    │ Extraction      │    │ (Fuzzy Match)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
        ▼                      ▼                      ▼                      ▼
   debug/qr-code/       debug/components/     debug/ocr-raw-text/   debug/result/
   (JSON)               (Images)              (JSON)                (JSON)
```

### 1.1 Cấu trúc nhãn mẫu

Dựa trên ảnh nhãn mẫu được cung cấp:

```
┌─────────────────────────────────────────────────────────────────┐
│  VA-S-002410-1                              11/19              │  ← Header (Order Info)
│                                                                 │
│  PTFY-API                           ┌───┐   1/1                │  ← Product Code + Position/Quantity
│                                     │ B │                      │
│                                     │QR │                      │  ← QR Code
│                                     │   │                      │
│                                     └───┘                      │
│  340                                                           │  ← Product Code (Garment)
│  3T                                                            │  ← Size
│  MIDNIGHT                                                      │  ← Color
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 QR Code Format

```
Format: MMDDYY-FACILITY-TYPE-ORDER-POSITION
Ví dụ:  110125-VA-M-000002-2

Trong đó:
- MMDDYY: Ngày (110125 = 11/01/2025)
- FACILITY: Mã cơ sở (VA, GA, ...)
- TYPE: Loại (M, S, ...)
- ORDER: Mã đơn hàng (000002)
- POSITION: Vị trí = position trong "position/quantity" (2 = 2/x)
```

### 1.3 Thông tin cần trích xuất

| Field | Nguồn | Ví dụ |
|-------|-------|-------|
| `fullOrderCode` | QR Code | `110125-VA-M-000002-2` |
| `positionQuantity` | OCR (trên QR) | `1/1` |
| `productCode` | OCR (dưới QR) | `340` |
| `size` | OCR (dưới QR) | `3T` |
| `color` | OCR (dưới QR) | `MIDNIGHT` |
| `position` | Parsed từ QR | `2` |
| `isValid` | Validate position | `true/false` |

---

## 2. Phân tích thư viện

### 2.1 QR Detection - pyzbar

**Thư viện**: `pyzbar` (pip install pyzbar)

**Yêu cầu hệ thống**:
- Linux: `sudo apt-get install libzbar0`
- macOS: `brew install zbar`
- Windows: DLL được bao gồm trong wheel

**Kết quả trả về** (từ `pyzbar.pyzbar.decode()`):

```python
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol

result = decode(image, symbols=[ZBarSymbol.QRCODE])
# Trả về List[Decoded]

# Mỗi Decoded object có:
Decoded(
    data=b'110125-VA-M-000002-2',        # bytes - Nội dung QR
    type='QRCODE',                        # str - Loại barcode
    rect=Rect(left=27, top=27, width=145, height=145),  # Bounding box
    polygon=[                             # 4 điểm góc của QR
        Point(x=27, y=27),
        Point(x=27, y=172),
        Point(x=172, y=172),
        Point(x=172, y=27)
    ],
    orientation='UP',                     # Hướng QR (UP/DOWN/LEFT/RIGHT)
    quality=1                             # Chất lượng nhận diện
)
```

**Lưu ý quan trọng**:
- `data` trả về bytes → cần decode sang string: `data.decode('utf-8')`
- `polygon` là List[Point] với 4 góc → sử dụng để tính toán vùng cắt component

### 2.2 OCR - PaddleOCR

**Thư viện**: `paddleocr` (pip install paddleocr)

**Cấu hình CPU only** (không GPU):

```python
from paddleocr import PaddleOCR

# Khởi tạo với CPU (chạy 1 lần)
ocr = PaddleOCR(
    use_angle_cls=True,     # Phân loại góc xoay
    lang='en',              # Ngôn ngữ
    use_gpu=False,          # ❌ KHÔNG dùng GPU
    show_log=False          # Tắt log
)

# OCR
result = ocr.ocr(image, cls=True)

# Kết quả:
# result[0] là list các text block, mỗi block có format:
# [
#   [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 4 điểm góc bounding box
#   ('TEXT_CONTENT', 0.95)                  # (text, confidence)
# ]
```

**Ví dụ kết quả từ nhãn mẫu**:
```python
[
    [[[100, 10], [150, 10], [150, 40], [100, 40]], ('1/1', 0.98)],      # Position/Quantity
    [[[50, 200], [100, 200], [100, 230], [50, 230]], ('340', 0.95)],    # Product Code
    [[[50, 240], [80, 240], [80, 270], [50, 270]], ('3T', 0.92)],       # Size
    [[[50, 280], [150, 280], [150, 310], [50, 310]], ('MIDNIGHT', 0.88)] # Color
]
```

---

## 3. Kiến trúc thiết kế (SOLID Principles)

### 3.1 Cấu trúc thư mục mới

```
core/
├── interfaces/
│   ├── ...
│   ├── qr_detector_interface.py      # IQrDetector
│   ├── component_extractor_interface.py  # IComponentExtractor
│   ├── ocr_extractor_interface.py    # IOcrExtractor
│   └── text_processor_interface.py   # ITextProcessor
├── qr/
│   ├── __init__.py
│   └── pyzbar_qr_detector.py        # PyzbarQrDetector
├── extractor/
│   ├── __init__.py
│   └── label_component_extractor.py # LabelComponentExtractor
├── ocr/
│   ├── __init__.py
│   └── paddle_ocr_extractor.py      # PaddleOcrExtractor
└── processor/
    ├── __init__.py
    ├── fuzzy_matcher.py              # FuzzyMatcher utility
    └── label_text_processor.py       # LabelTextProcessor

services/
├── ...
└── ocr_pipeline_service.py           # OcrPipelineService (Orchestrator)

data/                                  # ← Copy từ projects/ocr-labels-project/demo_ocr_label/data
├── colors.json                        # 4904 dòng - danh sách màu
├── products.json                      # 2172 dòng - danh sách sản phẩm
└── sizes.json                         # 138 dòng - danh sách size

output/debug/                          # ← Thư mục debug mới
├── qr-code/                           # Bước 5: Kết quả QR detection (JSON)
├── components/                        # Bước 6: Ảnh các vùng đã cắt
├── ocr-raw-text/                      # Bước 7: Kết quả OCR thô (JSON)
└── result/                            # Bước 8: Kết quả cuối cùng (JSON)

config/
└── app_config.json                   # Thêm config cho OCR pipeline
```

### 3.2 Interfaces (core/interfaces/)

#### 3.2.1 IQrDetector

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class QrDetectionResult:
    """Kết quả phát hiện QR code."""
    text: str                              # Nội dung QR (VD: "110125-VA-M-000002-2")
    polygon: List[Tuple[int, int]]         # 4 góc [(x,y), ...]
    rect: Tuple[int, int, int, int]        # (left, top, width, height)
    confidence: float                      # Độ tin cậy
    
    # Parsed fields từ QR code
    dateCode: str = ""                     # MMDDYY (110125)
    facility: str = ""                     # VA, GA, ...
    orderType: str = ""                    # M, S, ...
    orderNumber: str = ""                  # 000002
    position: int = 0                      # 2 (position trong position/quantity)

class IQrDetector(ABC):
    """Interface cho QR code detector."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[QrDetectionResult]:
        """
        Phát hiện QR code trong ảnh.
        
        Args:
            image: Ảnh đầu vào (BGR hoặc grayscale)
            
        Returns:
            QrDetectionResult nếu tìm thấy, None nếu không
        """
        pass
```

#### 3.2.2 IComponentExtractor

```python
@dataclass
class ComponentResult:
    """Kết quả trích xuất các vùng component."""
    mergedImage: np.ndarray                # Ảnh đã merge các vùng
    aboveQrRoi: np.ndarray                 # Vùng phía trên QR (position/quantity)
    belowQrRoi: np.ndarray                 # Vùng phía dưới QR (product, size, color)
    qrPolygon: List[Tuple[int, int]]       # Vị trí QR trong ảnh gốc

class IComponentExtractor(ABC):
    """Interface cho component extractor."""
    
    @abstractmethod
    def extractAndMerge(
        self, 
        image: np.ndarray, 
        qrPolygon: List[Tuple[int, int]]
    ) -> Optional[ComponentResult]:
        """
        Trích xuất và merge các vùng quan tâm.
        
        Args:
            image: Ảnh nhãn đã xử lý
            qrPolygon: 4 góc của QR code
            
        Returns:
            ComponentResult nếu thành công, None nếu thất bại
        """
        pass
```

#### 3.2.3 IOcrExtractor

```python
@dataclass
class TextBlock:
    """Một block text từ OCR."""
    text: str                              # Nội dung text
    confidence: float                      # Độ tin cậy (0-1)
    bbox: List[List[int]]                  # 4 góc bounding box
    
@dataclass
class OcrResult:
    """Kết quả OCR."""
    textBlocks: List[TextBlock]            # Danh sách text blocks
    rawResult: any = None                  # Kết quả gốc từ OCR engine

class IOcrExtractor(ABC):
    """Interface cho OCR extractor."""
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> OcrResult:
        """
        Trích xuất text từ ảnh.
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            OcrResult với danh sách text blocks
        """
        pass
```

#### 3.2.4 ITextProcessor

```python
@dataclass
class LabelData:
    """Dữ liệu nhãn đã được cấu trúc hóa."""
    # Thông tin từ QR Code
    fullOrderCode: str = ""                # "110125-VA-M-000002-2"
    dateCode: str = ""                     # "110125" (MMDDYY)
    facility: str = ""                     # "VA"
    orderType: str = ""                    # "M"
    orderNumber: str = ""                  # "000002"
    qrPosition: int = 0                    # 2 (từ QR)
    
    # Thông tin từ OCR
    positionQuantity: str = ""             # "1/1" (từ vùng trên QR)
    ocrPosition: int = 0                   # 1 (parsed từ positionQuantity)
    quantity: int = 0                      # 1 (parsed từ positionQuantity)
    productCode: str = ""                  # "340"
    size: str = ""                         # "3T"
    color: str = ""                        # "MIDNIGHT"
    
    # Validation
    isValid: bool = False                  # position từ QR == position từ OCR
    fieldConfidences: dict = field(default_factory=dict)  # Độ tin cậy từng field

class ITextProcessor(ABC):
    """Interface cho text post-processor."""
    
    @abstractmethod
    def process(
        self, 
        textBlocks: List[TextBlock], 
        qrResult: QrDetectionResult
    ) -> LabelData:
        """
        Xử lý hậu kỳ kết quả OCR.
        
        Args:
            textBlocks: Danh sách text blocks từ OCR
            qrResult: Kết quả QR detection để validate
            
        Returns:
            LabelData đã được cấu trúc hóa
        """
        pass
```

### 3.3 Implementations

#### 3.3.1 PyzbarQrDetector (core/qr/pyzbar_qr_detector.py)

```python
import re
from pyzbar.pyzbar import decode, ZBarSymbol
from core.interfaces.qr_detector_interface import IQrDetector, QrDetectionResult

class PyzbarQrDetector(IQrDetector):
    """QR detector sử dụng pyzbar."""
    
    # Pattern: MMDDYY-FACILITY-TYPE-ORDER-POSITION
    QR_PATTERN = re.compile(r'^(\d{6})-([A-Z]{2})-([A-Z])-(\d+)-(\d+)$')
    
    def __init__(self, symbolTypes: List[ZBarSymbol] = None):
        self._symbolTypes = symbolTypes or [ZBarSymbol.QRCODE]
    
    def detect(self, image: np.ndarray) -> Optional[QrDetectionResult]:
        try:
            results = decode(image, symbols=self._symbolTypes)
            if not results:
                return None
            
            # Lấy QR code đầu tiên
            qr = results[0]
            text = qr.data.decode('utf-8')
            
            result = QrDetectionResult(
                text=text,
                polygon=[(p.x, p.y) for p in qr.polygon],
                rect=(qr.rect.left, qr.rect.top, qr.rect.width, qr.rect.height),
                confidence=qr.quality / 100.0 if qr.quality else 1.0
            )
            
            # Parse QR code nếu match pattern
            match = self.QR_PATTERN.match(text)
            if match:
                result.dateCode = match.group(1)      # 110125
                result.facility = match.group(2)     # VA
                result.orderType = match.group(3)    # M
                result.orderNumber = match.group(4)  # 000002
                result.position = int(match.group(5)) # 2
            
            return result
        except Exception:
            return None
```

#### 3.3.2 LabelComponentExtractor (core/extractor/label_component_extractor.py)

**Logic cắt vùng component** (dựa trên cấu trúc nhãn thực tế):

```
┌─────────────────────────────────────────────────────────────────┐
│  VA-S-002410-1                              11/19              │
│  PTFY-API                           ┌───┐   1/1                │ ← ABOVE QR (position/quantity)
│                                     │QR │                      │
│                                     └───┘                      │
│  340                                                           │ ← BELOW QR (product code)
│  3T                                                            │ ← BELOW QR (size)
│  MIDNIGHT                                                      │ ← BELOW QR (color)
└─────────────────────────────────────────────────────────────────┘
```

**Vùng cắt:**
- **aboveQrRoi**: Vùng nhỏ phía trên-phải QR chứa position/quantity (VD: "1/1")
- **belowQrRoi**: Vùng lớn phía dưới QR chứa product code, size, color

```python
class LabelComponentExtractor(IComponentExtractor):
    """Trích xuất và merge các vùng component từ nhãn."""
    
    def __init__(
        self,
        aboveQrWidthRatio: float = 0.3,      # Vùng trên: 30% chiều rộng ảnh
        aboveQrHeightRatio: float = 0.15,    # Vùng trên: 15% chiều cao ảnh
        belowQrWidthRatio: float = 0.6,      # Vùng dưới: 60% chiều rộng ảnh
        belowQrHeightRatio: float = 0.4      # Vùng dưới: 40% chiều cao ảnh
    ):
        self._aboveQrWidthRatio = aboveQrWidthRatio
        self._aboveQrHeightRatio = aboveQrHeightRatio
        self._belowQrWidthRatio = belowQrWidthRatio
        self._belowQrHeightRatio = belowQrHeightRatio
    
    def extractAndMerge(
        self, 
        image: np.ndarray, 
        qrPolygon: List[Tuple[int, int]]
    ) -> Optional[ComponentResult]:
        h, w = image.shape[:2]
        
        # Tính toán vị trí QR
        qrCenterX = sum(p[0] for p in qrPolygon) // 4
        qrCenterY = sum(p[1] for p in qrPolygon) // 4
        qrTop = min(p[1] for p in qrPolygon)
        qrBottom = max(p[1] for p in qrPolygon)
        
        # 1. Cắt vùng phía trên-phải QR (position/quantity: "1/1")
        aboveQrRoi = self._extractAboveQr(image, qrTop, qrCenterX, w, h)
        
        # 2. Cắt vùng phía dưới QR (product, size, color)
        belowQrRoi = self._extractBelowQr(image, qrBottom, w, h)
        
        # 3. Merge 2 vùng thành 1 ảnh (above ở trên, below ở dưới)
        mergedImage = self._mergeComponents(aboveQrRoi, belowQrRoi)
        
        return ComponentResult(
            mergedImage=mergedImage,
            aboveQrRoi=aboveQrRoi,
            belowQrRoi=belowQrRoi,
            qrPolygon=qrPolygon
        )
    
    def _extractAboveQr(self, image, qrTop, qrCenterX, imgWidth, imgHeight):
        """Cắt vùng phía trên-phải QR."""
        roiHeight = int(imgHeight * self._aboveQrHeightRatio)
        roiWidth = int(imgWidth * self._aboveQrWidthRatio)
        
        # Vùng ngay phía trên QR, lệch về bên phải
        y1 = max(0, qrTop - roiHeight)
        y2 = qrTop
        x1 = qrCenterX
        x2 = min(imgWidth, x1 + roiWidth)
        
        return image[y1:y2, x1:x2]
    
    def _extractBelowQr(self, image, qrBottom, imgWidth, imgHeight):
        """Cắt vùng phía dưới QR."""
        roiHeight = int(imgHeight * self._belowQrHeightRatio)
        roiWidth = int(imgWidth * self._belowQrWidthRatio)
        
        # Vùng bên dưới QR, từ góc trái
        y1 = qrBottom
        y2 = min(imgHeight, qrBottom + roiHeight)
        x1 = 0
        x2 = roiWidth
        
        return image[y1:y2, x1:x2]
```

#### 3.3.3 PaddleOcrExtractor (core/ocr/paddle_ocr_extractor.py)

```python
from paddleocr import PaddleOCR
from core.interfaces.ocr_extractor_interface import IOcrExtractor, OcrResult, TextBlock

class PaddleOcrExtractor(IOcrExtractor):
    """OCR extractor sử dụng PaddleOCR."""
    
    def __init__(
        self,
        lang: str = 'en',
        useAngleCls: bool = True,
        useGpu: bool = False,
        detDbThresh: float = 0.3,
        detDbBoxThresh: float = 0.5,
        clsThresh: float = 0.9
    ):
        self._ocr = PaddleOCR(
            use_angle_cls=useAngleCls,
            lang=lang,
            use_gpu=useGpu,
            det_db_thresh=detDbThresh,
            det_db_box_thresh=detDbBoxThresh,
            cls_thresh=clsThresh,
            show_log=False
        )
    
    def extract(self, image: np.ndarray) -> OcrResult:
        result = self._ocr.ocr(image, cls=True)
        
        textBlocks = []
        if result and result[0]:
            for item in result[0]:
                bbox, (text, confidence) = item
                textBlocks.append(TextBlock(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                ))
        
        return OcrResult(textBlocks=textBlocks, rawResult=result)
```

#### 3.3.4 LabelTextProcessor (core/processor/label_text_processor.py)

```python
import re
from core.interfaces.text_processor_interface import ITextProcessor, LabelData, TextBlock, QrDetectionResult
from core.processor.fuzzy_matcher import FuzzyMatcher

class LabelTextProcessor(ITextProcessor):
    """Xử lý hậu kỳ text từ OCR với fuzzy matching."""
    
    def __init__(
        self,
        validProducts: List[str] = None,
        validSizes: List[str] = None,
        validColors: List[str] = None,
        minFuzzyScore: float = 0.75
    ):
        self._validProducts = validProducts or []
        self._validSizes = validSizes or []
        self._validColors = validColors or []
        self._minFuzzyScore = minFuzzyScore
        self._positionPattern = re.compile(r'^(\d+)\s*/\s*(\d+)$')  # "1/1", "2/10"
    
    def process(
        self, 
        textBlocks: List[TextBlock], 
        qrResult: QrDetectionResult
    ) -> LabelData:
        result = LabelData()
        result.fieldConfidences = {}
        
        # Copy thông tin từ QR
        result.fullOrderCode = qrResult.text
        result.dateCode = qrResult.dateCode
        result.facility = qrResult.facility
        result.orderType = qrResult.orderType
        result.orderNumber = qrResult.orderNumber
        result.qrPosition = qrResult.position
        
        # Sắp xếp text blocks theo Y (từ trên xuống)
        sortedBlocks = sorted(textBlocks, key=lambda b: b.bbox[0][1])
        
        for block in sortedBlocks:
            text = block.text.strip().upper()
            confidence = block.confidence
            
            # Thử parse position/quantity (VD: "1/1")
            posMatch = self._positionPattern.match(text)
            if posMatch and not result.positionQuantity:
                result.positionQuantity = text
                result.ocrPosition = int(posMatch.group(1))
                result.quantity = int(posMatch.group(2))
                result.fieldConfidences['positionQuantity'] = confidence
                continue
            
            # Thử match Product Code
            if not result.productCode:
                matched, score = FuzzyMatcher.bestMatch(text, self._validProducts)
                if score >= self._minFuzzyScore:
                    result.productCode = matched
                    result.fieldConfidences['productCode'] = confidence * score
                    continue
                elif text.isdigit() or (len(text) <= 6 and text.isalnum()):
                    # Giữ nguyên nếu là mã ngắn
                    result.productCode = text
                    result.fieldConfidences['productCode'] = confidence
                    continue
            
            # Thử match Size
            if not result.size:
                matched, score = FuzzyMatcher.bestMatch(text, self._validSizes)
                if score >= self._minFuzzyScore:
                    result.size = matched
                    result.fieldConfidences['size'] = confidence * score
                    continue
            
            # Thử match Color
            if not result.color:
                matched, score = FuzzyMatcher.bestMatch(text, self._validColors)
                if score >= self._minFuzzyScore:
                    result.color = matched
                    result.fieldConfidences['color'] = confidence * score
                    continue
        
        # Validate: qrPosition == ocrPosition
        result.isValid = (result.qrPosition == result.ocrPosition) and result.ocrPosition > 0
        
        return result
```

#### 3.3.5 FuzzyMatcher (core/processor/fuzzy_matcher.py)

```python
from typing import List, Tuple

class FuzzyMatcher:
    """Utility class cho fuzzy string matching."""
    
    @staticmethod
    def levenshteinDistance(a: str, b: str) -> int:
        """Tính Levenshtein distance giữa 2 chuỗi."""
        if not a: return len(b)
        if not b: return len(a)
        
        prev = list(range(len(b) + 1))
        curr = [0] * (len(b) + 1)
        
        for i in range(1, len(a) + 1):
            curr[0] = i
            for j in range(1, len(b) + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
            prev, curr = curr, prev
        
        return prev[len(b)]
    
    @staticmethod
    def levenshteinSimilarity(a: str, b: str) -> float:
        """Tính độ tương đồng Levenshtein (0-1)."""
        if not a and not b: return 1.0
        dist = FuzzyMatcher.levenshteinDistance(a, b)
        maxLen = max(len(a or ''), len(b or ''))
        return 1.0 - dist / maxLen if maxLen > 0 else 1.0
    
    @staticmethod
    def jaroWinklerSimilarity(s: str, t: str, prefixScale: float = 0.1) -> float:
        """Tính độ tương đồng Jaro-Winkler (0-1)."""
        if not s and not t: return 1.0
        if not s or not t: return 0.0
        
        matchWindow = max(len(s), len(t)) // 2 - 1
        if matchWindow < 0:
            matchWindow = 0
        
        sMatches = [False] * len(s)
        tMatches = [False] * len(t)
        matches = 0
        transpositions = 0
        
        for i in range(len(s)):
            start = max(0, i - matchWindow)
            end = min(i + matchWindow + 1, len(t))
            for j in range(start, end):
                if tMatches[j] or s[i] != t[j]:
                    continue
                sMatches[i] = True
                tMatches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        k = 0
        for i in range(len(s)):
            if not sMatches[i]:
                continue
            while not tMatches[k]:
                k += 1
            if s[i] != t[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches/len(s) + matches/len(t) + (matches - transpositions/2)/matches) / 3
        
        # Prefix bonus
        prefix = 0
        for i in range(min(4, len(s), len(t))):
            if s[i] == t[i]:
                prefix += 1
            else:
                break
        
        return jaro + prefix * prefixScale * (1 - jaro)
    
    @staticmethod
    def combinedSimilarity(a: str, b: str) -> float:
        """Kết hợp Levenshtein và Jaro-Winkler (lấy max)."""
        na = (a or '').strip().upper()
        nb = (b or '').strip().upper()
        lev = FuzzyMatcher.levenshteinSimilarity(na, nb)
        jw = FuzzyMatcher.jaroWinklerSimilarity(na, nb)
        return max(lev, jw)
    
    @staticmethod
    def bestMatch(text: str, candidates: List[str], minScore: float = 0.0) -> Tuple[str, float]:
        """
        Tìm candidate match tốt nhất.
        
        Returns:
            Tuple (matched_text, score). Trả về ("", 0.0) nếu không tìm thấy.
        """
        if not text or not candidates:
            return ("", 0.0)
        
        best = ""
        bestScore = minScore
        
        for c in candidates:
            score = FuzzyMatcher.combinedSimilarity(text, c)
            if score > bestScore:
                bestScore = score
                best = c
        
        return (best, bestScore)
```

### 3.4 OcrPipelineService (services/ocr_pipeline_service.py)

```python
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import cv2
import numpy as np

@dataclass
class OcrPipelineResult:
    """Kết quả từ OCR pipeline."""
    labelData: LabelData                   # Dữ liệu nhãn đã xử lý
    qrResult: Optional[QrDetectionResult]  # Kết quả QR detection
    componentResult: Optional[ComponentResult]  # Các vùng đã cắt
    ocrResult: Optional[OcrResult]         # Kết quả OCR thô
    processingTimeMs: float                # Thời gian xử lý

class OcrPipelineService:
    """Service điều phối OCR pipeline với debug output."""
    
    def __init__(
        self,
        qrDetector: IQrDetector,
        componentExtractor: IComponentExtractor,
        ocrExtractor: IOcrExtractor,
        textProcessor: ITextProcessor,
        debugEnabled: bool = False,
        debugBasePath: str = "output/debug"
    ):
        self._qrDetector = qrDetector
        self._componentExtractor = componentExtractor
        self._ocrExtractor = ocrExtractor
        self._textProcessor = textProcessor
        self._debugEnabled = debugEnabled
        self._debugBasePath = Path(debugBasePath)
        self._logger = logging.getLogger(__name__)
        
        # Tạo thư mục debug nếu cần
        if self._debugEnabled:
            for subdir in ["qr-code", "components", "ocr-raw-text", "result"]:
                (self._debugBasePath / subdir).mkdir(parents=True, exist_ok=True)
    
    def process(self, image: np.ndarray, timestamp: str = None) -> Optional[OcrPipelineResult]:
        """
        Xử lý ảnh nhãn qua toàn bộ pipeline.
        
        Args:
            image: Ảnh nhãn đã được preprocess (từ bước 1-4)
            timestamp: Timestamp cho debug files (VD: "20251217_155913_499")
            
        Returns:
            OcrPipelineResult nếu thành công, None nếu thất bại
        """
        startTime = time.time()
        ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
        
        # Bước 5: Detect QR
        self._logger.debug(f"Step 5: Detecting QR code...")
        qrResult = self._qrDetector.detect(image)
        if qrResult is None:
            self._logger.warning("No QR code detected")
            return None
        self._saveQrDebug(qrResult, ts)
        
        # Bước 6: Extract components
        self._logger.debug(f"Step 6: Extracting components...")
        componentResult = self._componentExtractor.extractAndMerge(
            image, qrResult.polygon
        )
        if componentResult is None:
            self._logger.warning("Failed to extract components")
            return None
        self._saveComponentsDebug(componentResult, ts)
        
        # Bước 7: OCR
        self._logger.debug(f"Step 7: Running OCR...")
        ocrResult = self._ocrExtractor.extract(componentResult.mergedImage)
        if not ocrResult.textBlocks:
            self._logger.warning("OCR returned no text")
            return None
        self._saveOcrDebug(ocrResult, ts)
        
        # Bước 8: Post-process
        self._logger.debug(f"Step 8: Post-processing text...")
        labelData = self._textProcessor.process(ocrResult.textBlocks, qrResult)
        
        processingTimeMs = (time.time() - startTime) * 1000
        
        result = OcrPipelineResult(
            labelData=labelData,
            qrResult=qrResult,
            componentResult=componentResult,
            ocrResult=ocrResult,
            processingTimeMs=processingTimeMs
        )
        
        self._saveResultDebug(result, ts)
        self._logger.info(f"OCR pipeline completed in {processingTimeMs:.2f}ms, valid={labelData.isValid}")
        
        return result
    
    def _saveQrDebug(self, qrResult: QrDetectionResult, ts: str):
        """Bước 5: Lưu kết quả QR detection."""
        if not self._debugEnabled:
            return
        data = {
            "text": qrResult.text,
            "polygon": qrResult.polygon,
            "rect": qrResult.rect,
            "confidence": qrResult.confidence,
            "parsed": {
                "dateCode": qrResult.dateCode,
                "facility": qrResult.facility,
                "orderType": qrResult.orderType,
                "orderNumber": qrResult.orderNumber,
                "position": qrResult.position
            }
        }
        path = self._debugBasePath / "qr-code" / f"qr_{ts}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _saveComponentsDebug(self, componentResult: ComponentResult, ts: str):
        """Bước 6: Lưu ảnh các vùng đã cắt."""
        if not self._debugEnabled:
            return
        basePath = self._debugBasePath / "components"
        cv2.imwrite(str(basePath / f"merged_{ts}.png"), componentResult.mergedImage)
        cv2.imwrite(str(basePath / f"above_qr_{ts}.png"), componentResult.aboveQrRoi)
        cv2.imwrite(str(basePath / f"below_qr_{ts}.png"), componentResult.belowQrRoi)
    
    def _saveOcrDebug(self, ocrResult: OcrResult, ts: str):
        """Bước 7: Lưu kết quả OCR thô."""
        if not self._debugEnabled:
            return
        data = {
            "textBlocks": [
                {
                    "text": block.text,
                    "confidence": block.confidence,
                    "bbox": block.bbox
                } for block in ocrResult.textBlocks
            ]
        }
        path = self._debugBasePath / "ocr-raw-text" / f"ocr_{ts}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _saveResultDebug(self, result: OcrPipelineResult, ts: str):
        """Bước 8: Lưu kết quả cuối cùng."""
        if not self._debugEnabled:
            return
        data = {
            "labelData": asdict(result.labelData),
            "processingTimeMs": result.processingTimeMs
        }
        path = self._debugBasePath / "result" / f"result_{ts}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
```

---

## 4. Cấu hình (config/app_config.json)

Thêm cấu hình mới cho OCR pipeline:

```json
{
  "...existing config...": "...",
  
  "ocrPipeline": {
    "enabled": true,
    "debugEnabled": true,
    "debugBasePath": "output/debug",
    
    "qrDetector": {
      "symbolTypes": ["QRCODE"]
    },
    
    "componentExtractor": {
      "aboveQr": {
        "widthRatio": 0.3,
        "heightRatio": 0.15
      },
      "belowQr": {
        "widthRatio": 0.6,
        "heightRatio": 0.4
      }
    },
    
    "ocr": {
      "lang": "en",
      "useAngleCls": true,
      "useGpu": false,
      "detDbThresh": 0.3,
      "detDbBoxThresh": 0.5,
      "clsThresh": 0.9
    },
    
    "textProcessor": {
      "minFuzzyScore": 0.75,
      "validSizesPath": "data/sizes.json",
      "validColorsPath": "data/colors.json",
      "validProductsPath": "data/products.json"
    }
  }
}
```

**Cấu trúc file JSON:**
- **colors.json**: `[{"name": "MIDNIGHT", "hex": "#191970", "basic_color": "BLUE"}, ...]`
- **products.json**: `[{"Code": "340", "Name": "Product Name", "Type": "TYPE"}, ...]`
- **sizes.json**: `[{"name": "3T"}, {"name": "XL"}, ...]`

---

## 5. Dependencies mới

Thêm vào `requirements.txt`:

```
# QR Code Detection
pyzbar>=0.1.9

# OCR (CPU only - KHÔNG sử dụng GPU)
paddlepaddle>=2.5.0
paddleocr>=2.7.0
```

**Lưu ý cài đặt trên Linux**:
```bash
# Cài đặt libzbar cho pyzbar
sudo apt-get install libzbar0
```

---

## 6. Cấu trúc thư mục output debug

```
output/debug/
├── qr-code/           # Bước 5: Kết quả QR detection
│   └── qr_20251217_155913.json
│       {
│         "text": "110125-VA-M-000002-2",
│         "polygon": [[x1,y1], ...],
│         "parsed": {
│           "dateCode": "110125",
│           "facility": "VA",
│           "orderType": "M",
│           "orderNumber": "000002",
│           "position": 2
│         }
│       }
│
├── components/        # Bước 6: Ảnh các vùng đã cắt
│   ├── merged_20251217_155913.png
│   ├── above_qr_20251217_155913.png  # position/quantity region
│   └── below_qr_20251217_155913.png  # product/size/color region
│
├── ocr-raw-text/      # Bước 7: Kết quả OCR thô
│   └── ocr_20251217_155913.json
│       {
│         "textBlocks": [
│           {"text": "1/1", "confidence": 0.95, "bbox": [...]},
│           {"text": "340", "confidence": 0.98, "bbox": [...]},
│           {"text": "3T", "confidence": 0.92, "bbox": [...]},
│           {"text": "MIDNIGHT", "confidence": 0.89, "bbox": [...]}
│         ]
│       }
│
└── result/            # Bước 8: Kết quả cuối cùng
    └── result_20251217_155913.json
        {
          "labelData": {
            "fullOrderCode": "110125-VA-M-000002-2",
            "dateCode": "110125",
            "facility": "VA",
            "orderType": "M",
            "orderNumber": "000002",
            "qrPosition": 2,
            "positionQuantity": "1/1",
            "ocrPosition": 1,
            "quantity": 1,
            "productCode": "340",
            "size": "3T",
            "color": "MIDNIGHT",
            "isValid": false
          },
          "processingTimeMs": 150.5
        }
```

---

## 7. Danh sách công việc (TODO)

### Phase 1: Chuẩn bị dữ liệu
- [ ] Copy thư mục `data/` từ `projects/ocr-labels-project/demo_ocr_label/data/`
  - [ ] `data/colors.json` (4904 dòng)
  - [ ] `data/products.json` (2172 dòng)
  - [ ] `data/sizes.json` (138 dòng)
- [ ] Tạo thư mục debug: `output/debug/{qr-code,components,ocr-raw-text,result}/`

### Phase 2: Core Interfaces
- [ ] Tạo `core/interfaces/qr_detector_interface.py`
- [ ] Tạo `core/interfaces/component_extractor_interface.py`
- [ ] Tạo `core/interfaces/ocr_extractor_interface.py`
- [ ] Tạo `core/interfaces/text_processor_interface.py`

### Phase 3: Implementations
- [ ] Tạo `core/qr/__init__.py`
- [ ] Tạo `core/qr/pyzbar_qr_detector.py`
- [ ] Tạo `core/extractor/__init__.py`
- [ ] Tạo `core/extractor/label_component_extractor.py`
- [ ] Tạo `core/ocr/__init__.py`
- [ ] Tạo `core/ocr/paddle_ocr_extractor.py`
- [ ] Tạo `core/processor/__init__.py`
- [ ] Tạo `core/processor/fuzzy_matcher.py`
- [ ] Tạo `core/processor/label_text_processor.py`

### Phase 4: Service Layer
- [ ] Tạo `services/ocr_pipeline_service.py`
- [ ] Cập nhật `main.py` với DI cho OCR pipeline
- [ ] Cập nhật `config/app_config.json`

### Phase 5: UI Integration
- [ ] Tạo `ui/widgets/ocr_result_widget.py` để hiển thị kết quả cuối cùng
- [ ] Tích hợp widget vào `main_window.py`
- [ ] Thêm toggle bật/tắt OCR pipeline

### Phase 6: Testing & Documentation
- [ ] Viết unit tests cho `FuzzyMatcher`
- [ ] Viết integration tests cho pipeline
- [ ] Cập nhật `CHANGELOG.md`
- [ ] Cập nhật `README.md`
- [ ] Cập nhật `SPECIFICATION.md`
- [ ] Cập nhật `requirements.txt`

---

## 8. Ước tính thời gian

| Phase | Công việc | Thời gian ước tính |
|-------|-----------|-------------------|
| 1 | Chuẩn bị dữ liệu | 0.5 giờ |
| 2 | Core Interfaces | 1-2 giờ |
| 3 | Implementations | 4-6 giờ |
| 4 | Service Layer | 2-3 giờ |
| 5 | UI Integration | 2-3 giờ |
| 6 | Testing & Documentation | 2-3 giờ |
| **Tổng** | | **11.5-17.5 giờ** |

---

*Tài liệu này được tạo dựa trên:*
- *Phân tích mã nguồn từ `projects/ocr-labels-project`*
- *Ảnh mẫu nhãn thực tế do người dùng cung cấp*
- *Tài liệu API của pyzbar và PaddleOCR*
