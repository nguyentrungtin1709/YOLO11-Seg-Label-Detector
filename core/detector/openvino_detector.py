"""
OpenVINO Detector Implementation

Implements IDetector using OpenVINO Runtime for YOLO11n and YOLO11n-seg model inference.
Supports both object detection and instance segmentation.
Follows SRP: Only handles detection/segmentation operations.

OpenVINO provides optimized inference on Intel hardware (CPU/GPU/VPU).
INT8 quantized models achieve 2-4x speedup compared to ONNX Runtime FP32 on Intel CPUs.
"""

import logging
from typing import List, Tuple, Optional, Dict
import time
import numpy as np
import cv2

try:
    from openvino.runtime import Core, Model, CompiledModel, InferRequest
except ImportError:
    Core = None
    Model = None
    CompiledModel = None
    InferRequest = None

from core.interfaces.detector_interface import IDetector, Detection


logger = logging.getLogger(__name__)


class OpenVINODetector(IDetector):
    """
    YOLO object detector and instance segmentor using OpenVINO Runtime.
    
    Supports YOLO11n (detection) and YOLO11n-seg (segmentation) 
    architectures exported to OpenVINO IR format (.xml + .bin).
    Handles preprocessing, inference, and postprocessing including NMS.
    
    Optimized for Intel hardware with INT8 quantization support.
    
    Follows SRP: Only responsible for detection/segmentation operations.
    """
    
    def __init__(
        self, 
        inputSize: int = 640, 
        classNames: Optional[List[str]] = None,
        isSegmentation: bool = False
    ):
        """
        Initialize OpenVINODetector.
        
        Args:
            inputSize: Input image size for the model (default: 640).
            classNames: List of class names the model can detect.
            isSegmentation: If True, enable instance segmentation with mask output.
        """
        self._core: Optional[Core] = None
        self._model: Optional[Model] = None
        self._compiledModel: Optional[CompiledModel] = None
        self._inferRequest: Optional[InferRequest] = None
        self._inputName: str = ""
        self._outputNames: List[str] = []
        self._inputSize = inputSize
        self._classNames = classNames or ["label"]
        self._isSegmentation = isSegmentation
        
        # NMS parameters
        self._nmsThreshold = 0.45
        self._maxDetections = 100
        
        # Segmentation parameters
        self._numMaskCoeffs = 32  # YOLO11-seg uses 32 mask coefficients
        self._protoMaskSize = 160  # Proto mask resolution is 160x160
    
    def loadModel(self, modelPath: str) -> bool:
        """
        Load OpenVINO model from IR file (.xml).
        
        Args:
            modelPath: Path to the OpenVINO IR model file (.xml).
                      The .bin file should be in the same directory.
            
        Returns:
            bool: True if model loaded successfully.
        """
        if Core is None:
            logger.error("OpenVINO Runtime is not installed")
            logger.error("Install with: pip install openvino>=2024.0.0")
            return False
        
        try:
            # Initialize OpenVINO Core
            self._core = Core()
            
            # Read model from IR files (.xml and .bin)
            logger.info(f"Loading OpenVINO model from: {modelPath}")
            self._model = self._core.read_model(model=modelPath)
            
            # Compile model for CPU device (can be changed to GPU, AUTO, etc.)
            # CPU is most universal, GPU requires Intel GPU
            device = "CPU"
            logger.info(f"Compiling model for device: {device}")
            self._compiledModel = self._core.compile_model(self._model, device)
            
            # Create infer request for synchronous inference
            self._inferRequest = self._compiledModel.create_infer_request()
            
            # Get input/output information
            # Use index instead of name for models without tensor names
            inputLayer = self._compiledModel.input(0)
            try:
                self._inputName = inputLayer.any_name
            except:
                # Fallback to index if model doesn't have names
                self._inputName = 0
            
            # Get all output indices (use indices instead of names)
            self._outputIndices = list(range(len(self._compiledModel.outputs)))
            self._outputNames = []
            for i in self._outputIndices:
                try:
                    output = self._compiledModel.output(i)
                    self._outputNames.append(output.any_name)
                except:
                    # Use index if no name available
                    self._outputNames.append(i)
            
            logger.info(f"Model loaded successfully from {modelPath}")
            logger.info(f"Input name: {self._inputName}")
            logger.info(f"Output names: {self._outputNames}")
            logger.info(f"Output count: {len(self._outputNames)}")
            logger.info(f"Segmentation mode: {self._isSegmentation}")
            
            # Validate segmentation model has expected outputs
            if self._isSegmentation and len(self._outputNames) < 2:
                logger.warning(
                    f"Segmentation mode enabled but model has only {len(self._outputNames)} outputs. "
                    "Expected 2 outputs (bbox + proto masks). Masks may not be available."
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            self._compiledModel = None
            return False
    
    def detect(self, image: np.ndarray, confidenceThreshold: float) -> List[Detection]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array (BGR format).
            confidenceThreshold: Minimum confidence score.
            
        Returns:
            List[Detection]: List of detected objects.
        """
        detections, _ = self.detectWithTiming(image, confidenceThreshold)
        return detections
    
    def detectWithTiming(
        self, 
        image: np.ndarray, 
        confidenceThreshold: float
    ) -> Tuple[List[Detection], Dict[str, float]]:
        """
        Detect objects in an image with timing information.
        
        Args:
            image: Input image as numpy array (BGR format).
            confidenceThreshold: Minimum confidence score.
            
        Returns:
            Tuple of (List[Detection], Dict with timing in ms).
            Timing dict keys: 'preprocess', 'inference', 'postprocess'
        """
        timing = {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0}
        
        if self._compiledModel is None or self._inferRequest is None:
            logger.warning("Model not loaded, returning empty detections")
            return [], timing
        
        try:
            # Get original image dimensions
            originalHeight, originalWidth = image.shape[:2]
            
            # Preprocess with timing
            startTime = time.perf_counter()
            inputTensor = self._preprocess(image)
            timing['preprocess'] = (time.perf_counter() - startTime) * 1000
            
            # Inference with timing
            startTime = time.perf_counter()
            
            # Infer using index or name
            if isinstance(self._inputName, int):
                self._inferRequest.infer({0: inputTensor})
            else:
                self._inferRequest.infer({self._inputName: inputTensor})
            
            # Get outputs using indices
            outputs = []
            for i in range(len(self._compiledModel.outputs)):
                output = self._inferRequest.get_output_tensor(i).data
                outputs.append(output)
            
            timing['inference'] = (time.perf_counter() - startTime) * 1000
            
            # Postprocess with timing
            startTime = time.perf_counter()
            detections = self._postprocess(
                outputs, 
                originalWidth, 
                originalHeight, 
                confidenceThreshold
            )
            timing['postprocess'] = (time.perf_counter() - startTime) * 1000
            
            return detections, timing
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [], timing
    
    def getClassNames(self) -> List[str]:
        """
        Get the list of class names.
        
        Returns:
            List[str]: List of class names.
        """
        return self._classNames
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO inference.
        
        Steps:
        1. Resize to input size
        2. Convert BGR to RGB
        3. Normalize to [0, 1]
        4. Transpose HWC to CHW
        5. Add batch dimension
        
        Args:
            image: Input image in BGR format.
            
        Returns:
            np.ndarray: Preprocessed image tensor.
        """
        # Resize with letterbox (maintain aspect ratio)
        resized = cv2.resize(image, (self._inputSize, self._inputSize))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _postprocess(
        self, 
        outputs: List[np.ndarray], 
        originalWidth: int, 
        originalHeight: int,
        confidenceThreshold: float
    ) -> List[Detection]:
        """
        Postprocess YOLO output.
        
        For detection models:
            - outputs[0]: [1, 4+num_classes, num_detections]
        
        For segmentation models:
            - outputs[0]: [1, 4+num_classes+32, num_detections] (bbox + class + mask coeffs)
            - outputs[1]: [1, 32, 160, 160] (proto masks)
        
        Steps:
        1. Parse raw output
        2. Filter by confidence
        3. Scale boxes to original image size
        4. Apply NMS
        5. Decode masks (if segmentation)
        
        Args:
            outputs: Raw model outputs.
            originalWidth: Original image width.
            originalHeight: Original image height.
            confidenceThreshold: Minimum confidence threshold.
            
        Returns:
            List[Detection]: Processed detections with optional masks.
        """
        detections = []
        
        # Get the main output (typically shape: [1, num_classes+4(+32), num_detections])
        output = outputs[0]
        
        # Get proto masks if segmentation model
        protoMasks = None
        if self._isSegmentation and len(outputs) >= 2:
            protoMasks = outputs[1]  # Shape: [1, 32, 160, 160]
        
        # YOLO output format: [1, 4+num_classes(+32), num_detections]
        # Transpose to [num_detections, 4+num_classes(+32)]
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
            
            # Calculate expected output dimension
            numClasses = len(self._classNames)
            expectedDim = 4 + numClasses
            if self._isSegmentation:
                expectedDim += self._numMaskCoeffs  # +32 for mask coefficients
            
            # Check if we need to transpose (YOLO format)
            if output.shape[0] <= expectedDim + 10:  # Allow some tolerance
                output = output.T
        
        # Calculate scale factors
        scaleX = originalWidth / self._inputSize
        scaleY = originalHeight / self._inputSize
        
        boxes = []
        scores = []
        classIds = []
        maskCoeffsList = []
        
        numClasses = len(self._classNames)
        
        for detection in output:
            # YOLO format: [x_center, y_center, width, height, class_scores..., (mask_coeffs...)]
            if len(detection) < 5:
                continue
            
            # Get class scores (after bbox, before mask coefficients)
            classScores = detection[4:4+numClasses]
            classId = int(np.argmax(classScores))
            confidence = float(classScores[classId])
            
            # Filter by confidence
            if confidence < confidenceThreshold:
                continue
            
            # Get bbox (center format)
            xCenter = detection[0] * scaleX
            yCenter = detection[1] * scaleY
            width = detection[2] * scaleX
            height = detection[3] * scaleY
            
            # Convert to corner format
            x1 = int(xCenter - width / 2)
            y1 = int(yCenter - height / 2)
            x2 = int(xCenter + width / 2)
            y2 = int(yCenter + height / 2)
            
            # Clip to image bounds
            x1 = max(0, min(x1, originalWidth - 1))
            y1 = max(0, min(y1, originalHeight - 1))
            x2 = max(0, min(x2, originalWidth))
            y2 = max(0, min(y2, originalHeight))
            
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h] for NMS
            scores.append(confidence)
            classIds.append(classId)
            
            # Get mask coefficients if segmentation
            if self._isSegmentation and len(detection) >= 4 + numClasses + self._numMaskCoeffs:
                maskCoeffs = detection[4+numClasses:4+numClasses+self._numMaskCoeffs]
                maskCoeffsList.append(maskCoeffs)
            else:
                maskCoeffsList.append(None)
        
        # Apply NMS
        if len(boxes) == 0:
            return []
        
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            scores, 
            confidenceThreshold, 
            self._nmsThreshold,
            top_k=self._maxDetections
        )
        
        # Create Detection objects
        for idx in indices:
            i = int(idx)
            x, y, w, h = boxes[i]
            bbox = (x, y, x + w, y + h)
            className = self._classNames[classIds[i]]
            confidence = scores[i]
            
            # Decode mask if available
            mask = None
            if self._isSegmentation and maskCoeffsList[i] is not None and protoMasks is not None:
                mask = self._decodeMask(
                    maskCoeffsList[i],
                    protoMasks,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    originalWidth,
                    originalHeight
                )
            
            detection = Detection(
                bbox=bbox,
                className=className,
                confidence=confidence,
                mask=mask
            )
            detections.append(detection)
        
        return detections
    
    def _decodeMask(
        self,
        maskCoeffs: np.ndarray,
        protoMasks: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        originalWidth: int,
        originalHeight: int
    ) -> np.ndarray:
        """
        Decode segmentation mask from coefficients and proto masks.
        
        Formula: mask = sigmoid(maskCoeffs @ protoMasks)
        
        Args:
            maskCoeffs: Mask coefficients for one object [32].
            protoMasks: Prototype masks from model output [1, 32, 160, 160].
            x1, y1, x2, y2: Bounding box coordinates in original image.
            originalWidth, originalHeight: Original image dimensions.
        
        Returns:
            Binary mask (H x W) with dtype=uint8, values 0 or 255.
        """
        try:
            # Remove batch dimension from proto masks: [1, 32, 160, 160] -> [32, 160, 160]
            if len(protoMasks.shape) == 4:
                protoMasks = protoMasks[0]
            
            # Reshape proto masks for matrix multiplication: [32, 160*160]
            protoFlat = protoMasks.reshape(self._numMaskCoeffs, -1)
            
            # Matrix multiplication: [32] @ [32, 160*160] -> [160*160]
            maskFlat = np.matmul(maskCoeffs, protoFlat)
            
            # Reshape back to 2D: [160, 160]
            protoH, protoW = protoMasks.shape[1], protoMasks.shape[2]
            mask = maskFlat.reshape(protoH, protoW)
            
            # Apply sigmoid activation
            mask = 1.0 / (1.0 + np.exp(-mask))
            
            # Resize mask to original image size
            mask = cv2.resize(mask, (originalWidth, originalHeight), interpolation=cv2.INTER_LINEAR)
            
            # Crop to bounding box and zero out areas outside bbox
            # Create a mask that's only active inside the bounding box
            fullMask = np.zeros((originalHeight, originalWidth), dtype=np.float32)
            
            # Ensure bbox coordinates are valid
            x1c = max(0, min(x1, originalWidth - 1))
            y1c = max(0, min(y1, originalHeight - 1))
            x2c = max(0, min(x2, originalWidth))
            y2c = max(0, min(y2, originalHeight))
            
            if x2c > x1c and y2c > y1c:
                fullMask[y1c:y2c, x1c:x2c] = mask[y1c:y2c, x1c:x2c]
            
            # Convert to binary mask (threshold = 0.5), output as uint8 (0 or 255)
            binaryMask = (fullMask > 0.5).astype(np.uint8) * 255
            
            return binaryMask
            
        except Exception as e:
            logger.error(f"Failed to decode mask: {e}")
            return None
