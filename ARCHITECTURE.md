# System Architecture

## Overview

This document describes the current architecture of the Label Detector application. The system follows a layered architecture with clear separation of concerns, adhering to SOLID principles.

## Architecture Layers

The application is organized into four main layers:

```
┌─────────────────────────────────────────────────────┐
│              UI Layer / Scripts Layer               │
│         (main_window.py, detection.py)              │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Orchestrator Layer                     │
│           (PipelineOrchestrator)                    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Services Layer                        │
│    (S1-S8 Services + ConfigService)                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                Core Layer                           │
│     (Detectors, Preprocessors, OCR, etc.)           │
└─────────────────────────────────────────────────────┘
```

## Layer Descriptions

### 1. Core Layer

**Location**: `core/`

**Purpose**: Contains the main processing logic and business rules.

**Responsibilities**:
- Pure processing logic without any dependency on configuration or services
- Implements specific algorithms and operations
- Defines interfaces for extensibility

**Components**:
- **Interfaces** (`core/interfaces/`):
  - `IDetector`: Object detection interface
  - `IPreprocessor`: Image preprocessing interface
  - `IEnhancer`: Image enhancement interface
  - `IQrDetector`: QR code detection interface
  - `IOcrExtractor`: OCR extraction interface
  - `ITextProcessor`: Text processing interface
  - `IComponentExtractor`: Component extraction interface
  - `IWriter`: Output writing interface

- **Implementations**:
  - `core/detector/`: YOLO detectors (ONNX, OpenVINO)
  - `core/preprocessor/`: Geometric transformation, orientation correction
  - `core/enhancer/`: Brightness, sharpness enhancement
  - `core/qr/`: QR code detection (pyzbar, zxing)
  - `core/ocr/`: OCR extraction (PaddleOCR)
  - `core/processor/`: Fuzzy matching, text processing
  - `core/extractor/`: Label component extraction
  - `core/writer/`: Local file writing
  - `core/camera/`: Camera capture

**Characteristics**:
- No dependency on services or configuration
- Reusable across different applications
- Testable in isolation
- Each class follows Single Responsibility Principle

### 2. Services Layer

**Location**: `services/impl/`

**Purpose**: Manages the lifecycle of core components and provides additional functionality like logging, debugging, and timing.

**Responsibilities**:
- Initialize core components
- Handle logging and error reporting
- Manage debug output (save intermediate results)
- Track performance metrics and timing
- Receive configuration as constructor parameters
- No direct dependency on ConfigService

**Components**:
- `S1CameraService`: Camera capture service
- `S2DetectionService`: Object detection service
- `S3PreprocessingService`: Image preprocessing service
- `S4EnhancementService`: Image enhancement service
- `S5QrDetectionService`: QR code detection service
- `S6ComponentExtractionService`: Component extraction service
- `S7OcrService`: OCR extraction service
- `S8PostprocessingService`: Text processing and validation service
- `ConfigService`: Configuration management (special service)

**Initialization Pattern**:
```python
class S2DetectionService:
    def __init__(
        self,
        backend: str,
        modelPath: str,
        inputSize: int,
        confidenceThreshold: float,
        debugBasePath: str,
        debugEnabled: bool,
        # ... other parameters
    ):
        # Create core component internally
        self._detector = createDetector(
            backend=backend,
            modelPath=modelPath,
            inputSize=inputSize
        )
```

**Key Points**:
- Services receive all configuration as constructor parameters
- Services create their own core components
- Services do not depend on ConfigService
- Services handle debug output and timing

### 3. ConfigService (Special Service)

**Location**: `services/impl/config_service.py`

**Purpose**: Centralized configuration management, separate from other services.

**Responsibilities**:
- Read configuration from JSON file
- Provide typed getters for configuration values
- Validate configuration
- Provide default values

**Characteristics**:
- Does not depend on any other service
- Used only by the Orchestrator
- Other services do not depend on it
- Implements configuration schema

**Example Methods**:
```python
class ConfigService:
    def getDetectionBackend(self) -> str
    def getModelPath(self) -> str
    def getInputSize(self) -> int
    def getConfidenceThreshold(self) -> float
    def isDebugEnabled(self) -> bool
    # ... many more getters
```

### 4. Orchestrator Layer

**Location**: `ui/pipeline_orchestrator.py`

**Purpose**: Coordinates the initialization and execution of all services.

**Responsibilities**:
- Initialize ConfigService first
- Read all configuration values from ConfigService
- Create all services with configuration parameters
- Coordinate pipeline execution
- Provide service access to UI/scripts

**Initialization Flow**:
```python
class PipelineOrchestrator:
    def __init__(self, configPath: str):
        # Step 1: Initialize ConfigService
        self._configService = ConfigService(configPath)
        
        # Step 2: Read configuration values
        backend = self._configService.getDetectionBackend()
        modelPath = self._configService.getModelPath()
        debugBasePath = self._configService.getDebugBasePath()
        # ... read all needed config
        
        # Step 3: Initialize all services with parameters
        self._s2DetectionService = S2DetectionService(
            backend=backend,
            modelPath=modelPath,
            debugBasePath=debugBasePath,
            # ... pass all parameters
        )
        # ... initialize other services
```

**Pipeline Execution**:
```python
def processFrame(self, frameId: str, image: np.ndarray):
    # S1: Camera (already has image)
    
    # S2: Detection
    detectionResult = self._s2DetectionService.detect(image)
    
    # S3: Preprocessing
    preprocessResult = self._s3PreprocessingService.process(
        detectionResult.croppedImage
    )
    
    # S4: Enhancement
    enhancedImage = self._s4EnhancementService.enhance(
        preprocessResult.image
    )
    
    # ... continue with S5-S8
```

### 5. UI / Scripts Layer

**Location**: `ui/`, `scripts/`

**Purpose**: User interface and command-line scripts.

**Responsibilities**:
- Create PipelineOrchestrator with config path
- Use orchestrator to access services
- Handle user interaction
- Display results
- Execute batch processing

**Components**:
- `ui/main_window.py`: Main GUI application
- `scripts/detection.py`: Batch detection script
- `scripts/test_openvino.py`: Backend testing script

**Usage Pattern**:
```python
# In UI or script
orchestrator = PipelineOrchestrator("config/application_config.json")

# Access services through orchestrator
result = orchestrator.processFrame(frameId, image)

# Or access individual services
s2Service = orchestrator.getS2DetectionService()
detections = s2Service.detect(image)
```

## Dependency Flow

```
UI/Scripts
    │
    └──> PipelineOrchestrator
            │
            ├──> ConfigService (reads config)
            │
            └──> S1-S8 Services (initialized with config values)
                    │
                    └──> Core Components (created by services)
```

**Important Rules**:
1. Services do NOT depend on ConfigService
2. Services receive configuration as constructor parameters
3. Only PipelineOrchestrator depends on ConfigService
4. Core components do NOT depend on services
5. Each layer only depends on the layer below it

## Design Principles Applied

### Single Responsibility Principle (SRP)
- Each core component handles one specific operation
- Each service manages one pipeline step
- ConfigService only handles configuration
- Orchestrator only handles coordination

### Open/Closed Principle (OCP)
- New backends can be added without modifying existing code (Factory Pattern)
- New services can be added without changing the orchestrator structure
- Core components implement interfaces for extensibility

### Liskov Substitution Principle (LSP)
- All detectors implement IDetector and are interchangeable
- All QR detectors implement IQrDetector and are interchangeable
- Services can use any implementation of core interfaces

### Interface Segregation Principle (ISP)
- Small, focused interfaces (IDetector, IEnhancer, etc.)
- Services only depend on interfaces they actually use
- No client is forced to depend on unused methods

### Dependency Inversion Principle (DIP)
- Services depend on abstractions (interfaces), not concrete classes
- High-level modules (services) do not depend on low-level modules (core)
- Both depend on abstractions defined in core/interfaces/

## Configuration Flow

```
application_config.json
        │
        ▼
    ConfigService (reads and validates)
        │
        ▼
PipelineOrchestrator (extracts values)
        │
        ▼
Services (receive as constructor parameters)
        │
        ▼
Core Components (initialized with parameters)
```

## Key Architectural Decisions

### Why Services Don't Depend on ConfigService?
- **Testability**: Services can be tested without configuration files
- **Flexibility**: Services can be reused in different contexts
- **Explicit Dependencies**: All dependencies are visible in constructor
- **Decoupling**: Changes to config format don't affect services

### Why Use Factory Pattern?
- **OCP Compliance**: Add new backends without modifying existing code
- **Encapsulation**: Hide complex creation logic
- **Flexibility**: Runtime backend selection based on config

### Why Separate Orchestrator?
- **Centralized Coordination**: One place to manage all services
- **Single Configuration Point**: ConfigService used only here
- **Clear Entry Point**: UI/scripts have one interface to the system
- **Easier Testing**: Can mock orchestrator for UI testing

## Error Handling

Each layer handles errors appropriately:

**Core Layer**:
- Returns None or empty results on failure
- Logs errors with context
- Does not throw exceptions to caller

**Services Layer**:
- Catches core layer errors
- Adds service-level context to logs
- Returns safe default values
- Tracks errors in timing metrics

**Orchestrator Layer**:
- Aggregates errors from services
- Decides how to handle pipeline failures
- Provides error information to UI

**UI Layer**:
- Displays errors to user
- Provides retry mechanisms
- Shows partial results when possible

## Testing Strategy

**Core Layer**:
- Unit tests for each component
- Mock dependencies (interfaces)
- Test edge cases and error conditions

**Services Layer**:
- Integration tests with real core components
- Mock debug/logging outputs
- Test parameter validation

**Orchestrator**:
- End-to-end tests
- Mock services for faster tests
- Test configuration loading

**UI/Scripts**:
- Manual testing
- Automated UI tests (optional)
- Batch processing verification

## Future Extensibility

The architecture supports:

1. **New Backends**: Add new detector implementations (TensorRT, CoreML, etc.)
2. **New Services**: Add new pipeline steps (S9, S10, etc.)
3. **Multiple Pipelines**: Create different orchestrators for different use cases
4. **Alternative UIs**: Web UI, CLI, API can all use the same orchestrator
5. **Cloud Deployment**: Services can be deployed as microservices
6. **Async Processing**: Services can be made async without changing architecture

## Conclusion

This architecture provides:
- Clear separation of concerns
- High testability
- Easy maintenance and extension
- Flexibility to change configuration without code changes
- Reusable components across different applications
- Strong adherence to SOLID principles
