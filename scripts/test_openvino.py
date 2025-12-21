#!/usr/bin/env python3
"""
Test script to verify OpenVINO backend implementation.

Tests:
1. Factory function works correctly
2. Both backends can be created
3. Backend availability detection
4. Model loading (if models exist)

Usage:
    python test_openvino.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.detector import createDetector, getSupportedBackends, isBackendAvailable


def test_factory_function():
    """Test that factory function exists and works."""
    print("=" * 60)
    print("TEST 1: Factory Function")
    print("=" * 60)
    
    try:
        # Test getting supported backends
        backends = getSupportedBackends()
        print(f"✓ Supported backends: {backends}")
        assert "onnx" in backends
        assert "openvino" in backends
        print("✓ Factory function works correctly")
        return True
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")
        return False


def test_backend_availability():
    """Test backend availability detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Backend Availability")
    print("=" * 60)
    
    try:
        onnx_available = isBackendAvailable("onnx")
        openvino_available = isBackendAvailable("openvino")
        
        print(f"ONNX Runtime available: {onnx_available}")
        print(f"OpenVINO Runtime available: {openvino_available}")
        
        if not onnx_available:
            print("⚠ ONNX Runtime not installed!")
            print("  Install: pip install onnxruntime>=1.16.0")
        
        if not openvino_available:
            print("⚠ OpenVINO Runtime not installed (optional)")
            print("  Install: pip install openvino>=2024.0.0")
        
        print("✓ Backend availability detection works")
        return True
    except Exception as e:
        print(f"✗ Backend availability test failed: {e}")
        return False


def test_create_onnx_detector():
    """Test creating ONNX detector."""
    print("\n" + "=" * 60)
    print("TEST 3: Create ONNX Detector")
    print("=" * 60)
    
    if not isBackendAvailable("onnx"):
        print("⚠ Skipping test - ONNX Runtime not available")
        return True
    
    try:
        detector = createDetector(
            backend="onnx",
            modelPath="",  # Don't load model in test
            inputSize=640,
            classNames=["label"],
            isSegmentation=True
        )
        print(f"✓ ONNX detector created: {type(detector).__name__}")
        print(f"  Class names: {detector.getClassNames()}")
        return True
    except Exception as e:
        print(f"✗ ONNX detector creation failed: {e}")
        return False


def test_create_openvino_detector():
    """Test creating OpenVINO detector."""
    print("\n" + "=" * 60)
    print("TEST 4: Create OpenVINO Detector")
    print("=" * 60)
    
    if not isBackendAvailable("openvino"):
        print("⚠ Skipping test - OpenVINO Runtime not available")
        print("  This is optional. Install with: pip install openvino>=2024.0.0")
        return True
    
    try:
        detector = createDetector(
            backend="openvino",
            modelPath="",  # Don't load model in test
            inputSize=640,
            classNames=["label"],
            isSegmentation=True
        )
        print(f"✓ OpenVINO detector created: {type(detector).__name__}")
        print(f"  Class names: {detector.getClassNames()}")
        return True
    except Exception as e:
        print(f"✗ OpenVINO detector creation failed: {e}")
        return False


def test_invalid_backend():
    """Test error handling for invalid backend."""
    print("\n" + "=" * 60)
    print("TEST 5: Invalid Backend Error Handling")
    print("=" * 60)
    
    try:
        detector = createDetector(
            backend="invalid_backend",
            modelPath="",
            inputSize=640
        )
        print("✗ Should have raised ValueError for invalid backend")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected exception: {e}")
        return False


def test_model_loading():
    """Test model loading if models exist."""
    print("\n" + "=" * 60)
    print("TEST 6: Model Loading (Optional)")
    print("=" * 60)
    
    onnx_model_path = PROJECT_ROOT / "models" / "yolo11n-seg-version-1-0-0.onnx"
    openvino_model_path = PROJECT_ROOT / "models" / "yolo11n-seg-version-1-0-0_int8_openvino_model" / "yolo11n-seg-version-1-0-0.xml"
    
    success = True
    
    # Test ONNX model loading
    if onnx_model_path.exists() and isBackendAvailable("onnx"):
        try:
            detector = createDetector(
                backend="onnx",
                modelPath=str(onnx_model_path),
                inputSize=640,
                classNames=["label"],
                isSegmentation=True
            )
            print(f"✓ ONNX model loaded: {onnx_model_path.name}")
        except Exception as e:
            print(f"✗ ONNX model loading failed: {e}")
            success = False
    else:
        print(f"⚠ ONNX model not found: {onnx_model_path}")
    
    # Test OpenVINO model loading
    if openvino_model_path.exists() and isBackendAvailable("openvino"):
        try:
            detector = createDetector(
                backend="openvino",
                modelPath=str(openvino_model_path),
                inputSize=640,
                classNames=["label"],
                isSegmentation=True
            )
            print(f"✓ OpenVINO model loaded: {openvino_model_path.name}")
        except Exception as e:
            print(f"✗ OpenVINO model loading failed: {e}")
            success = False
    else:
        print(f"⚠ OpenVINO model not found: {openvino_model_path}")
    
    return success


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OpenVINO Backend Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Factory Function", test_factory_function),
        ("Backend Availability", test_backend_availability),
        ("Create ONNX Detector", test_create_onnx_detector),
        ("Create OpenVINO Detector", test_create_openvino_detector),
        ("Invalid Backend Handling", test_invalid_backend),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            results.append((test_name, test_func()))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
