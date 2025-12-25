"""
================================================================================
BỘ LỌC CHẤT LƯỢNG ẢNH - 4 TIÊU CHÍ
================================================================================

File này chứa 4 hàm lọc ảnh ĐỘC LẬP, chưa tích hợp vào hệ thống.
Mỗi hàm kiểm tra 1 tiêu chí và trả về True/False.

Sử dụng:
    python quality_filters.py <đường_dẫn_ảnh>
    
Ví dụ:
    python quality_filters.py output/debug/s3_preprocessing/image_001.png

================================================================================
"""

import cv2
import numpy as np
import sys
import os


# ==============================================================================
# FILTER 1: SIZE (Kích thước)
# ==============================================================================
def check_size(image, min_width=270, min_height=180):
    """
    Kiểm tra kích thước ảnh.
    
    THAM SỐ:
    --------
    - min_width: int = 270
        Chiều rộng tối thiểu (pixels).
        Ảnh nhỏ hơn sẽ bị loại vì không đủ chi tiết để đọc QR/text.
        
    - min_height: int = 180
        Chiều cao tối thiểu (pixels).
        Thường tỷ lệ label là 3:2 (270x180).
    
    TỐC ĐỘ: ~0.001 ms/ảnh (gần như không tốn thời gian)
    
    Returns:
        (passed, width, height)
    """
    height, width = image.shape[:2]
    passed = (width >= min_width) and (height >= min_height)
    return passed, width, height


# ==============================================================================
# FILTER 2: CONTRAST (Độ tương phản)
# ==============================================================================
def check_contrast(image, min_contrast=30):
    """
    Kiểm tra độ tương phản bằng độ lệch chuẩn (std deviation).
    
    THAM SỐ:
    --------
    - min_contrast: float = 30
        Độ lệch chuẩn tối thiểu của pixel grayscale (0-255).
        
        Giá trị tham khảo:
        - 0-20:  Ảnh rất phẳng (một màu)
        - 20-40: Contrast thấp
        - 40-60: Contrast trung bình
        - 60+:   Contrast cao
        
        Đặt 30 để loại ảnh quá phẳng, không có chi tiết.
        
    CÁCH TÍNH:
    ----------
    1. Chuyển ảnh sang grayscale
    2. Tính std deviation của tất cả pixels
    3. So sánh với ngưỡng
    
    TỐC ĐỘ: ~0.136 ms/ảnh
    
    Returns:
        (passed, contrast_value)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    passed = contrast >= min_contrast
    return passed, round(contrast, 2)


# ==============================================================================
# FILTER 3: SHARPNESS (Độ sắc nét)
# ==============================================================================
def check_sharpness(image, min_sharpness=300):
    """
    Kiểm tra độ sắc nét bằng Laplacian variance.
    
    THAM SỐ:
    --------
    - min_sharpness: float = 300
        Variance của Laplacian tối thiểu.
        
        Giá trị tham khảo:
        - 0-100:    Ảnh rất mờ
        - 100-300:  Ảnh hơi mờ
        - 300-800:  Ảnh sắc nét
        - 800+:     Ảnh rất sắc (nhiều cạnh)
        
        Đặt 300 để loại ảnh mờ không đọc được QR/text.
        
    CÁCH TÍNH:
    ----------
    1. Chuyển ảnh sang grayscale
    2. Áp dụng Laplacian operator (phát hiện cạnh)
    3. Tính variance của kết quả
    4. Variance cao = nhiều cạnh = ảnh sắc nét
    
    TỐC ĐỘ: ~0.280 ms/ảnh (chậm nhất trong 4 filter)
    
    Returns:
        (passed, sharpness_value)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    passed = sharpness >= min_sharpness
    return passed, round(sharpness, 2)


# ==============================================================================
# FILTER 4: BRIGHTNESS (Độ sáng)
# ==============================================================================
def check_brightness(image, min_brightness=60, max_brightness=240):
    """
    Kiểm tra độ sáng bằng mean pixel value.
    
    THAM SỐ:
    --------
    - min_brightness: float = 60
        Giá trị trung bình pixel tối thiểu (0-255).
        
        Giá trị tham khảo:
        - 0:     Đen hoàn toàn
        - 60:    Tối (ngưỡng loại)
        - 128:   Trung bình
        - 240:   Quá sáng (ngưỡng loại)
        - 255:   Trắng hoàn toàn
        
    - max_brightness: float = 240
        Giá trị trung bình pixel tối đa.
        Ảnh quá sáng sẽ bị mất chi tiết (overexposed).
        
    CÁCH TÍNH:
    ----------
    1. Chuyển ảnh sang grayscale
    2. Tính mean của tất cả pixels
    3. Kiểm tra trong khoảng [min, max]
    
    TỐC ĐỘ: ~0.037 ms/ảnh
    
    Returns:
        (passed, brightness_value, reason)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    
    if brightness < min_brightness:
        return False, round(brightness, 2), "too_dark"
    elif brightness > max_brightness:
        return False, round(brightness, 2), "too_bright"
    else:
        return True, round(brightness, 2), "ok"


# ==============================================================================
# HÀM TỔNG HỢP - KIỂM TRA TẤT CẢ 4 TIÊU CHÍ
# ==============================================================================
def check_all_quality(image, 
                       min_width=270, min_height=180,
                       min_contrast=30,
                       min_sharpness=300,
                       min_brightness=60, max_brightness=240):
    """
    Kiểm tra tất cả 4 tiêu chí chất lượng.
    
    Ảnh phải đạt TẤT CẢ 4 tiêu chí mới được coi là PASS.
    
    Returns:
        {
            'passed': bool,
            'fail_reason': str hoặc None,
            'metrics': {
                'width': int,
                'height': int,
                'contrast': float,
                'sharpness': float,
                'brightness': float
            }
        }
    """
    # 1. Check Size
    size_ok, width, height = check_size(image, min_width, min_height)
    if not size_ok:
        return {
            'passed': False,
            'fail_reason': f"Size too small ({width}x{height} < {min_width}x{min_height})",
            'metrics': {'width': width, 'height': height}
        }
    
    # 2. Check Contrast
    contrast_ok, contrast = check_contrast(image, min_contrast)
    if not contrast_ok:
        return {
            'passed': False,
            'fail_reason': f"Contrast too low ({contrast} < {min_contrast})",
            'metrics': {'width': width, 'height': height, 'contrast': contrast}
        }
    
    # 3. Check Sharpness
    sharpness_ok, sharpness = check_sharpness(image, min_sharpness)
    if not sharpness_ok:
        return {
            'passed': False,
            'fail_reason': f"Image blurry ({sharpness} < {min_sharpness})",
            'metrics': {'width': width, 'height': height, 'contrast': contrast, 'sharpness': sharpness}
        }
    
    # 4. Check Brightness
    brightness_ok, brightness, reason = check_brightness(image, min_brightness, max_brightness)
    if not brightness_ok:
        if reason == "too_dark":
            fail_reason = f"Image too dark ({brightness} < {min_brightness})"
        else:
            fail_reason = f"Image too bright ({brightness} > {max_brightness})"
        return {
            'passed': False,
            'fail_reason': fail_reason,
            'metrics': {
                'width': width, 'height': height,
                'contrast': contrast, 'sharpness': sharpness, 'brightness': brightness
            }
        }
    
    # ALL PASSED
    return {
        'passed': True,
        'fail_reason': None,
        'metrics': {
            'width': width, 'height': height,
            'contrast': contrast, 'sharpness': sharpness, 'brightness': brightness
        }
    }


# ==============================================================================
# MAIN - TEST VỚI 1 ẢNH
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python quality_filters.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        sys.exit(1)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        sys.exit(1)
    
    print("=" * 70)
    print(f"🔍 KIỂM TRA CHẤT LƯỢNG ẢNH: {os.path.basename(image_path)}")
    print("=" * 70)
    
    # Test từng filter
    size_ok, w, h = check_size(image)
    print(f"\n1. SIZE:       {'✅ PASS' if size_ok else '❌ FAIL'} ({w}x{h})")
    
    contrast_ok, contrast = check_contrast(image)
    print(f"2. CONTRAST:   {'✅ PASS' if contrast_ok else '❌ FAIL'} (std={contrast})")
    
    sharpness_ok, sharpness = check_sharpness(image)
    print(f"3. SHARPNESS:  {'✅ PASS' if sharpness_ok else '❌ FAIL'} (variance={sharpness})")
    
    brightness_ok, brightness, reason = check_brightness(image)
    print(f"4. BRIGHTNESS: {'✅ PASS' if brightness_ok else '❌ FAIL'} (mean={brightness})")
    
    # Kết quả tổng hợp
    result = check_all_quality(image)
    print("\n" + "-" * 70)
    if result['passed']:
        print("🎉 KẾT QUẢ: ✅ PASS - Ảnh đạt chất lượng!")
    else:
        print(f"❌ KẾT QUẢ: FAIL - {result['fail_reason']}")
    print("=" * 70)
