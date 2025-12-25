import cv2
import numpy as np
import os
import time

folder = 'output/debug/s3_preprocessing'
images = sorted([f for f in os.listdir(folder) if f.endswith('.png')])[:100]

# Tích lũy thời gian từng bước
size_times = []
contrast_times = []
sharpness_times = []
brightness_times = []
grayscale_times = []

for img_name in images:
    img_path = os.path.join(folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue
    
    # 1. Size check
    start = time.time()
    h, w = image.shape[:2]
    size_times.append((time.time() - start) * 1000)
    
    # 2. Grayscale conversion (dùng chung cho 3 bước sau)
    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_times.append((time.time() - start) * 1000)
    
    # 3. Contrast (std deviation)
    start = time.time()
    contrast = float(np.std(gray))
    contrast_times.append((time.time() - start) * 1000)
    
    # 4. Brightness (mean)
    start = time.time()
    brightness = float(np.mean(gray))
    brightness_times.append((time.time() - start) * 1000)
    
    # 5. Sharpness (Laplacian variance)
    start = time.time()
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    sharpness_times.append((time.time() - start) * 1000)

print('=' * 60)
print('⏱️ THỜI GIAN TỪNG BƯỚC LỌC (100 ảnh)')
print('=' * 60)
print()
print(f'1. Size check:        Mean = {np.mean(size_times):.4f} ms')
print(f'2. Grayscale convert: Mean = {np.mean(grayscale_times):.4f} ms')
print(f'3. Contrast (std):    Mean = {np.mean(contrast_times):.4f} ms')
print(f'4. Brightness (mean): Mean = {np.mean(brightness_times):.4f} ms')
print(f'5. Sharpness (Laplacian): Mean = {np.mean(sharpness_times):.4f} ms')
print()
print('-' * 60)
total = np.mean(size_times) + np.mean(grayscale_times) + np.mean(contrast_times) + np.mean(brightness_times) + np.mean(sharpness_times)
print(f'   TỔNG CỘNG:         Mean = {total:.4f} ms')
print('=' * 60)
