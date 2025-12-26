# Samples - Dữ liệu đầu vào

Thư mục này chứa dữ liệu ảnh đầu vào cho script `detection.py`.

## Mục đích

- Lưu trữ ảnh nhãn sản phẩm để xử lý batch qua pipeline phát hiện
- Hỗ trợ cấu trúc thư mục con để tổ chức theo bộ dữ liệu (set1, set2, ...)

## Cấu trúc

```
samples/
├── README.md           # File này
├── set1/               # Bộ dữ liệu 1
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── set2/               # Bộ dữ liệu 2
│   └── ...
└── ...
```

## Định dạng hỗ trợ

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`
- `.webp`

## Cách sử dụng

```bash
# Xử lý tất cả ảnh trong samples/
python scripts/detection.py --input samples --debug

# Xử lý một bộ dữ liệu cụ thể
python scripts/detection.py --input samples/set1 --debug
```

## Lưu ý

- Thư mục `samples/` và nội dung bên trong được ignore bởi git (trừ file README.md này)
- Dữ liệu ảnh không được commit vào repository
- Script `detection.py` sẽ quét đệ quy tất cả thư mục con
