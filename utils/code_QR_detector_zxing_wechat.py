import sys
import os
import cv2
import numpy as np
import time
import zxingcpp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QLabel, QFileDialog, QTextEdit, QProgressBar, 
                             QDialog, QTabWidget, QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- WORKER XỬ LÝ ẢNH ---
class BatchScanner(QThread):
    progress_sig = pyqtSignal(int)
    result_sig = pyqtSignal(dict)
    log_sig = pyqtSignal(str)

    def __init__(self, mode, detector, folder_path):
        super().__init__()
        self.mode = mode 
        self.detector = detector
        self.folder_path = folder_path

    # --- CÁC BỘ LỌC ---
    def apply_enhancement(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def apply_binary_processing(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def apply_invert(self, img_bgr):
        return cv2.bitwise_not(img_bgr)

    def apply_denoise(self, img_bgr):
        return cv2.medianBlur(img_bgr, 3)

    def apply_morphology(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

    def run(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_extensions)]
        total = len(files)
        times, success = [], 0
        failed_files = [] 
        
        # [THỐNG KÊ 1] Đếm bộ lọc
        filter_counts = {
            "Raw": 0, "Enhance": 0, "Binary": 0, 
            "Invert": 0, "Denoise": 0, "Morph": 0
        }

        # [THỐNG KÊ 2] Đếm Scale (Scale nào ăn được bao nhiêu ảnh)
        # Khởi tạo dict cho các scale
        scales_check_list = [
            1.0,                            # Gốc
            1.25, 1.5, 1.75, 2.0, 2.25, 2.75, 3.0,  # Phóng to
            0.75, 0.5, 0.25                 # Thu nhỏ
        ]
        scale_counts = {s: 0 for s in scales_check_list}

        for i, filename in enumerate(files):
            full_path = os.path.join(self.folder_path, filename)
            img_orig = cv2.imread(full_path)
            if img_orig is None: continue
            
            start = time.perf_counter()
            found = False
            strategy = "Fail"
            winning_filter = None 
            winning_scale = None
            
            # --- VÒNG LẶP QUÉT SCALE ---
            for s in scales_check_list:
                if s == 1.0: 
                    img_resized = img_orig
                else:
                    h, w = img_orig.shape[:2]
                    method = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
                    img_resized = cv2.resize(img_orig, (int(w*s), int(h*s)), interpolation=method)
                
                # --- VÒNG LẶP QUÉT FILTER ---
                processors = [
                    ("Raw", lambda x: x),
                    ("Enhance", self.apply_enhancement),
                    ("Binary", self.apply_binary_processing),
                    ("Invert", self.apply_invert),
                    ("Denoise", self.apply_denoise),
                    ("Morph", self.apply_morphology)
                ]

                for proc_name, proc_func in processors:
                    try:
                        test_img = proc_func(img_resized)
                        
                        if self.mode == 'wechat':
                            txt, _ = self.detector.detectAndDecode(test_img)
                            if txt and txt[0]: found = True
                        else:
                            res = zxingcpp.read_barcodes(test_img, formats=zxingcpp.BarcodeFormat.QRCode)
                            if any(r.valid for r in res): found = True
                        
                        if found:
                            strategy = f"Zoom {s}x ({proc_name})"
                            winning_filter = proc_name 
                            winning_scale = s
                            break
                    except Exception: 
                        pass
                
                if found: break 

            duration = (time.perf_counter() - start) * 1000
            times.append(duration)
            
            if found: 
                success += 1
                # Cộng dồn thống kê
                if winning_filter in filter_counts:
                    filter_counts[winning_filter] += 1
                if winning_scale in scale_counts:
                    scale_counts[winning_scale] += 1
                
                log_msg = f"[{self.mode.upper()}] {filename} - OK | {strategy} | {duration:.1f}ms"
            else:
                failed_files.append(full_path)
                log_msg = f"[{self.mode.upper()}] {filename} - FAIL | {duration:.1f}ms"
            
            self.log_sig.emit(log_msg)
            self.progress_sig.emit(int((i+1)/total * 100))

        self.result_sig.emit({
            "mode": self.mode, "total": total, "success": success,
            "times": times, "avg": np.mean(times) if times else 0, "std": np.std(times) if times else 0,
            "failed_files": failed_files,
            "filter_counts": filter_counts,
            "scale_counts": scale_counts # [MỚI] Gửi data scale
        })

# --- UI REPORT DIALOG ---
class ReportDialog(QDialog):
    def __init__(self, data_wechat, data_zxing):
        super().__init__()
        self.setWindowTitle("Báo cáo phân tích so sánh hệ thống QR")
        self.resize(1100, 850)
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.tabs.addTab(self.create_bar_tab(data_wechat, data_zxing), "Tổng quan")
        self.tabs.addTab(self.create_filter_tab(data_wechat, data_zxing), "Thống kê Bộ lọc")
        self.tabs.addTab(self.create_scale_tab(data_wechat, data_zxing), "Thống kê Scale") # [MỚI]
        self.tabs.addTab(self.create_time_tab(data_wechat, data_zxing), "Thời gian")
        self.tabs.addTab(self.create_table_tab(data_wechat, data_zxing), "Bảng Chi tiết")
        self.tabs.addTab(self.create_failed_gallery_tab(data_wechat), "Ảnh lỗi (WeChat)")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    # [TAB MỚI] Biểu đồ Scale
    def create_scale_tab(self, wc, zx):
        fig, ax = plt.subplots()
        
        # Lấy danh sách scale và sắp xếp lại cho đẹp (0.25 -> 3.0)
        # Key trong dict là float, ta cần sort
        all_scales = sorted([
             1.0, 
            1.25, 1.5, 1.75, 2.0, 2.25, 2.75, 3.0
        ])
        
        labels = [f"{s}x" for s in all_scales]
        wc_vals = [wc.get('scale_counts', {}).get(s, 0) for s in all_scales]
        zx_vals = [zx.get('scale_counts', {}).get(s, 0) for s in all_scales]

        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, wc_vals, width, label='WeChat', color='#4CAF50')
        rects2 = ax.bar(x + width/2, zx_vals, width, label='ZXing', color='#2196F3')

        ax.set_ylabel('Số lần thành công')
        ax.set_title('Mức độ Zoom (Scale) hiệu quả nhất')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        
        # Thêm số liệu lên đỉnh cột
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        
        fig.tight_layout()
        return FigureCanvas(fig)

    def create_filter_tab(self, wc, zx):
        fig, ax = plt.subplots()
        filters = ["Raw", "Enhance", "Binary", "Invert", "Denoise", "Morph"]
        wc_counts = [wc.get('filter_counts', {}).get(f, 0) for f in filters]
        zx_counts = [zx.get('filter_counts', {}).get(f, 0) for f in filters]

        x = np.arange(len(filters))
        width = 0.35

        rects1 = ax.bar(x - width/2, wc_counts, width, label='WeChat', color='#4CAF50')
        rects2 = ax.bar(x + width/2, zx_counts, width, label='ZXing', color='#2196F3')

        ax.set_ylabel('Số lần thành công')
        ax.set_title('Hiệu quả của từng bộ lọc')
        ax.set_xticks(x)
        ax.set_xticklabels(filters)
        ax.legend()
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        return FigureCanvas(fig)

    def create_failed_gallery_tab(self, wc):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        grid_layout = QGridLayout(content_widget)
        
        failed_list = wc.get('failed_files', [])
        if not failed_list:
            grid_layout.addWidget(QLabel("Không có ảnh lỗi."), 0, 0)
        else:
            cols = 4 
            for index, img_path in enumerate(failed_list):
                cell = QWidget()
                cell_layout = QVBoxLayout(cell)
                img_label = QLabel()
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    img_label.setPixmap(pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    img_label.setAlignment(Qt.AlignCenter)
                name_label = QLabel(os.path.basename(img_path))
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setStyleSheet("font-size: 10px; color: gray;")
                cell_layout.addWidget(img_label)
                cell_layout.addWidget(name_label)
                grid_layout.addWidget(cell, index // cols, index % cols)

        scroll.setWidget(content_widget)
        return scroll

    def create_bar_tab(self, wc, zx):
        fig, ax = plt.subplots()
        ax.bar(['WeChat QR', 'ZXing'], [wc['success'], zx['success']], color=['#4CAF50', '#2196F3'])
        ax.set_title("Tổng số mẫu nhận diện thành công")
        for i, v in enumerate([wc['success'], zx['success']]):
            ax.text(i, v, str(v), ha='center', fontweight='bold', va='bottom')
        return FigureCanvas(fig)

    def create_time_tab(self, wc, zx):
        fig, ax = plt.subplots()
        if wc['times']: ax.plot(wc['times'], label='WeChat', color='#4CAF50')
        if zx['times']: ax.plot(zx['times'], label='ZXing', color='#2196F3')
        ax.legend(); ax.set_title("Thời gian xử lý (ms)")
        return FigureCanvas(fig)

    def create_table_tab(self, wc, zx):
        fig, ax = plt.subplots()
        ax.axis('off')
        wc_succ = f"{wc['success']}/{wc['total']}"
        zx_succ = f"{zx['success']}/{zx['total']}"
        table_data = [
            ["Chỉ số", "WeChat", "ZXing"],
            ["Thành công", wc_succ, zx_succ],
            ["Trung bình", f"{wc['avg']:.2f} ms", f"{zx['avg']:.2f} ms"],
        ]
        ax.table(cellText=table_data, loc='center', cellLoc='center').scale(1, 2)
        return FigureCanvas(fig)

# --- MAIN APP ---
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Benchmark QR Pro - Filter & Scale Stats")
        self.setGeometry(100, 100, 700, 600)
        self.results = {}
        self.init_detector()
        self.init_ui()

    def init_detector(self):
        model_dir = "model" 
        try:
            self.wechat = cv2.wechat_qrcode.WeChatQRCode(
                os.path.join(model_dir, "detect.prototxt"), 
                os.path.join(model_dir, "detect.caffemodel"), 
                os.path.join(model_dir, "sr.prototxt"), 
                os.path.join(model_dir, "sr.caffemodel")
            )
            print("Đã load model WeChat QR")
        except:
            print("CẢNH BÁO: Không tìm thấy model")
            self.wechat = None

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        self.btn_wc = QPushButton("Chạy WeChat QR")
        self.btn_zx = QPushButton("Chạy ZXing-cpp")
        self.btn_report = QPushButton("XEM BÁO CÁO")
        self.btn_report.setEnabled(False)
        self.log = QTextEdit()
        self.pbar = QProgressBar()
        
        self.btn_wc.clicked.connect(lambda: self.start_scan('wechat'))
        self.btn_zx.clicked.connect(lambda: self.start_scan('zxing'))
        self.btn_report.clicked.connect(self.show_report)
        
        layout.addWidget(self.btn_wc); layout.addWidget(self.btn_zx)
        layout.addWidget(self.pbar); layout.addWidget(self.log)
        layout.addWidget(self.btn_report)
        self.setCentralWidget(central)

    def start_scan(self, mode):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        if not folder: return
        self.log.clear()
        if mode == 'wechat' and not self.wechat:
            self.log.append("Lỗi: Chưa có Model WeChat.")
            return

        self.worker = BatchScanner(mode, self.wechat if mode == 'wechat' else None, folder)
        self.worker.log_sig.connect(self.log.append)
        self.worker.progress_sig.connect(self.pbar.setValue)
        self.worker.result_sig.connect(self.save_result)
        self.worker.start()

    def save_result(self, res):
        self.results[res['mode']] = res
        self.btn_report.setEnabled(True)

    def show_report(self):
        dummy = {"success": 0, "total": 1, "times": [], "avg": 0, "std": 0, "failed_files": [], "filter_counts": {}, "scale_counts": {}}
        wc = self.results.get('wechat', dummy)
        zx = self.results.get('zxing', dummy)
        ReportDialog(wc, zx).exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainApp(); ex.show()
    sys.exit(app.exec_())