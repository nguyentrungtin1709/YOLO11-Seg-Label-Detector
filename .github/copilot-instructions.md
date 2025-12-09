# Hướng dẫn cho GitHub Copilot

Bạn là một trợ lý lập trình AI chuyên gia. Khi tạo hoặc refactor code cho dự án này, bạn BẮT BUỘC phải tuân thủ các quy tắc dưới đây.

## 2. Nguyên tắc thiết kế SOLID (BẮT BUỘC)
Mọi đoạn mã hướng đối tượng (OOP) được tạo ra phải tuân thủ nghiêm ngặt 5 nguyên tắc SOLID:

1.  **S - Single Responsibility Principle (SRP)**:
    - Mỗi class chỉ được có một lý do duy nhất để thay đổi.
    - Tách biệt logic xử lý UI (Form), logic nghiệp vụ (Processing), và logic dữ liệu (File I/O, Database).
    - Ví dụ: Không viết code gọi API OCR trực tiếp trong sự kiện `Button_Click` của Form. Hãy tách nó ra một class `Service`.

2.  **O - Open/Closed Principle (OCP)**:
    - Các thực thể phần mềm (class, module) nên mở để mở rộng nhưng đóng để sửa đổi.
    - Sử dụng `interface` hoặc `abstract class` để định nghĩa các hành vi chung (ví dụ: `IEngine`), cho phép thêm engine mới mà không sửa code cũ.

3.  **L - Liskov Substitution Principle (LSP)**:
    - Các object của class con phải có thể thay thế class cha mà không làm sai lệch tính đúng đắn của chương trình.
    - Đảm bảo các class con thực thi đúng hợp đồng (contract) của interface/class cha.

4.  **I - Interface Segregation Principle (ISP)**:
    - Không ép buộc client phụ thuộc vào interface mà họ không sử dụng.
    - Chia nhỏ các interface lớn thành các interface nhỏ, chuyên biệt (ví dụ: tách `IImageLoader` và `IProcessor` thay vì gộp chung).

5.  **D - Dependency Inversion Principle (DIP)**:
    - Module cấp cao không nên phụ thuộc vào module cấp thấp. Cả hai nên phụ thuộc vào abstraction.
    - Sử dụng Dependency Injection (DI) để tiêm các dependency vào class thay vì khởi tạo trực tiếp (new) bên trong class.

## 3. Chiến lược phát triển ML Pipeline (Modular & Configurable)
Khi làm việc với các phần liên quan đến xử lý ảnh và ML pipeline:

1.  **Tính Mô-đun (Modularity)**:
    - Thiết kế pipeline thành các bước độc lập: `Preprocessing` -> `Inference` -> `Postprocessing`.
    - Mỗi bước phải là một "hộp đen" (black box) với Input/Output rõ ràng.
    - Ví dụ: Class `ImageCropper` chỉ nhận ảnh gốc + tọa độ và trả về ảnh crop, không quan tâm ảnh đó đến từ đâu.

2.  **Tính Cấu hình (Configurability)**:
    - TUYỆT ĐỐI KHÔNG hard-code các tham số (ngưỡng threshold, đường dẫn model, màu sắc, kích thước).
    - Tất cả tham số phải được đọc từ file cấu hình (ví dụ: `config.json` hoặc class `AppConfig`).
    - Các class xử lý nên nhận tham số cấu hình thông qua Constructor hoặc Method injection.

## 4. Quy tắc viết mã (Code Style)
- **Naming**: `PascalCase` cho Class. `camelCase` cho biến cục bộ/tham số/Method/Property.
- **Comments**: Giải thích logic phức tạp bằng tiếng Anh.
- **Error Handling**: Luôn sử dụng `try-catch` khi thao tác với File, Network hoặc External Process (ML inference).

## 5. Quy trình làm việc
- Trước khi viết code, hãy phân tích yêu cầu xem có vi phạm SOLID không.
- Nếu code hiện tại vi phạm, hãy đề xuất Refactor trước khi thêm tính năng mới.