# Tài Liệu Cánh Tay Robot 4-DOF & Điều Khiển Arduino PS2

Kho tài liệu thiết kế cơ khí 3D và mã nguồn lập trình điều khiển cánh tay robot (Robotic Arm).

---

## 📁 Cấu Trúc Thư Mục

### 1. ile_in_3d cánh tay robot 4dof
Chứa toàn bộ bản vẽ thiết kế 3D cánh tay robot 4 bậc tự do (4-DOF):
- **STL Files**: Các file định dạng .STL đã được tối ưu, sẵn sàng nạp vào phần mềm cắt lớp (Cura, PrusaSlicer, Bambu Studio) để in 3D.
- **CAD & STEP**:
  - Complete Robotic Arm Assembly...STEP: File lắp ráp hoàn chỉnh chuẩn quốc tế STEP.
  - File thiết kế gốc SolidWorks (.SLDPRT, .SLDASM).
- **Tài liệu linh kiện (BOM)**:
  - Danh sách ốc vít, bu lông, vòng bi (51105, F624zz, F686zz).
  - Động cơ bước NEMA 17 (17HS4401) và puly/dây đai GT2-20T.

### 2. cánh tay robot + de robot + ps2 (ZYARJX-1 robot arm)
Chứa tài liệu kỹ thuật, sơ đồ mạch và mã nguồn:
- **3.code**: Mã nguồn Arduino (PS2X_ArmCar_Shield.ino) và thư viện điều khiển PS2X_lib.
- **5.Schematic diagram**: Sơ đồ đấu nối mạch điều khiển, shield động cơ và Arduino Uno.
- **7.PS2**: Hướng dẫn kết nối và cấu hình tay cầm không dây PS2.

---

> [!NOTE]
> Các file video hướng dẫn lắp ráp (.wmv) và bộ cài phần mềm nặng (~1.7 GB) đã được cấu hình loại trừ qua .gitignore để tối ưu dung lượng kho lưu trữ.