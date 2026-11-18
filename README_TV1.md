# Phần TV1 – Tiền xử lý → Tách nền → Hình thái học → Tách vật dính

Toàn bộ thuật toán tự viết bằng **NumPy** (không gọi hàm xử lý ảnh của OpenCV).
OpenCV chỉ dùng trong `baseline_opencv/` để **so sánh**.

## File
| File | Nội dung | Hàm chính |
|---|---|---|
| `src/preprocess.py` | Bước 1–3 | `to_gray`, `resize_max`, `conv2d`, `gaussian_kernel`, `gaussian_blur`, `preprocess` |
| `src/segment.py` | Bước 4 | `otsu_threshold`, `segment` (ảnh xám), `color_difference` + `segment_color` (độ khác màu nền) |
| `src/morphology.py` | Bước 5 | `disk`, `erode`, `dilate`, `closing`, `fill_holes`, `clean` |
| `src/separate.py` | Bước 6–7 | `distance_transform` (co lặp, Chương 5), `find_markers`, `watershed`, `separate`, `label_bfs` |
| `src/pipeline.py` | Ghép các bước | `run(img)` → dict ảnh từng bước + `count` |
| `baseline_opencv/compare_tv1.py` | So sánh từng bước với OpenCV | → `results/tv1/so_sanh_opencv.csv` |
| `src/run_tv1_eval.py` | Đếm xu, có/không tách | → `results/tv1/xu_<tap>.csv` |
| `src/run_bbbc004.py` | Ảnh tế bào: IoU + đếm | → `results/tv1/bbbc004.csv` |
| `src/figure_steps.py`, `src/debug_sheet.py` | Vẽ hình minh họa / tìm lỗi | → `results/tv1/*.png` |

## Chạy
```
pip install numpy pillow pandas matplotlib opencv-python
python src/run_tv1_eval.py chinh          # đếm trên tập chỉnh
python baseline_opencv/compare_tv1.py     # so sánh với OpenCV
python src/run_bbbc004.py                 # ảnh tế bào
python src/figure_steps.py <ten_anh.jpg> ra.png
```
Dùng trong code khác:
```python
from src.pipeline import run
kq = run(anh_rgb)          # kq["count"], kq["binary"], kq["separated"], kq["labels"] ...
```

## Kết quả (tập chỉnh, 18 ảnh xu – KHÔNG dùng tập kiểm tra để chỉnh)
| Mức | Đếm đúng (không tách → có tách) | Sai số TB, MAE (không tách → có tách) |
|---|---|---|
| Dễ | 50% → 100% | 0,50 → 0,00 |
| Vừa | 17% → 100% | 3,17 → 0,00 |
| Khó | 0% → 33% | 4,83 → 1,33 |
| **Tổng** | **22% → 78%** | **2,83 → 0,44** |

Ảnh tế bào BBBC004 (đáp án 300/ảnh), cùng cách tách nền độ khác màu nền như ứng dụng: IoU bước tách nền ≈ 0,88.
Đếm: rời nhau 278 → **302**; dính 45%: 187 → **255**.

Ứng dụng chỉ có **một** cách tách nền (độ khác màu nền). Hàm `segment` (Otsu ảnh xám) chỉ còn dùng cho thí nghiệm so sánh của TV2.

So sánh với OpenCV: các bước xám hóa, Gauss, Otsu, co, giãn, đóng, watershed giống **100%**; biến đổi khoảng cách
giống ~80% (co lặp cho khoảng cách "bát giác", OpenCV tính khoảng cách Euclid); chậm hơn OpenCV 2–65 lần (OpenCV viết bằng C++).

## Biến đổi khoảng cách bằng phép co lặp (đổi ngày 18/11)
Bản trước dùng thuật toán Chamfer 3-4 (không có trong bài giảng). Nay thay bằng **phép co của Chương 5**:
co ảnh liên tục, khoảng cách của một điểm = số lần co mà điểm đó còn sống. Phần tử cấu trúc 3×3 dùng **xen kẽ**
chữ thập (+) và vuông → vật teo thành hình bát giác, gần tròn nhất. Chọn trên tập chỉnh (không nhìn tập kiểm tra):

| Phần tử cấu trúc khi co | Tập chỉnh (18 ảnh) |
|---|---|
| Chamfer 3-4 (bản cũ) | 77,8%, MAE 0,44 |
| Vuông 3×3 | 66,7%, MAE 0,72 |
| Chữ thập 3×3 | 72,2%, MAE 0,83 |
| **Xen kẽ chữ thập / vuông (dùng)** | **77,8%, MAE 0,44** |

Tập kiểm tra (chạy lại sau khi đổi): 52,8% → 51,4% (lệch 1 ảnh đúng), MAE 2,07 → 2,12.

## Thí nghiệm bỏ từng bước (120 ảnh Kaggle chưa dùng + 18 ảnh tập chỉnh, không dùng tập kiểm tra)
| Bỏ bước | Đúng (120 ảnh mới) | Đúng (18 ảnh chỉnh) |
|---|---|---|
| (đầy đủ) | 48,3% | 77,8% |
| Làm mịn Gauss | 50,8% | 61,1% |
| Phép đóng | 51,7% | 72,2% |
| Lấp lỗ | **22,5%** | **27,8%** |
| Watershed | **22,5%** | **22,2%** |
| Độ khác màu nền (dùng ảnh xám) | 48,3% | 50,0% |

*Chạy lại với biến đổi khoảng cách mới (18/11). Lọc trung vị và phép mở đã bỏ từ trước: thử trên bản cũ, bỏ hai bước này không đổi kết quả.*

→ Đã bỏ lọc trung vị và phép mở. Giữ Gauss và phép đóng: trên 120 ảnh mới bỏ đi thì nhỉnh hơn 2–3 điểm (trong mức ngẫu nhiên), nhưng trên tập chỉnh bỏ đi thì giảm rõ (Gauss 78% → 61%).

## Hai cải tiến đã làm (nên kể khi thuyết trình)
1. **Tách nền theo "độ khác màu nền"** thay vì ảnh xám: màu nền = trung vị màu ở viền ảnh;
   mỗi điểm đo khoảng cách màu tới nền rồi mới Otsu. Giúp xu bạc trên nền trắng, nền màu (đỏ, xanh).
   Trên tập chỉnh: 50% → 78% đếm đúng.
2. **Đường cắt watershed dày 2 pixel**: nếu chỉ 1 pixel, bước đếm (8 lân cận) đi chéo qua đường cắt
   và gộp 2 xu lại.

## Ca thất bại (cho mục phân tích lỗi)
4/18 ảnh tập chỉnh sai, đều mức Khó: xu **chồng lên nhau** (không chỉ chạm). Khi chồng, vùng chung
không có "yên ngựa" rõ trong biến đổi khoảng cách → watershed thiếu hạt giống → đếm thiếu.

## Câu hỏi thầy có thể hỏi
- *Vì sao lật kernel trong `conv2d`?* Định nghĩa tích chập yêu cầu lật; không lật là tương quan (correlation). Với kernel Gauss đối xứng thì như nhau.
- *Otsu chọn ngưỡng thế nào?* Thử mọi ngưỡng 0–255, chọn ngưỡng làm phương sai GIỮA hai nhóm lớn nhất.
- *Đóng là gì?* Giãn rồi co: lấp khe hở, nối nét đứt trên mặt xu.
- *Sao không dùng lọc trung vị và phép mở?* Đã thử bỏ từng bước trên 138 ảnh không thuộc tập kiểm tra:
  lọc trung vị không đổi kết quả (đang tắt), phép mở cũng không đổi vì bước đếm đã bỏ vùng < 5% vật lớn nhất.
  Bỏ đi cho pipeline gọn. Ngược lại, bỏ **lấp lỗ** hoặc **watershed** thì % đếm đúng giảm gần một nửa.
- *Biến đổi khoảng cách tính thế nào?* Co ảnh liên tục (Chương 5); điểm sống qua k lần co có khoảng cách k.
  Xen kẽ SE chữ thập và vuông để vật teo đều thành gần tròn (chỉ vuông → teo thành vuông, chỉ chữ thập → hình thoi).
- *Hạt giống watershed là gì?* Đỉnh của "ngọn đồi" khoảng cách: điểm lớn nhất trong cửa sổ quanh nó (lọc max),
  giống bước "loại bỏ điểm không cực đại" của Canny. Không có hạt giống thì vân trên mặt xu tạo nhiều thung lũng nhỏ
  và watershed cắt vụn đồng xu (phân đoạn quá mức).
- *Tại sao cần watershed?* Hai xu chạm nhau thành 1 vùng trắng; watershed dùng 2 "đỉnh" khoảng cách để cắt đôi.
- *Tham số quan trọng?* `peak_ratio=0.4` (cửa sổ tìm đỉnh), `min_ratio=0.2` (đỉnh phải đủ cao) – chọn trên tập chỉnh.
