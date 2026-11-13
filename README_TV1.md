# Phần TV1 – Tiền xử lý → Tách nền → Hình thái học → Tách vật dính

Toàn bộ thuật toán tự viết bằng **NumPy** (không gọi hàm xử lý ảnh của OpenCV).
OpenCV chỉ dùng trong `baseline_opencv/` để **so sánh**.

## File
| File | Nội dung | Hàm chính |
|---|---|---|
| `src/preprocess.py` | Bước 1–3 | `to_gray`, `resize_max`, `conv2d`, `gaussian_kernel`, `gaussian_blur`, `median_filter`, `preprocess` |
| `src/segment.py` | Bước 4 | `otsu_threshold`, `segment` (ảnh xám), `color_difference` + `segment_color` (độ khác màu nền) |
| `src/morphology.py` | Bước 5 | `disk`, `erode`, `dilate`, `opening`, `closing`, `fill_holes`, `clean` |
| `src/separate.py` | Bước 6–7 | `distance_transform` (Chamfer 3-4), `find_markers`, `watershed`, `separate`, `label_bfs` |
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

Ảnh tế bào BBBC004 (đáp án 300/ảnh): IoU bước tách nền ≈ 0,88.
Đếm: rời nhau 277 → **300**; dính 45%: 186 → **250**.

So sánh với OpenCV: các bước 1–5 giống **100%**; biến đổi khoảng cách giống ~83%
(Chamfer là phép xấp xỉ khoảng cách Euclid); chậm hơn OpenCV 2–270 lần (OpenCV viết bằng C++).

## Hai cải tiến đã làm (nên kể khi thuyết trình)
1. **Tách nền theo "độ khác màu nền"** thay vì ảnh xám: màu nền = trung vị màu ở viền ảnh;
   mỗi điểm đo khoảng cách màu tới nền rồi mới Otsu. Giúp xu bạc trên nền trắng, nền màu (đỏ, xanh).
   Trên tập chỉnh: 56% → 78% đếm đúng.
2. **Đường cắt watershed dày 2 pixel**: nếu chỉ 1 pixel, bước đếm (8 lân cận) đi chéo qua đường cắt
   và gộp 2 xu lại.

## Ca thất bại (cho mục phân tích lỗi)
4/18 ảnh tập chỉnh sai, đều mức Khó: xu **chồng lên nhau** (không chỉ chạm). Khi chồng, vùng chung
không có "yên ngựa" rõ trong biến đổi khoảng cách → watershed thiếu hạt giống → đếm thiếu.

## Câu hỏi thầy có thể hỏi
- *Vì sao lật kernel trong `conv2d`?* Định nghĩa tích chập yêu cầu lật; không lật là tương quan (correlation). Với kernel Gauss đối xứng thì như nhau.
- *Otsu chọn ngưỡng thế nào?* Thử mọi ngưỡng 0–255, chọn ngưỡng làm phương sai GIỮA hai nhóm lớn nhất.
- *Mở khác đóng?* Mở = co rồi giãn (xóa chấm nhỏ); đóng = giãn rồi co (lấp khe hở).
- *Tại sao cần watershed?* Hai xu chạm nhau thành 1 vùng trắng; watershed dùng 2 "đỉnh" khoảng cách để cắt đôi.
- *Tham số quan trọng?* `peak_ratio=0.4` (cửa sổ tìm đỉnh), `min_ratio=0.2` (đỉnh phải đủ cao) – chọn trên tập chỉnh.
