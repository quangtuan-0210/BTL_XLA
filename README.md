# BTL Xử lý ảnh – Ứng dụng phát hiện và đếm đối tượng trong ảnh

Đếm đồng xu (và vật thể tách biệt trên nền tương đối đồng nhất) bằng các thuật toán
xử lý ảnh cổ điển **tự cài đặt bằng NumPy** (không dùng hàm xử lý ảnh của OpenCV).
OpenCV chỉ dùng để so sánh (`baseline_opencv/`).

```
ảnh màu → tiền xử lý → tách nền (Otsu) → hình thái học → tách vật dính (watershed) → đếm, đo
```

## Cấu trúc
| Thư mục | Nội dung | Ai làm |
|---|---|---|
| `src/preprocess.py`, `segment.py`, `morphology.py`, `separate.py`, `pipeline.py` | Thuật toán tiền xử lý → tách vật dính | TV1 |
| `src/count.py`, `measure.py`, `evaluate.py` *(chưa có)* | Đếm, đo, đánh giá chính thức | TV2 |
| `app.py` | Giao diện Streamlit (chọn ảnh, chỉnh tham số, xem từng bước, bảng thông tin vật) | TV3 |
| `src/denomination.py` *(chưa có)* | Đoán mệnh giá, tính tổng tiền | TV3 |
| `data/` | 90 ảnh xu + `labels.csv`; 40 ảnh tế bào BBBC004 – xem `data/README_data.md` | TV3 |
| `data_tools/` | Script tạo bộ dữ liệu + bảng/biểu đồ phân tích dữ liệu | TV3 |
| `baseline_opencv/` | So sánh từng bước với OpenCV | TV1 |
| `results/` | Bảng kết quả, hình minh họa | |

## Cài đặt và chạy
```bash
pip install -r requirements.txt
python src/run_tv1_eval.py chinh        # đếm thử trên tập chỉnh (18 ảnh)
python baseline_opencv/compare_tv1.py   # so sánh với OpenCV
```
Giao diện:
```bash
streamlit run app.py      # trình duyệt tự mở http://localhost:8501
```
Dùng trong code:
```python
from src.pipeline import run
kq = run(anh_rgb)   # kq["count"] = số vật; kq["binary"], kq["separated"], kq["labels"]: ảnh từng bước
```

## Quy ước dữ liệu
- `data/labels.csv`: cột `tap` = `chinh` (18 ảnh, được dùng để chỉnh tham số) / `kiem_tra` (72 ảnh, **chỉ chạy 1 lần cuối** để lấy số liệu báo cáo).
- Bộ Kaggle gốc (1.444 ảnh, ~115 MB) không đưa lên repo. Chỉ cần nếu muốn chạy lại `data_tools/step1..5`:
  tải "Count Coins Image Dataset" trên Kaggle, giải nén vào `archive/`.

Chi tiết phần TV1: `README_TV1.md`.
