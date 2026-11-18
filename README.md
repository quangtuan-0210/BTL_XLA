# BTL Xử lý ảnh – Ứng dụng phát hiện và đếm đối tượng trong ảnh

Đếm đồng xu (và vật thể tách biệt trên nền tương đối đồng nhất) bằng các thuật toán
xử lý ảnh cổ điển **tự cài đặt bằng NumPy** (không dùng hàm xử lý ảnh của OpenCV).
OpenCV chỉ dùng để so sánh (`baseline_opencv/`).

```
ảnh màu → tiền xử lý → tách nền (Otsu) → hình thái học → tách vật dính (watershed) → gán nhãn BFS → đếm, đo
```

## Cấu trúc
| Thư mục | Nội dung | Ai làm |
|---|---|---|
| `src/preprocess.py`, `segment.py`, `morphology.py`, `separate.py`, `pipeline.py` | Thuật toán tiền xử lý → tách vật dính | TV1 |
| `src/count.py`, `measure.py`, `visualize.py`, `evaluate.py` | Gán nhãn BFS, đếm, đo từng vật, vẽ kết quả, đánh giá | TV2 |
| `app.py` | Giao diện Streamlit (chọn ảnh, chỉnh tham số, xem từng bước, bảng thông tin vật) | TV3 |
| `src/denomination.py` *(chưa có)* | Đoán mệnh giá, tính tổng tiền | TV3 |
| `data/` | 90 ảnh xu + `labels.csv`; 40 ảnh tế bào BBBC004 – xem `data/README_data.md` | TV3 |
| `data_tools/` | Script tạo bộ dữ liệu + bảng/biểu đồ phân tích dữ liệu | TV3 |
| `baseline_opencv/` | So sánh từng bước với OpenCV (`compare_tv1.py`, `compare_tv2.py`) + pipeline OpenCV đầy đủ (`pipeline_opencv.py`) | TV1, TV2 |
| `tests/` | Kiểm thử trên ảnh tổng hợp nhỏ: `test_tv1.py` (19 bài), `test_tv2.py` (27 bài) | TV1, TV2 |
| `results/tv1/`, `results/tv2/` | Bảng kết quả, biểu đồ, hình minh họa, phân tích thất bại | TV1, TV2 |

## Kết quả chính (tập kiểm tra, 72 ảnh – chạy 1 lần)
| Pipeline | Đếm đúng | MAE | Thời gian / ảnh |
|---|---|---|---|
| **Tự viết (chính)** | **51,4%** (Dễ 83% · Vừa 42% · Khó 29%) | **2,12** | 325 ms |
| Tự viết, bỏ watershed | 27,8% | 3,44 | 75 ms |
| OpenCV, cùng thiết kế | 59,7% | 2,00 | 9 ms |

Tập chỉnh (18 ảnh, dùng chọn tham số): 77,8%, MAE 0,44. Phân tích và phân tích thất bại: `README_TV2.md`.

## Cài đặt và chạy
Chạy mọi lệnh từ thư mục gốc dự án (các script đọc `data/...` theo đường dẫn tương đối).
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Lệnh | Việc làm | Ghi file |
|---|---|---|
| `python tests/test_tv1.py` | Kiểm thử phần TV1 (~1 giây) | không |
| `python tests/test_tv2.py` | Kiểm thử phần TV2 (~1 giây) | không |
| `python baseline_opencv/compare_tv1.py` | So từng bước TV1 với OpenCV | `results/tv1/so_sanh_opencv.csv` |
| `python baseline_opencv/compare_tv2.py` | So gán nhãn + đo với OpenCV | `results/tv2/so_sanh_opencv_tv2.csv` |
| `streamlit run app.py` | Giao diện, trình duyệt mở http://localhost:8501 | không |

Các lệnh có ghi file sẽ ghi đè kết quả đã commit (cột thời gian luôn thay đổi theo máy).
Muốn trả về bản cũ: `git restore results/`.

**zsh:** đừng dán chú thích `# ...` cùng dòng với lệnh – zsh mặc định không coi `#` là chú thích
(lỗi `not enough directory stack entries`). Hoặc thêm `setopt interactivecomments` vào `~/.zshrc`.

Dùng trong code:
```python
from src.pipeline import run
kq = run(anh_rgb)
kq["count"]      # số vật
kq["objects"]    # list[dict] thông số từng vật (pixel ảnh gốc): dien_tich, tam_x, tam_y, rong, cao, duong_kinh...
kq["binary"], kq["cleaned"], kq["separated"], kq["labels"]   # ảnh từng bước
```

## Quy ước dữ liệu
- `data/labels.csv`: cột `tap` = `chinh` (18 ảnh, được dùng để chỉnh tham số) / `kiem_tra` (72 ảnh, **chỉ chạy 1 lần cuối** để lấy số liệu báo cáo).
  Tập kiểm tra đã chạy ngày 17/11/2026 (`results/tv2/*_kiem_tra.*`).
- Bộ Kaggle gốc (1.444 ảnh, ~115 MB) không đưa lên repo. Chỉ cần nếu muốn chạy lại `data_tools/step1..5`:
  tải "Count Coins Image Dataset" trên Kaggle, giải nén vào `archive/`.

Chi tiết từng phần: `README_TV1.md`, `README_TV2.md`.
