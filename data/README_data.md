# Dữ liệu – BTL Xử lý ảnh: Phát hiện và đếm đồng xu

## Nguồn
- **Count Coins Image Dataset** (Kaggle, tác giả balabaskar). Ảnh lấy từ Google Images, số xu đếm tay, đáp án trong `coins_count_values.csv`.
- Ghi rõ nguồn này trong báo cáo (mục 1.6 – minh bạch dữ liệu). Kiểm tra lại giấy phép trên trang Kaggle trước khi nộp.

## File
- `images/` – 90 ảnh bộ chính, tên dạng `<nuoc>_<ma>.jpg`
- `labels.csv` – đáp án, mỗi dòng một ảnh

| Cột | Ý nghĩa |
|---|---|
| file | tên ảnh trong `images/` |
| so_luong | số xu thật (đáp án) |
| muc_do | de / vua / kho |
| tap | `chinh` (20%, dùng chỉnh tham số) / `kiem_tra` (80%, chỉ chạy 1 lần để báo cáo) |
| nen_gradient, chenh_sang_goc, do_sang_tb | chỉ số dùng để gán mức độ |
| da_duyet, ghi_chu | Twin điền sau khi xem ảnh |

## Tiêu chí lọc
- Bỏ ảnh trùng nội dung (so mã MD5): 1444 → 1428
- Giữ ảnh có 2–30 xu (1 xu thì không phải bài toán đếm; >30 xu thường chồng lên nhau), cạnh ngắn ≥ 150 px: còn 942
- Chọn 30 ảnh/mức, trải đều 7 loại xu, seed = 42: 90 ảnh

## Tiêu chí mức độ (điểm khó 0–4)
+1 nếu nền có chi tiết (gradient viền > 5) · +1 nếu nền rất lộn xộn (> 9,5) · +1 nếu chênh sáng 4 góc > 35 · +1 nếu > 12 xu.
0 điểm = Dễ, 1 = Vừa, ≥ 2 = Khó.

## Chạy lại
```
python data_tools/step1_scan.py     # quét, bỏ trùng, đo chỉ số
python data_tools/step2_select.py   # lọc, gán mức độ, chia tập, copy ảnh
python data_tools/step3_analyze.py  # bảng + biểu đồ cho báo cáo
```
Chú ý: chạy lại step2 sẽ ghi đè `labels.csv` (mất phần đã duyệt tay).

## Bộ phụ: BBBC004 (ảnh tế bào tổng hợp)
- Nguồn: Broad Bioimage Benchmark Collection, https://bbbc.broadinstitute.org/BBBC004
- Trích dẫn đề nghị: Ruusuvuori et al., EUSIPCO 2008; Ljosa et al., Nature Methods 2012.
- `bbbc004/images/` – 40 ảnh 950×950: `000_xx` (0% dính nhau) và `045_xx` (45% dính nhau), 20 ảnh mỗi mức
- `bbbc004/masks/` – mặt nạ đáp án tách nền (trắng = tế bào), dùng để chấm bước phân ngưỡng (IoU)
- `bbbc004/labels_bbbc004.csv` – mỗi ảnh đúng 300 tế bào
- Dùng để: (1) thử thuật toán trên vật khác đồng xu, (2) đo hiệu quả bước tách vật dính (so 0% với 45%)

## Duyệt bằng mắt (05/11/2026)
- Bỏ 26 ảnh không phù hợp (trang web, poster, ảnh ghép, xu bị cắt mép, xu xếp chồng, ảnh trùng) → chuyển sang `data/_bo/`, không xóa.
- Sửa mức độ theo tiêu chí nhìn bằng mắt: **Dễ** = xu rời nhau, nền trơn · **Vừa** = xu chạm nhau hoặc nền màu/có bóng · **Khó** = xu chồng nhau hoặc nền có vân (gỗ, vải, giấy).
- Thêm 26 ảnh bù từ kho còn lại (đã bỏ ảnh gần-trùng), để giữ 30 ảnh/mức.
- Lý do từng quyết định: `data_tools/out/review_log.csv`. Bản trước khi duyệt: `data_tools/out/labels_truoc_duyet.csv`.
- **Không chạy lại `step2_select.py`** nữa (sẽ ghi đè kết quả duyệt).
