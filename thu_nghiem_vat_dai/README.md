# Thử nghiệm: đếm vật dài (viên thuốc con nhộng)

Nhánh `cai-tien-vat-dai`, làm **sau khi đã nộp báo cáo**. Nhánh `main` (bản nộp) không đổi.

## Vấn đề
Bước 6–7 gốc lấy hạt giống là **đỉnh** của bản đồ khoảng cách (mỗi vật một tâm).
Viên thuốc dài có cả một sống núi dọc thân → bị cắt đôi; hai viên nằm sát theo chiều dài
không có chỗ eo → bị gộp. Chỉ dựa vào hình dạng khối trắng là không đủ.

## Cách làm thử (`src/tach_vat_dai_b6_7.py`)
1. Độ lớn gradient **Sobel** trên 3 kênh màu (Chương 6, phát hiện biên).
2. Hạt giống = **lõi trơn** của từng viên: điểm có gradient thấp, co nhẹ, gán nhãn.
3. **Watershed trên ảnh gradient** (như slide Chương 6): điểm gradient thấp loang trước.
4. **Gộp hai nửa** của cùng một viên (vết nối viên 2 màu, vệt bóng) bằng mô-men bậc 2
   (Chương 7): vùng gộp vẫn đầy, không quá dài, không rộng hơn thì gộp.

Bật bằng `run(img, loai_vat="dai")`, hoặc chọn "Loại vật: Dài, bề mặt trơn" trên giao diện.
Mặc định vẫn là `loai_vat="tron"` (thuật toán gốc).

## Dữ liệu thử
Ảnh **tổng hợp** do `sinh_anh_thuoc.py` tạo (biết trước số viên): viên 1 hoặc 2 màu, có bóng
khối trụ, vệt sáng, bóng đổ, nhiễu; 4–10 viên/ảnh, có viên chạm hoặc chồng nhẹ.
Tập chỉnh 10 ảnh (seed 1) để chọn tham số, tập kiểm tra 30 ảnh (seed 2) chỉ chạy để báo cáo.

## Kết quả

| | Bản gốc (`tron`) | Chế độ mới (`dai`) |
|---|---|---|
| Tập kiểm tra tổng hợp (30 ảnh): đếm đúng | 3/30 (10,0%) | **16/30 (53,3%)** |
| MAE (viên/ảnh) | 4,43 | **0,47** |
| Lệch tối đa ±1 viên | 7/30 | **30/30** |
| Đếm thiếu / thừa (số ảnh) | 0 / 27 | 13 / 1 |
| Ảnh chụp thật 8 viên | 9 (cắt đôi + gộp) | 7 (1 vùng gộp 3 viên, 1 viên bị cắt dọc) |
| Đồng xu, tập chỉnh 18 ảnh | 14/18 | 1/18 → **không dùng cho đồng xu** |

## Chế độ tự chọn (`loai_vat="tu_dong"`)
Sau bước 5, chỉ xét các vùng đơn lẻ, gọn (độ đầy ≥ 0,9). Có ≥ 2 vùng như vậy và trung vị
dài/rộng ≥ 1,7 → chế độ vật dài; còn lại → gần tròn. Ngưỡng chọn trên tập chỉnh
(18 ảnh xu, 10 ảnh thuốc): xu đơn lẻ có dài/rộng ≤ 1,44, viên thuốc ≥ 1,90.

**Test hỗn hợp** (`danh_gia_hon_hop.py`): 72 ảnh xu kiểm tra + 30 ảnh thuốc kiểm tra.

| Cách | Đồng xu (72) | Viên thuốc (30) | Tổng (102) |
|---|---|---|---|
| Luôn gần tròn (bản nộp) | 37 (51,4%), MAE 2,12 | 3 (10,0%), MAE 4,43 | 40 (39,2%), MAE 2,80 |
| Luôn vật dài | 9 (12,5%), MAE 3,71 | 16 (53,3%), MAE 0,47 | 25 (24,5%), MAE 2,75 |
| **Tự chọn** | **37 (51,4%)**, MAE 2,25 | 12 (40,0%), MAE 1,20 | **49 (48,0%)**, MAE 1,94 |

Máy chọn đúng chế độ: 70/72 ảnh xu, 23/30 ảnh thuốc. Nếu chọn đúng hết thì tổng tối đa
53/102 (52%). Ảnh thuốc bị chọn nhầm là ảnh có nhiều viên dính nhau, còn quá ít viên nằm riêng
để nhận ra hình dạng dài (ảnh chụp thật 8 viên cũng bị chọn nhầm vì lý do này).

## Nhận xét
- Trên ảnh tổng hợp, lỗi cắt đôi gần như hết (bản gốc thừa ở 27/30 ảnh). Lỗi còn lại chủ yếu
  là **gộp nhầm**: bước gộp hai nửa đôi khi gộp hai viên chạm đầu nhau.
- Trên ảnh thật duy nhất có trong tay, kết quả chưa tốt: ảnh thật khác ảnh tổng hợp
  (bóng, phản quang, chữ in trên viên). Cần **bộ ảnh chụp thật có đáp án** để chọn lại
  tham số và đánh giá đúng.
- Mặt đồng xu có hoa văn nổi → gradient lớn khắp nơi → cắt vụn. Vì vậy đây là một chế độ
  riêng, người dùng chọn theo loại vật.

## Chạy lại
```
python thu_nghiem_vat_dai/sinh_anh_thuoc.py thu_nghiem_vat_dai/anh/ 10 1
python thu_nghiem_vat_dai/sinh_anh_thuoc.py thu_nghiem_vat_dai/anh/ 30 2
python thu_nghiem_vat_dai/danh_gia.py thu_nghiem_vat_dai/anh/labels_2.json
```
