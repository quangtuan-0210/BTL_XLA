# Phần TV2 – Gán nhãn → Đếm → Đo → Đánh giá

Toàn bộ thuật toán tự viết bằng **NumPy** (không gọi hàm xử lý ảnh của OpenCV).
OpenCV chỉ dùng trong `baseline_opencv/` để **so sánh**.

## File
| File | Nội dung | Hàm chính |
|---|---|---|
| `src/count.py` | Bước 1–2: gán nhãn liên thông bằng BFS (8 lân cận), lọc vùng nhiễu, đếm | `label_components`, `remove_small`, `count_objects` |
| `src/measure.py` | Bước 3: diện tích, tâm, khung bao, đường kính tương đương, tỉ lệ khung, độ đặc | `measure_objects`, `objects_table`, `suspicious` |
| `src/visualize.py` | Bước 4: tô màu, viền, đánh số từng vật lên ảnh | `draw_objects`, `caption`, `grid` |
| `src/evaluate.py` | Bước 5: % đếm đúng, MAE, sai lệch, theo mức Dễ / Vừa / Khó | `evaluate(pred_csv, label_csv)`, `result_table`, `errors` |
| `src/run_tv2_eval.py` | Bước 6: chạy 4 cấu hình pipeline trên một tập ảnh | → `results/tv2/bang_ket_qua_<tap>.csv`, `du_doan_<tap>.csv`, `vat_<tap>.csv` |
| `src/run_tv2_figures.py` | Biểu đồ + ảnh minh họa thành công / thất bại | → `results/tv2/hinh*.png`, `minh_hoa_*.png`, `that_bai_<tap>.csv` |
| `baseline_opencv/pipeline_opencv.py` | Pipeline đối chứng: cùng thiết kế, dùng hàm OpenCV | `run_opencv` |
| `baseline_opencv/compare_tv2.py` | So sánh gán nhãn + đo với OpenCV | → `results/tv2/so_sanh_opencv_tv2.csv` |
| `tests/test_tv2.py` | 27 bài test trên ảnh tổng hợp nhỏ (biết trước đáp án) | |
| `results/tv2/phan_tich_that_bai_kiem_tra.csv` | Bước 7: 10 ảnh sai nặng nhất + nguyên nhân + bằng chứng | |

`src/pipeline.py` đã dùng `count_objects` (thay bản đếm tạm `count_tmp`): `run(img)` trả thêm
`objects` (thông số từng vật, đổi về pixel ảnh gốc). `app.py` dùng `objects_table` và `draw_objects`.

## Chạy
```
python tests/test_tv2.py                    # kiểm thử nhanh, ~1 giây
python baseline_opencv/compare_tv2.py       # so gán nhãn + đo với OpenCV
python src/run_tv2_eval.py chinh            # đánh giá trên tập chỉnh (~20 giây)
python src/run_tv2_figures.py chinh         # biểu đồ + ảnh minh họa
```
Tập kiểm tra **đã chạy 1 lần** (17/11/2026) với tham số chốt trên tập chỉnh; kết quả nằm trong
`results/tv2/*_kiem_tra.*`. Không chạy lại `run_tv2_eval.py kiem_tra` trừ khi cả nhóm đổi tham số
(chọn lại trên tập chỉnh) và ghi rõ điều đó trong báo cáo.

Dùng trong code khác:
```python
from src.count import count_objects
n, objects = count_objects(anh_nhi_phan)                # objects: list[dict] - id, dien_tich, tam_x, tam_y,
                                                        # x, y, rong, cao, duong_kinh, ti_le_khung, do_dac
from src.evaluate import evaluate
kq = evaluate("results/tv2/pred_kiem_tra_tu_viet.csv", "data/labels.csv")
kq["tong"]["ti_le_dung"], kq["tong"]["mae"], kq["theo_muc"]["kho"]
```

## Thiết kế thí nghiệm
- **Dữ liệu:** 90 ảnh xu (TV3). Tập chỉnh 18 ảnh (20%) dùng để chọn tham số; tập kiểm tra 72 ảnh (80%,
  24 ảnh/mức) chỉ chạy một lần để lấy số liệu báo cáo. Thuật toán cổ điển, không huấn luyện.
- **Chỉ số** (mỗi ảnh: e = số đếm − số thật): % ảnh đếm đúng (e = 0), **MAE** = trung bình |e|,
  sai lệch TB = trung bình e (âm = đếm thiếu), sai số tương đối = trung bình |e| / số thật, số ảnh đếm thiếu / thừa.
- **4 cấu hình**, cùng tham số `DEFAULT` trong `src/pipeline.py` – mỗi cấu hình đổi đúng một yếu tố:

| Cấu hình | Khác pipeline chính ở | Trả lời câu hỏi |
|---|---|---|
| `tu_viet` | (pipeline chính: độ khác màu nền + watershed) | – |
| `tu_viet_khong_tach` | bỏ watershed | Bước tách vật dính đóng góp bao nhiêu? |
| `tu_viet_xam` | Otsu trên ảnh xám thay vì độ khác màu nền | Cải tiến tách nền của TV1 có giúp không? |
| `opencv` | các bước dùng hàm OpenCV | Cài đặt tự viết có kém thư viện không? |

## Kết quả định lượng – tập kiểm tra (72 ảnh)
| Cấu hình | Đếm đúng | MAE | Sai lệch TB | Sai số tương đối | Thời gian / ảnh |
|---|---|---|---|---|---|
| **Tự viết (chính)** | **51,4%** | **2,12** | −0,90 | 26,5% | 325 ms |
| Tự viết, không watershed | 27,8% | 3,44 | −3,33 | 43,9% | 75 ms |
| Tự viết, Otsu ảnh xám | 47,2% | 2,28 | −0,75 | 33,7% | 306 ms |
| OpenCV (cùng thiết kế) | 59,7% | 2,00 | −1,39 | 23,2% | 9 ms |

*Cập nhật 18/11 (lần 2): biến đổi khoảng cách đổi từ Chamfer 3-4 sang phép co lặp của Chương 5 (chọn trên tập chỉnh,
xem README_TV1) và chạy lại tập kiểm tra: 52,8% → 51,4%, MAE 2,07 → 2,12.*

*Cập nhật 18/11 (lần 1): pipeline bỏ lọc trung vị và phép mở (xem README_TV1, mục "Thí nghiệm bỏ từng bước" – quyết định
dựa trên 138 ảnh không thuộc tập kiểm tra). Bản trước có phép mở: tự viết 54,2% / MAE 2,04 (lệch 1 ảnh). Thời gian
đo trên máy khác lần trước (OpenCV 5.0) nên chỉ so tương đối giữa các cấu hình.*

Pipeline chính theo mức độ:
| Mức | Đếm đúng (không tách → có tách) | MAE (không tách → có tách) |
|---|---|---|
| Dễ | 70,8% → **83,3%** | 0,29 → 0,46 |
| Vừa | 4,2% → **41,7%** | 4,29 → 1,29 |
| Khó | 8,3% → **29,2%** | 5,75 → 4,62 |
| **Tổng** | **27,8% → 51,4%** | **3,44 → 2,12** |

49/72 ảnh (68,1%) lệch không quá 1 xu. Tập chỉnh (để đối chiếu): đúng 77,8%, MAE 0,44.
Biểu đồ: `results/tv2/hinh1_ti_le_dung_kiem_tra.png` (đúng), `hinh2_mae_…` (MAE), `hinh3_sai_so_…`
(phân bố sai số), `hinh4_thoi_gian_…` (thời gian).

**Nhận xét:**
1. **Watershed là bước đóng góp lớn nhất**: +23,6 điểm % tổng, đặc biệt mức Vừa (xu chạm nhau) 4% → 42%.
   Ở mức Dễ MAE tăng (0,29 → 0,46): có 2 ảnh bị đếm thừa sau watershed (không tách: 1 ảnh thừa) –
   watershed cắt một vật không tròn (xu vỡ mặt, dải bóng dài ở mép ảnh) thành nhiều mảnh (nhóm E bên dưới).
2. **Sai lệch TB âm** ở mọi cấu hình: lỗi chủ yếu là **đếm thiếu** (22 ảnh thiếu / 13 ảnh thừa) – các xu bị gộp.
3. **Cải tiến "độ khác màu nền"** hơn ảnh xám 4,2 điểm % trên tập kiểm tra (trên tập chỉnh hơn 28 điểm) và
   giảm sai số tương đối (33,7% → 26,5%). Nói trung thực: cải tiến có ích nhưng nhỏ hơn số liệu tập chỉnh gợi ý.
4. **Chênh lệch tập chỉnh (78%) và tập kiểm tra (51%)**: tham số được chọn trên chỉ 18 ảnh nên "khớp" với
   tập chỉnh; tập kiểm tra có nhiều ca khó mà tập chỉnh không có (cụm 15–20 xu, nền vải caro, ảnh có khung tối).
   Đây là lý do phải tách riêng tập kiểm tra.
5. **Tự viết vs OpenCV**: độ chính xác gần nhau (MAE 2,12 vs 2,00; 53/72 ảnh cho cùng số đếm); OpenCV
   nhanh hơn ~35 lần vì viết bằng C++. Khác biệt còn lại nằm ở watershed (xem dưới).

## So sánh với OpenCV
**Gán nhãn + đo** (18 ảnh tập chỉnh, 84 vật – `results/tv2/so_sanh_opencv_tv2.csv`):

| Bước | Hàm OpenCV | Giống | Tự viết | OpenCV |
|---|---|---|---|---|
| Gán nhãn BFS (8 lân cận) | `connectedComponents` | 100% (cùng cách chia vùng) | 31 ms | 0,15 ms |
| Diện tích, khung bao, tâm | `connectedComponentsWithStats` | 100% (tâm lệch ≤ 0,05 px) | 34 ms (cả gán nhãn) | 0,54 ms |

BFS chậm hơn bản "đoạn chạy + union-find" của TV1 (2,5 ms) vì duyệt từng pixel bằng Python, nhưng đúng
định nghĩa BFS trong phân công và dễ giải thích; kết quả hai bản giống hệt nhau.

**Pipeline đầy đủ:** để so sánh công bằng, ba quy ước của OpenCV phải chỉnh cho giống bản tự viết
(tìm ra khi so trên tập chỉnh; nếu để mặc định chỉ 8/18 ảnh đúng so với 14/18 của bản tự viết):
1. `cv2.distanceTransform` **không coi ngoài ảnh là nền** → vật chạm mép ảnh có khoảng cách lớn bất thường (60 thay vì 28),
   cửa sổ tìm đỉnh to ra và các xu nhỏ mất hạt giống. Sửa: đệm 1 viền 0.
2. `cv2.morphologyEx` mặc định coi ngoài ảnh là vật khi co → dùng `borderValue=0`; dùng chung phần tử cấu trúc `disk(r)`.
3. `cv2.watershed` loang theo **độ chênh màu** giữa 2 điểm kề nhau. Nếu đưa ảnh màu gốc (như ví dụ trong tài liệu OpenCV),
   hoa văn trên mặt xu tạo chênh lệch lớn và nhãn "nền" loang ngược vào chiếm gần hết đồng xu (có xu chỉ còn 67 pixel).
   Sửa: đưa ảnh địa hình "khoảng cách đảo ngược, nền = 255" – cùng ý tưởng với bản tự viết.

## Kết quả định tính
- `results/tv2/minh_hoa_thanh_cong_kiem_tra.png`: 6 ảnh đếm đúng (2 ảnh/mức) – ảnh gốc | sau làm sạch | kết quả.
- `results/tv2/minh_hoa_that_bai_kiem_tra.png`: 10 ảnh sai nặng nhất. Số **màu đỏ** trên ảnh = vật nghi gộp nhiều xu
  (`suspicious`: tỉ lệ rộng/cao > 1,4 hoặc độ đặc < 0,65; hình tròn có độ đặc ≈ π/4 ≈ 0,785).

## Phân tích thất bại (10 ảnh sai nặng nhất, tập kiểm tra)
Chi tiết từng ảnh: `results/tv2/phan_tich_that_bai_kiem_tra.csv`. Năm nhóm nguyên nhân:

| Nhóm | Số ảnh | Ví dụ | Nguyên nhân |
|---|---|---|---|
| **A. Lấp lỗ trong cụm dày** | 3 | `all_23560e97ad` (thật 15, đếm 3) | Xu xếp sát thành cụm → khe nền giữa các xu bị bao kín → `fill_holes` coi là lỗ và tô trắng → cả cụm thành khối đặc, không còn "yên ngựa" cho watershed |
| **B. Giả định "viền ảnh là nền" sai** | 2 | `peso_d61f49a8b8` (17 → 6) | Màu nền = trung vị màu ở viền ảnh. Ảnh có khung tối, hoặc xu phủ kín tới mép → màu nền sai → vật/nền đảo ngược |
| **C. Nền có vân / phản chiếu** | 2 | `yen_e16bf3970f` (6 → 1), `peso_857e443a5c` (8 → 28) | Vân vải caro / sọc khác màu nền nhiều như đồng xu → Otsu không tách được; xu phản chiếu vỡ thành nhiều mảnh |
| **D. Xu chồng nhiều tầng + bóng** | 1 | `yen_5e1ea72c72` (13 → 6) | Vùng chồng không có yên ngựa trong biến đổi khoảng cách (giống ca thất bại TV1 trên tập chỉnh) |
| **E. Vùng méo / dải dài bị cắt thừa** | 2 | `peso_12fab03c12` (3 → 8) | Dải bóng dài ở mép ảnh hoặc xu vỡ mặt có hình lõm → "ngọn đồi" khoảng cách có nhiều đỉnh → watershed cắt thành nhiều mảnh |

**Kiểm chứng nhóm A** (chỉ để phân tích, không đổi tham số theo tập kiểm tra): tắt lấp lỗ thì `all_23560e97ad` 3 → 15 (đúng),
`all_a9515c596b` 1 → 18 (thật 20), `yen_a11e78a456` 2 → 11 (thật 16). Nhưng tắt hẳn thì tập chỉnh tụt 14/18 → 5/18
(hoa văn tạo lỗ bên trong xu, xu vỡ thành nhiều mảnh). → **Đề xuất cho TV1:** chỉ lấp lỗ có diện tích nhỏ hơn
khoảng một nửa đồng xu (khe giữa các xu thường lớn hơn lỗ hoa văn); chọn ngưỡng trên tập chỉnh.

**Hạn chế của chỉ số "đếm đúng":** ảnh `yen_cff7ba9cf9` (tập chỉnh) đếm 13/14 – gần đúng – nhưng ảnh nhị phân sai hoàn toàn
(vân gỗ thành vật, xu thành nền); số đếm gần đúng là do trùng hợp. Chỉ số số lượng không phát hiện được lỗi này;
cần xem ảnh kết quả (hoặc IoU với mặt nạ như TV1 làm trên BBBC004).

## Câu hỏi thầy có thể hỏi
- *BFS khác DFS / union-find thế nào?* BFS loang theo "lớp" bằng hàng đợi (FIFO); DFS dùng ngăn xếp (đệ quy sâu dễ tràn
  với vật lớn); union-find gộp nhãn theo 2 lượt quét. Cả ba cho cùng kết quả; BFS mỗi điểm vào hàng đợi đúng 1 lần → O(số pixel).
- *Vì sao 8 lân cận?* Viền vật có răng cưa chéo; 4 lân cận có thể tách 1 xu thành nhiều mảnh. Đổi lại đường cắt watershed phải dày 2 px.
- *Vì sao lọc nhiễu theo tỉ lệ vật lớn nhất (5%) mà không theo số pixel?* Xu trong bộ dữ liệu to từ ~20 px đến ~200 px;
  một ngưỡng pixel cố định không hợp cho mọi ảnh. Nhược điểm: nếu ảnh có 1 vùng sai rất to, xu thật nhỏ có thể bị lọc mất.
- *Đường kính tương đương là gì?* d = 2·√(A/π): đường kính hình tròn có cùng diện tích – ít nhạy với viền răng cưa hơn đo chiều rộng.
- *MAE khác % đếm đúng?* % đúng chỉ biết đúng/sai; MAE cho biết sai bao nhiêu. Ảnh lệch 1 xu và lệch 19 xu đều "sai" nhưng MAE khác hẳn.
- *Vì sao tập kiểm tra thấp hơn tập chỉnh nhiều?* Tham số chọn trên 18 ảnh; tập kiểm tra có những kiểu ảnh khó mà tập chỉnh không có (xem phân tích thất bại).
- *Bản tự viết chậm hơn OpenCV 27 lần, có chấp nhận được?* 0,2 s/ảnh vẫn đủ cho ứng dụng tương tác; chậm nhất là watershed
  (hàng đợi ưu tiên duyệt từng pixel bằng Python), OpenCV viết bằng C++.
