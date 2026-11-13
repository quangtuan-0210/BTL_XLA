"""So sánh từng bước TV1 (tự viết) với hàm OpenCV tương ứng: độ giống nhau + thời gian chạy.
Chạy trên 18 ảnh tập chỉnh. Kết quả: results/tv1/so_sanh_opencv.csv"""
import os, sys, time, numpy as np, pandas as pd, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import to_gray, resize_max, gaussian_blur, median_filter
from src.segment import otsu_threshold
from src.morphology import erode, dilate, opening, closing, disk
from src.separate import distance_transform, find_markers, watershed, label_bfs

def timed(f, *a, rep=3):
    t = time.perf_counter()
    for _ in range(rep): out = f(*a)
    return out, (time.perf_counter() - t) / rep * 1000

lab = pd.read_csv("data/labels.csv"); lab = lab[lab.tap == "chinh"]
rec = {}
def add(step, cv_name, same, t_me, t_cv):
    rec.setdefault(step, dict(buoc=step, ham_opencv=cv_name, giong=[], ms_tu_viet=[], ms_opencv=[]))
    rec[step]["giong"].append(same); rec[step]["ms_tu_viet"].append(t_me); rec[step]["ms_opencv"].append(t_cv)

for r in lab.itertuples():
    rgb, _ = resize_max(np.array(Image.open("data/images/" + r.file).convert("RGB")), 512)
    rgb8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    g, a = timed(to_gray, rgb8); gc, b = timed(cv2.cvtColor, rgb8, cv2.COLOR_RGB2GRAY)
    add("1. Xám hóa", "cvtColor", np.mean(np.abs(g - gc) <= 1), a, b)
    g8 = np.clip(np.round(g), 0, 255).astype(np.uint8); gf = g8.astype(np.float64)
    x, a = timed(gaussian_blur, gf, 5, 1.0); y, b = timed(cv2.GaussianBlur, gf, (5, 5), 1.0)
    add("2. Lọc Gauss 5x5", "GaussianBlur", np.mean(np.abs(x - y) < 1e-6), a, b)
    x, a = timed(median_filter, g8, 3); y, b = timed(cv2.medianBlur, g8, 3)
    add("3. Lọc trung vị 3x3", "medianBlur", np.mean(x[1:-1, 1:-1] == y[1:-1, 1:-1]), a, b)
    t1, a = timed(otsu_threshold, g8); (t2, _), b = timed(cv2.threshold, g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    add("4. Ngưỡng Otsu", "threshold(OTSU)", float(t1 == t2), a, b)
    bi = (g8 > t1).astype(np.uint8); se = disk(3)
    kw = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)
    x, a = timed(erode, bi, se); y, b = timed(lambda: cv2.erode(bi, se, **kw)); add("5a. Co", "erode", np.mean(x == y), a, b)
    x, a = timed(dilate, bi, se); y, b = timed(lambda: cv2.dilate(bi, se, **kw)); add("5b. Giãn", "dilate", np.mean(x == y), a, b)
    x, a = timed(opening, bi, se); y, b = timed(lambda: cv2.morphologyEx(bi, cv2.MORPH_OPEN, se, **kw)); add("5c. Mở", "morphologyEx(OPEN)", np.mean(x == y), a, b)
    x, a = timed(closing, bi, se); y, b = timed(lambda: cv2.morphologyEx(bi, cv2.MORPH_CLOSE, se, **kw)); add("5d. Đóng", "morphologyEx(CLOSE)", np.mean(x[5:-5, 5:-5] == y[5:-5, 5:-5]), a, b)
    x, a = timed(distance_transform, bi, rep=1); y, b = timed(cv2.distanceTransform, bi, cv2.DIST_L2, 5)
    m = bi > 0
    # chamfer 3-4 là xấp xỉ: đo theo SAI SỐ TƯƠNG ĐỐI (lệch / giá trị đúng), đổi thành "% giống" = 100 - sai số
    add("6. Biến đổi khoảng cách", "distanceTransform(L2)", 1 - np.mean(np.abs(x - y)[m] / np.maximum(y[m], 1)) if m.any() else 1.0, a, b)
    mk, _ = find_markers(x)
    ws, a = timed(watershed, x, mk, m, rep=1)
    # OpenCV watershed ngập theo độ chênh màu giữa 2 điểm kề nhau (gradient), bản tự viết ngập theo
    # khoảng cách -> đường cắt nằm khác chỗ, không so từng pixel. Chỉ so: số vùng tạo ra từ cùng hạt giống.
    mk32 = mk.astype(np.int32).copy(); mk32[~m] = mk.max() + 1
    wc, b = timed(lambda: cv2.watershed(np.ascontiguousarray(rgb8[..., ::-1]), mk32.copy()), rep=1)
    n_me = len(np.unique(ws[ws > 0])); n_cv = len(np.unique(wc[(wc > 0) & (wc <= mk.max())]))
    add("7. Watershed (cùng hạt giống)", "watershed", float(n_me == n_cv), a, b)
    (l1, n1), a = timed(label_bfs, bi, rep=1); (n2, _), b = timed(cv2.connectedComponents, bi, None, 8)
    add("(phụ) Gán nhãn BFS", "connectedComponents", float(n1 == n2 - 1), a, b)

rows = []
for v in rec.values():
    rows.append(dict(buoc=v["buoc"], ham_opencv=v["ham_opencv"],
                     ti_le_giong_pct=round(100 * np.mean(v["giong"]), 1),
                     ms_tu_viet=round(np.mean(v["ms_tu_viet"]), 2), ms_opencv=round(np.mean(v["ms_opencv"]), 3),
                     cham_hon_x_lan=round(np.mean(v["ms_tu_viet"]) / max(np.mean(v["ms_opencv"]), 1e-3))))
d = pd.DataFrame(rows); os.makedirs("results/tv1", exist_ok=True)
d.to_csv("results/tv1/so_sanh_opencv.csv", index=False, encoding="utf-8")
print(d.to_string(index=False))
