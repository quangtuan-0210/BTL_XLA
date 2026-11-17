"""So sánh phần TV2 (tự viết) với OpenCV: gán nhãn + đo thông số. Độ giống nhau + thời gian chạy.
Đầu vào: ảnh nhị phân đã tách vật dính của 18 ảnh tập chỉnh (từ pipeline tự viết).
Kết quả: results/tv2/so_sanh_opencv_tv2.csv

    label_components (BFS)   <-> cv2.connectedComponents(connectivity=8)
    measure_objects          <-> cv2.connectedComponentsWithStats (diện tích, khung bao, tâm)
"""
import os, sys, time
import cv2, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run
from src.count import label_components
from src.measure import measure_objects
from src.separate import label_bfs


def timed(f, *a, rep=3):
    t = time.perf_counter()
    for _ in range(rep):
        out = f(*a)
    return out, (time.perf_counter() - t) / rep * 1000


def same_partition(a, b):
    """Hai ảnh nhãn chia vùng giống hệt nhau (chỉ có thể khác SỐ HIỆU nhãn)?"""
    m = (a > 0) | (b > 0)
    if not np.array_equal(a > 0, b > 0):
        return False
    pairs = np.unique(np.stack([a[m], b[m]]), axis=1)          # các cặp (nhãn a, nhãn b) xuất hiện
    return len(pairs[0]) == len(np.unique(pairs[0])) == len(np.unique(pairs[1]))  # tương ứng 1-1


lab = pd.read_csv("data/labels.csv"); lab = lab[lab.tap == "chinh"]
rec = {k: dict(giong=[], me=[], cv=[]) for k in ("label", "label_run", "area", "bbox", "centroid", "measure_time")}
n_obj = 0
for r in lab.itertuples():
    img = np.array(Image.open("data/images/" + r.file).convert("RGB"))
    sep = run(img)["separated"].astype(np.uint8)
    (l_me, n_me), t_me = timed(label_components, sep)
    (n_cv, l_cv, stats, cents), t_cv = timed(lambda: cv2.connectedComponentsWithStats(sep, connectivity=8))
    _, t_cv0 = timed(lambda: cv2.connectedComponents(sep, connectivity=8))
    rec["label"]["giong"].append(float(n_me == n_cv - 1 and same_partition(l_me, l_cv)))
    rec["label"]["me"].append(t_me); rec["label"]["cv"].append(t_cv0)
    (l_run, _), t_run = timed(label_bfs, sep)
    rec["label_run"]["giong"].append(float(same_partition(l_run, l_cv))); rec["label_run"]["me"].append(t_run); rec["label_run"]["cv"].append(t_cv0)

    objs, t_meas = timed(measure_objects, l_me)
    rec["measure_time"]["giong"].append(1.0); rec["measure_time"]["me"].append(t_meas + t_me); rec["measure_time"]["cv"].append(t_cv)
    # ghép vật của 2 bên theo nhãn của OpenCV tại một điểm thuộc vật
    for o in objs:
        ys, xs = np.nonzero(l_me == o["id"])
        k = l_cv[ys[0], xs[0]]
        x, y, w, h, a = stats[k]
        rec["area"]["giong"].append(float(o["dien_tich"] == a))
        rec["bbox"]["giong"].append(float((o["x"], o["y"], o["rong"], o["cao"]) == (x, y, w, h)))
        rec["centroid"]["giong"].append(float(abs(o["tam_x"] - cents[k][0]) <= 0.05 and abs(o["tam_y"] - cents[k][1]) <= 0.05))
        n_obj += 1

names = {"label": ("1. Gán nhãn BFS (8 lân cận)", "connectedComponents"),
         "label_run": ("1b. Gán nhãn theo đoạn chạy (bản TV1)", "connectedComponents"),
         "area": ("3a. Diện tích", "ConnectedComponentsWithStats: AREA"),
         "bbox": ("3b. Khung bao", "...WithStats: LEFT, TOP, WIDTH, HEIGHT"),
         "centroid": ("3c. Tâm (lệch <= 0,05 px)", "...WithStats: centroids"),
         "measure_time": ("Gán nhãn + đo (tổng thời gian)", "connectedComponentsWithStats")}
rows = []
for k, (step, cvname) in names.items():
    v = rec[k]
    row = dict(buoc=step, ham_opencv=cvname, ti_le_giong_pct=round(100 * np.mean(v["giong"]), 1))
    if v["me"]:
        row.update(ms_tu_viet=round(np.mean(v["me"]), 2), ms_opencv=round(np.mean(v["cv"]), 3),
                   cham_hon_x_lan=round(np.mean(v["me"]) / max(np.mean(v["cv"]), 1e-3)))
    rows.append(row)
d = pd.DataFrame(rows)
os.makedirs("results/tv2", exist_ok=True)
d.to_csv("results/tv2/so_sanh_opencv_tv2.csv", index=False)
print(d.to_string(index=False))
print(f"({len(lab)} ảnh, {n_obj} vật được so thông số)")
