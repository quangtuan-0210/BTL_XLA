"""Chạy pipeline TV1 trên bộ đồng xu và BBBC004, xuất bảng kết quả vào results/tv1/."""
import os, sys, time, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run
OUT = "results/tv1"; os.makedirs(OUT, exist_ok=True)
split = sys.argv[1] if len(sys.argv) > 1 else "chinh"
lab = pd.read_csv("data/labels.csv")
if split != "all": lab = lab[lab.tap == split]
rows = []
for r in lab.itertuples():
    img = np.array(Image.open("data/images/" + r.file).convert("RGB"))
    t = time.time(); a = run(img, use_separate=False); b = run(img, use_separate=True); dt = time.time() - t
    rows.append(dict(file=r.file, muc_do=r.muc_do, tap=r.tap, that=r.so_luong,
                     dem_khong_tach=a["count"], dem_co_tach=b["count"], giay=round(dt, 2)))
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/xu_{split}.csv", index=False)
for c in ["dem_khong_tach", "dem_co_tach"]:
    d["dung_" + c] = d[c] == d.that; d["sai_" + c] = (d[c] - d.that).abs()
g = d.groupby("muc_do").agg(so_anh=("file", "size"),
    dung_khong_tach=("dung_dem_khong_tach", "mean"), dung_co_tach=("dung_dem_co_tach", "mean"),
    MAE_khong_tach=("sai_dem_khong_tach", "mean"), MAE_co_tach=("sai_dem_co_tach", "mean")).round(2)
print(f"== Dong xu, tap '{split}' ({len(d)} anh) =="); print(g.to_string())
print("Tong: dung %.0f%% -> %.0f%% | MAE %.2f -> %.2f | TB %.2fs/anh" % (
    100*d.dung_dem_khong_tach.mean(), 100*d.dung_dem_co_tach.mean(),
    d.sai_dem_khong_tach.mean(), d.sai_dem_co_tach.mean(), d.giay.mean() / 2))
