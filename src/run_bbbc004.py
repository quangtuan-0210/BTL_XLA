"""Chạy pipeline TV1 trên ảnh tế bào BBBC004: đo IoU bước tách nền + đếm (đáp án 300/ảnh)."""
import os, sys, glob, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run
rows = []
files = sorted(glob.glob("data/bbbc004/images/*.png"))
lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, len(files))
for f in files[lo:hi]:
    g = np.array(Image.open(f)).astype(float); rgb = np.stack([g] * 3, -1)
    m = np.array(Image.open(f.replace("images", "masks"))) > 0
    kw = dict(max_side=950, mode="gray", open_r=1, close_r=1, min_area_ratio=0.02)
    a = run(rgb, use_separate=False, **kw); b = run(rgb, use_separate=True, **kw)
    bi = a["binary"].astype(bool)
    rows.append(dict(file=os.path.basename(f), dinh_pct=int(os.path.basename(f)[:3]) * 100 // 100,
                     iou_otsu=(bi & m).sum() / (bi | m).sum(), dem_khong_tach=a["count"], dem_co_tach=b["count"]))
out = "results/tv1/bbbc004.csv"
d = pd.DataFrame(rows)
if lo > 0 and os.path.exists(out): d = pd.concat([pd.read_csv(out), d]).drop_duplicates("file", keep="last")
d.to_csv(out, index=False)
d["muc"] = np.where(d.file.str.startswith("000"), "0% dính", "45% dính")
print(d.groupby("muc")[["iou_otsu", "dem_khong_tach", "dem_co_tach"]].mean().round(3).to_string())
print("(dap an: 300 te bao / anh)")
