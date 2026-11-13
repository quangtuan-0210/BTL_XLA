"""Vẽ ảnh từng bước cho nhiều ảnh: gốc | nhị phân | sau làm sạch | nhãn sau tách. Dùng để tìm lỗi."""
import os, sys, numpy as np, pandas as pd
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run
from src.preprocess import resize_max
split, out = sys.argv[1], sys.argv[2]
kw = dict(a.split("=") for a in sys.argv[3:]); kw = {k: float(v) if "." in v else int(v) for k, v in kw.items()}
lab = pd.read_csv("data/labels.csv"); lab = lab[lab.tap == split] if split != "all" else lab
T = 150; rng = np.random.default_rng(0); pal = rng.integers(60, 255, (5000, 3)); pal[0] = 0
sheet = Image.new("RGB", (4 * T, len(lab) * (T + 14)), "white"); dr = ImageDraw.Draw(sheet)
for i, r in enumerate(lab.itertuples()):
    img = np.array(Image.open("data/images/" + r.file).convert("RGB"))
    a = run(img, **kw)
    small, _ = resize_max(img, 512)
    tiles = [small.astype(np.uint8), a["binary"] * 255, a["cleaned"] * 255, pal[a["labels"]].astype(np.uint8)]
    for j, t in enumerate(tiles):
        im = Image.fromarray(np.asarray(t, np.uint8)); im.thumbnail((T - 4, T - 4))
        sheet.paste(im.convert("RGB"), (j * T + 2, i * (T + 14) + 2))
    dr.text((2, i * (T + 14) + T), f"{r.file[:14]} {r.muc_do} that={r.so_luong} dem={a['count']} t={a['threshold']}", fill="red" if a["count"] != r.so_luong else "green")
sheet.save(out); print("ok")
