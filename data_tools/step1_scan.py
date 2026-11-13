"""Buoc 1: quet toan bo anh, loai anh trung, do cac dac trung co ban.
Dau ra: data_tools/out/all_images.csv
"""
import csv, hashlib, os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "archive")
IMG_DIR = os.path.join(ARCH, "coins_images", "coins_images")
OUT = os.path.join(ROOT, "data_tools", "out")
os.makedirs(OUT, exist_ok=True)

def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

rows = list(csv.DictReader(open(os.path.join(ARCH, "coins_count_values.csv"))))
seen, out = {}, []
for r in rows:
    p = os.path.join(IMG_DIR, r["folder"], r["image_name"])
    if not os.path.exists(p):
        continue
    h = md5(p)
    if h in seen:              # anh trung noi dung -> bo
        continue
    seen[h] = p
    img = Image.open(p).convert("RGB")
    w, hgt = img.size
    g = np.asarray(img.convert("L"), dtype=np.float32)
    # vien anh (5% moi canh) dung de uoc luong nen
    b = max(2, int(0.05 * min(w, hgt)))
    border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    # do "lon xon" cua nen: do lech chuan gradient o vien
    gy, gx = np.gradient(g)
    grad = np.hypot(gx, gy)
    gb = np.concatenate([grad[:b].ravel(), grad[-b:].ravel(), grad[:, :b].ravel(), grad[:, -b:].ravel()])
    # do sang khong deu: chenh lech trung binh 4 goc anh
    q = [g[:hgt//2, :w//2].mean(), g[:hgt//2, w//2:].mean(), g[hgt//2:, :w//2].mean(), g[hgt//2:, w//2:].mean()]
    out.append({
        "ten_anh": r["image_name"], "thu_muc": r["folder"], "so_luong": int(float(r["coins_count"])),
        "rong": w, "cao": hgt, "do_sang_tb": round(float(g.mean()), 1),
        "do_tuong_phan": round(float(g.std()), 1),
        "nen_std": round(float(border.std()), 1), "nen_gradient": round(float(gb.mean()), 2),
        "chenh_sang_goc": round(float(max(q) - min(q)), 1),
        "duong_dan": os.path.relpath(p, ROOT).replace(os.sep, "/"),
    })

with open(os.path.join(OUT, "all_images.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    wr.writeheader(); wr.writerows(out)
print("tong dong CSV:", len(rows), "| anh duy nhat:", len(out))
