"""Test hỗn hợp: ảnh đồng xu (tập kiểm tra 72 ảnh) + ảnh viên thuốc tổng hợp (tập kiểm tra 30 ảnh).
So 3 cách: luôn "tron", luôn "dai", "tu_dong" (máy tự chọn theo hình dạng).
Cách dùng: python thu_nghiem_vat_dai/danh_gia_hon_hop.py   (cần sinh ảnh thuốc seed 2 trước)"""
import json, os, sys
import numpy as np, pandas as pd
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.pipeline import run

ds = []
lab = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
for f, n in lab[lab.tap == "kiem_tra"][["file", "so_luong"]].values:
    ds.append(("xu", os.path.join(ROOT, "data", "images", f), n))
for l in json.load(open(os.path.join(ROOT, "thu_nghiem_vat_dai", "anh", "labels_2.json"))):
    ds.append(("thuoc", os.path.join(ROOT, "thu_nghiem_vat_dai", "anh", l["file"]), l["so_luong"]))

rows = []
for loai, f, n in ds:
    img = np.array(Image.open(f).convert("RGB"))
    r = {k: run(img, loai_vat=k) for k in ("tron", "dai", "tu_dong")}
    rows.append(dict(loai=loai, file=os.path.basename(f), that=n, tron=r["tron"]["count"], dai=r["dai"]["count"],
                     tu_dong=r["tu_dong"]["count"], may_chon=r["tu_dong"]["loai_vat"]))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "thu_nghiem_vat_dai", "ket_qua_hon_hop.csv"), index=False)
print("Máy chọn đúng chế độ:")
for loai, dung in (("xu", "tron"), ("thuoc", "dai")):
    s = d[d.loai == loai]; print(f"  {loai}: {(s.may_chon == dung).sum()}/{len(s)}")
print("\nTỉ lệ đếm đúng / MAE:")
for k in ("tron", "dai", "tu_dong"):
    out = []
    for loai in ("xu", "thuoc", None):
        s = d if loai is None else d[d.loai == loai]; e = s[k] - s.that
        out.append(f"{'tổng' if loai is None else loai} {(e == 0).sum()}/{len(s)} ({100 * (e == 0).mean():.1f}%), MAE {e.abs().mean():.2f}")
    print(f"  {k:8s}: " + " | ".join(out))
