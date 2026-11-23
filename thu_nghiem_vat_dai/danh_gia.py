"""Thử nghiệm chế độ vật dài: so bản gốc (loai_vat="tron") với chế độ mới (loai_vat="dai")
trên ảnh viên thuốc tổng hợp có đáp án.

Cách dùng (chạy từ thư mục gốc dự án):
    python thu_nghiem_vat_dai/sinh_anh_thuoc.py thu_nghiem_vat_dai/anh/ 10 1   # tập chỉnh (10 ảnh)
    python thu_nghiem_vat_dai/sinh_anh_thuoc.py thu_nghiem_vat_dai/anh/ 30 2   # tập kiểm tra (30 ảnh)
    python thu_nghiem_vat_dai/danh_gia.py thu_nghiem_vat_dai/anh/labels_2.json
Tham số chế độ mới (q=40, co=2, ti_le_loi=0.2, ti_le_max=4.2) chọn trên tập chỉnh (seed 1).
"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run

nhan = json.load(open(sys.argv[1])); thu_muc = os.path.dirname(sys.argv[1])
kq = {"tron": [], "dai": []}
for l in nhan:
    img = np.array(Image.open(os.path.join(thu_muc, l["file"])).convert("RGB"))
    for k in kq:
        kq[k].append(run(img, loai_vat=k)["count"] - l["so_luong"])
for k, e in kq.items():
    e = np.array(e)
    print(f"{k:5s}: đúng {(e == 0).sum()}/{len(e)} ({100 * (e == 0).mean():.1f}%), MAE {np.abs(e).mean():.2f}, "
          f"lệch ±1: {(np.abs(e) <= 1).sum()}/{len(e)}, thiếu/thừa: {(e < 0).sum()}/{(e > 0).sum()}")
