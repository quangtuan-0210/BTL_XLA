"""Đánh giá định lượng số lượng đếm được so với đáp án.

Chỉ số (với mỗi ảnh i: d_i = số đếm được, t_i = số thật, e_i = d_i - t_i):
    Tỉ lệ đếm đúng  = (số ảnh có d_i = t_i) / (tổng số ảnh)
    MAE             = trung bình |e_i|          (mỗi ảnh lệch trung bình bao nhiêu vật)
    Sai lệch TB     = trung bình e_i            (âm = có xu hướng đếm THIẾU, dương = đếm THỪA)
    Sai số tương đối= trung bình |e_i| / t_i    (lệch 1 trên ảnh 2 xu nặng hơn lệch 1 trên ảnh 20 xu)
    Số ảnh đếm thiếu / thừa
Tất cả tính cho toàn tập và riêng từng mức Dễ / Vừa / Khó.

File dự đoán (pred_csv) có ít nhất 2 cột: file, dem. Có thể có thêm cột giay (thời gian / ảnh).
"""
import numpy as np
import pandas as pd

LEVELS = ["de", "vua", "kho"]
LEVEL_VI = {"de": "Dễ", "vua": "Vừa", "kho": "Khó", "tong": "Tổng"}


def _metrics(d):
    e = d["dem"] - d["so_luong"]
    out = dict(so_anh=int(len(d)),
               ti_le_dung=float((e == 0).mean()) if len(d) else float("nan"),
               mae=float(e.abs().mean()) if len(d) else float("nan"),
               sai_lech_tb=float(e.mean()) if len(d) else float("nan"),
               sai_so_tuong_doi=float((e.abs() / d["so_luong"]).mean()) if len(d) else float("nan"),
               so_anh_dem_thieu=int((e < 0).sum()),
               so_anh_dem_thua=int((e > 0).sum()))
    if "giay" in d:
        out["giay_tb"] = float(d["giay"].mean())
    return out


def evaluate(pred_csv, label_csv="data/labels.csv"):
    """Hàm đã thống nhất trong doc chia việc. Trả về dict:
        {"tong": {...chỉ số...}, "theo_muc": {"de": {...}, "vua": {...}, "kho": {...}}}
    pred_csv / label_csv có thể là đường dẫn hoặc DataFrame."""
    pred = pred_csv if isinstance(pred_csv, pd.DataFrame) else pd.read_csv(pred_csv)
    lab = label_csv if isinstance(label_csv, pd.DataFrame) else pd.read_csv(label_csv)
    keep = [c for c in ("file", "dem", "giay") if c in pred]
    d = pred[keep].merge(lab[["file", "so_luong", "muc_do"]], on="file", how="inner")
    if len(d) != len(pred):
        missing = set(pred["file"]) - set(d["file"])
        raise ValueError(f"{len(missing)} ảnh trong file dự đoán không có trong đáp án: {sorted(missing)[:5]}")
    return dict(tong=_metrics(d), theo_muc={m: _metrics(d[d.muc_do == m]) for m in LEVELS})


def result_table(results):
    """{tên_cấu_hình: dict từ evaluate()} -> bảng dài: cau_hinh, muc_do, các chỉ số."""
    rows = []
    for name, r in results.items():
        for m in LEVELS + ["tong"]:
            met = r["tong"] if m == "tong" else r["theo_muc"][m]
            rows.append(dict(cau_hinh=name, muc_do=m, **met))
    t = pd.DataFrame(rows)
    for c in ("ti_le_dung", "sai_so_tuong_doi"):
        t[c] = (100 * t[c]).round(1)
    for c in ("mae", "sai_lech_tb"):
        t[c] = t[c].round(2)
    if "giay_tb" in t:
        t["giay_tb"] = t["giay_tb"].round(3)
    return t.rename(columns={"ti_le_dung": "ti_le_dung_pct", "sai_so_tuong_doi": "sai_so_tuong_doi_pct"})


def errors(pred_csv, label_csv="data/labels.csv"):
    """Bảng từng ảnh kèm sai số, sắp xếp theo |sai số| giảm dần (dùng cho phân tích thất bại)."""
    pred = pred_csv if isinstance(pred_csv, pd.DataFrame) else pd.read_csv(pred_csv)
    lab = label_csv if isinstance(label_csv, pd.DataFrame) else pd.read_csv(label_csv)
    d = pred[["file", "dem"]].merge(lab[["file", "so_luong", "muc_do"]], on="file")
    d["sai_so"] = d["dem"] - d["so_luong"]
    return d.reindex(d["sai_so"].abs().sort_values(ascending=False).index)
