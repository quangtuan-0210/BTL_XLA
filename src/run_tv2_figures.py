"""TV2 - Biểu đồ đánh giá + bộ ảnh minh họa ca thành công / thất bại.
Cần chạy src/run_tv2_eval.py <tap> trước.

Cách dùng:  python src/run_tv2_figures.py kiem_tra      (hoặc chinh)
Kết quả trong results/tv2/:
    hinh1_ti_le_dung_<tap>.png   tỉ lệ đếm đúng theo mức độ x cấu hình
    hinh2_mae_<tap>.png          MAE theo mức độ x cấu hình
    hinh3_sai_so_<tap>.png       phân bố sai số (đếm - thật) của pipeline chính
    hinh4_thoi_gian_<tap>.png    thời gian xử lý trung bình / ảnh
    minh_hoa_thanh_cong_<tap>.png, minh_hoa_that_bai_<tap>.png
    that_bai_<tap>.csv           danh sách ảnh sai của pipeline chính (để ghi nguyên nhân)
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run
from src.preprocess import resize_max
from src.measure import suspicious
from src.visualize import draw_objects, caption, grid
from src.evaluate import errors, LEVEL_VI

OUT = "results/tv2"
# Bảng màu phân loại (thứ tự cố định, đã kiểm tra mù màu): xanh, cam, xanh ngọc, vàng
COLORS = {"tu_viet": "#2a78d6", "tu_viet_khong_tach": "#eb6834", "tu_viet_xam": "#1baf7a", "opencv": "#eda100"}
NAMES = {"tu_viet": "Tự viết (chính)", "tu_viet_khong_tach": "Tự viết, không watershed",
         "tu_viet_xam": "Tự viết, Otsu ảnh xám", "opencv": "OpenCV (cùng thiết kế)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def grouped_bars(t, col, title, ylabel, fname, fmt, ymax=None):
    levels = ["de", "vua", "kho", "tong"]
    cfgs = [c for c in COLORS if c in set(t.cau_hinh)]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.8 / len(cfgs)
    x = np.arange(len(levels))
    for k, c in enumerate(cfgs):
        v = [t[(t.cau_hinh == c) & (t.muc_do == m)][col].iloc[0] for m in levels]
        bars = ax.bar(x + (k - (len(cfgs) - 1) / 2) * width, v, width * 0.9, color=COLORS[c], label=NAMES[c], zorder=3)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(val), ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(x, [LEVEL_VI[m] for m in levels])
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150); plt.close(fig)


def error_hist(d, split):
    e = (d.dem_tu_viet - d.that).clip(-5, 5)          # gộp sai số lớn vào 2 cột đầu/cuối
    vals = np.arange(-5, 6)
    cnt = [(e == v).sum() for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(vals, cnt, 0.85, color=COLORS["tu_viet"], zorder=3)
    for b, c in zip(bars, cnt):
        if c:
            ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(vals, ["≤ −5"] + [f"{v:+d}" if v else "0" for v in vals[1:-1]] + ["≥ +5"])
    ax.set_xlabel("Sai số = số đếm − số thật   (âm: đếm thiếu, dương: đếm thừa)")
    ax.set_ylabel("Số ảnh")
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_title(f"Phân bố sai số – pipeline tự viết (tập {split}, {len(d)} ảnh)", loc="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/hinh3_sai_so_{split}.png", dpi=150); plt.close(fig)


def time_bars(t, split):
    tt = t[t.muc_do == "tong"].set_index("cau_hinh")
    cfgs = [c for c in COLORS if c in tt.index]
    v = [1000 * tt.loc[c, "giay_tb"] for c in cfgs]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    bars = ax.barh(range(len(cfgs)), v, 0.6, color=[COLORS[c] for c in cfgs], zorder=3)
    for b, val in zip(bars, v):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {val:.0f} ms", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(cfgs)), [NAMES[c] for c in cfgs])
    ax.invert_yaxis()
    ax.set_xlabel("Thời gian xử lý trung bình / ảnh (ms)")
    ax.set_xlim(0, max(v) * 1.2)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_title(f"Thời gian chạy (tập {split})", loc="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/hinh4_thoi_gian_{split}.png", dpi=150); plt.close(fig)


def tile(file, T=260):
    """Ảnh gốc | sau làm sạch | kết quả đếm (cùng chiều cao T)."""
    img = np.array(Image.open("data/images/" + file).convert("RGB"))
    a = run(img)
    small = resize_max(img, 512)[0].astype(np.uint8)
    over = draw_objects(small, a["labels"], suspicious_ids=suspicious(a["objects"]))
    parts = [Image.fromarray(small), Image.fromarray((a["cleaned"] * 255).astype(np.uint8)).convert("RGB"), over]
    out = []
    for p in parts:
        p = p.copy(); p.thumbnail((T, T)); out.append(p)
    w = sum(p.width for p in out) + 8
    row = Image.new("RGB", (w, max(p.height for p in out)), "white")
    x = 0
    for p in out:
        row.paste(p, (x, 0)); x += p.width + 4
    return row, a["count"]


def sheets(d, split):
    lab = pd.read_csv("data/labels.csv").set_index("file")
    ok = d[d.dem_tu_viet == d.that]
    pick = pd.concat([ok[ok.muc_do == m].head(2) for m in ["de", "vua", "kho"]])
    tiles = []
    for r in pick.itertuples():
        im, n = tile(r.file)
        tiles.append(caption(im, f"{r.file}  |  {LEVEL_VI[r.muc_do]}  |  thật {r.that}, đếm {n}  ✓", (0, 110, 0)))
    if tiles:
        grid(tiles, 1).save(f"{OUT}/minh_hoa_thanh_cong_{split}.png")

    err = errors(d[["file"]].assign(dem=d.dem_tu_viet), lab.reset_index())
    err = err[err.sai_so != 0]
    err.to_csv(f"{OUT}/that_bai_{split}.csv", index=False)
    tiles = []
    for r in err.head(10).itertuples():
        im, n = tile(r.file)
        tiles.append(caption(im, f"{r.file}  |  {LEVEL_VI[r.muc_do]}  |  thật {r.so_luong}, đếm {n}  (sai {r.sai_so:+d})", (190, 0, 0)))
    if tiles:
        grid(tiles, 1).save(f"{OUT}/minh_hoa_that_bai_{split}.png")
    return err


def main(split):
    d = pd.read_csv(f"{OUT}/du_doan_{split}.csv")
    t = pd.read_csv(f"{OUT}/bang_ket_qua_{split}.csv")
    grouped_bars(t, "ti_le_dung_pct", f"Tỉ lệ ảnh đếm đúng (tập {split}, {len(d)} ảnh)", "% ảnh đếm đúng",
                 f"hinh1_ti_le_dung_{split}.png", lambda v: f"{v:.0f}", ymax=108)
    grouped_bars(t, "mae", f"Sai số tuyệt đối trung bình – MAE (tập {split})", "MAE (số vật lệch / ảnh)",
                 f"hinh2_mae_{split}.png", lambda v: f"{v:.2f}")
    error_hist(d, split)
    time_bars(t, split)
    err = sheets(d, split)
    print(f"Đã lưu biểu đồ và ảnh minh họa vào {OUT}/ ({len(err)} ảnh sai của pipeline chính)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "chinh")
