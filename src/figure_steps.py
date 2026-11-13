"""Vẽ hình minh họa các bước TV1 cho 1 ảnh (dùng cho báo cáo/slide).
Cách dùng: python src/figure_steps.py <ten_file_trong_data/images> <file_png_ra>"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run
from src.preprocess import resize_max
f, out = sys.argv[1], sys.argv[2]
img = np.array(Image.open("data/images/" + f).convert("RGB")); a = run(img); small, _ = resize_max(img, 512)
rng = np.random.default_rng(3); pal = rng.integers(60, 255, (a["labels"].max() + 1, 3)); pal[0] = 0
panels = [("1. Ảnh gốc (thu nhỏ)", small.astype(np.uint8), None), ("2. Xám hóa + lọc Gauss", a["gray"], "gray"),
          (f"3. Tách nền Otsu (ngưỡng {a['threshold']})", a["binary"], "gray"), ("4. Hình thái học: đóng, lấp lỗ, mở", a["cleaned"], "gray"),
          ("5. Biến đổi khoảng cách", a["dist"], "magma"), (f"6. Watershed → đếm được {a['count']} xu", pal[a["labels"]].astype(np.uint8), None)]
fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
for k, (t, im, cm) in enumerate(panels):
    ax.flat[k].imshow(im, cmap=cm); ax.flat[k].set_title(t, fontsize=11, loc="left"); ax.flat[k].axis("off")
fig.tight_layout(); fig.savefig(out, dpi=120); print("saved", out, "count", a["count"])
