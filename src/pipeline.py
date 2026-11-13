"""Ghép toàn bộ pipeline TV1 (và một bản đếm tạm, chờ TV2 thay bằng count.py chính thức).

    ảnh màu -> preprocess -> segment (Otsu) -> clean (hình thái học) -> separate (watershed) -> đếm
"""
import numpy as np
from .preprocess import preprocess
from .segment import segment, color_difference, segment_color
from .preprocess import resize_max, gaussian_blur
from .morphology import clean
from .separate import separate, label_bfs

DEFAULT = dict(max_side=512, blur_size=5, sigma=1.0, median_size=0,
               open_r=2, close_r=3, peak_ratio=0.4, min_ratio=0.2, min_area_ratio=0.05, mode="color")


def count_tmp(binary, min_area_ratio=0.05, mode="color"):
    """Đếm tạm: gán nhãn, bỏ vùng có diện tích < min_area_ratio * vùng lớn nhất (vụn nhiễu)."""
    lab, n = label_bfs(binary)
    if n == 0:
        return 0, lab
    areas = np.bincount(lab.ravel())[1:]
    keep = np.nonzero(areas >= min_area_ratio * areas.max())[0] + 1
    lab2 = np.where(np.isin(lab, keep), lab, 0)
    return len(keep), lab2


def run(img_rgb, use_separate=True, **kw):
    """Chạy cả pipeline. Trả về dict chứa ảnh từng bước + số lượng."""
    p = {**DEFAULT, **kw}
    gray, scale = preprocess(img_rgb, p["max_side"], p["blur_size"], p["sigma"], p["median_size"])
    if p["mode"] == "color":   # tách theo độ khác màu nền
        small, _ = resize_max(img_rgb, p["max_side"])
        feat = gaussian_blur(color_difference(small), p["blur_size"], p["sigma"])
        binary, t = segment_color(feat)
    else:                      # tách theo ảnh xám (Otsu cổ điển)
        binary, t = segment(gray)
    cleaned = clean(binary, p["open_r"], p["close_r"])
    if use_separate:
        sep, lab_ws, dist = separate(cleaned, p["peak_ratio"], p["min_ratio"], return_labels=True)
    else:
        sep, dist = cleaned, None
    n, labels = count_tmp(sep, p["min_area_ratio"])
    return dict(gray=gray, threshold=t, binary=binary, cleaned=cleaned, dist=dist,
                separated=sep, labels=labels, count=n, scale=scale)
