"""Ghép toàn bộ pipeline:

    ảnh màu -> preprocess -> segment (Otsu) -> clean (hình thái học) -> separate (watershed)
            -> count_objects (gán nhãn BFS, lọc nhiễu, đo từng vật)
"""
from .preprocess_b1_3 import preprocess
from .segment_b2_4 import segment, color_difference, segment_color
from .preprocess_b1_3 import resize_max, gaussian_blur
from .morphology_b5 import clean
from .separate_b6_7 import separate
from .tach_vat_dai_b6_7 import separate_vat_dai, chon_loai_vat
from .count_b8 import count_objects

DEFAULT = dict(max_side=512, blur_size=5, sigma=1.0,
               close_r=3, peak_ratio=0.4, min_ratio=0.2, min_area_ratio=0.05, mode="color",
               loai_vat="tron")   # "tron": đồng xu, tế bào (bản gốc) | "dai": viên thuốc dài | "tu_dong": máy tự chọn (thử nghiệm)


def run(img_rgb, use_separate=True, **kw):
    """Chạy cả pipeline. Trả về dict chứa ảnh từng bước, số lượng (count),
    ảnh nhãn (labels, kích thước ảnh đã thu nhỏ) và thông số từng vật (objects, đổi về ảnh gốc)."""
    p = {**DEFAULT, **kw}
    gray, scale = preprocess(img_rgb, p["max_side"], p["blur_size"], p["sigma"])
    small, _ = resize_max(img_rgb, p["max_side"])
    if p["mode"] == "color":   # tách theo độ khác màu nền
        feat = gaussian_blur(color_difference(small), p["blur_size"], p["sigma"])
        binary, t = segment_color(feat)
    else:                      # tách theo ảnh xám (Otsu cổ điển)
        binary, t = segment(gray)
    cleaned = clean(binary, p["close_r"])
    loai = chon_loai_vat(cleaned) if p["loai_vat"] == "tu_dong" else p["loai_vat"]
    if use_separate and loai == "dai":     # thử nghiệm: watershed trên gradient
        sep, lab_ws, dist = separate_vat_dai(cleaned, small)
    elif use_separate:
        sep, lab_ws, dist = separate(cleaned, p["peak_ratio"], p["min_ratio"], return_labels=True)
    else:
        sep, dist = cleaned, None
    n, objects, labels = count_objects(sep, min_area_ratio=p["min_area_ratio"], scale=scale,
                                       return_labels=True)
    return dict(gray=gray, threshold=t, binary=binary, cleaned=cleaned, dist=dist,
                separated=sep, labels=labels, count=n, objects=objects, scale=scale,
                loai_vat=loai)
