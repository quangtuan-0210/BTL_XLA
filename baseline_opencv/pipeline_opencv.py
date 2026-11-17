"""Pipeline đối chứng dùng HÀM CÓ SẴN của OpenCV, cùng thiết kế và cùng tham số với pipeline tự viết.

    ảnh màu -> cv2.resize -> độ khác màu nền -> cv2.GaussianBlur -> cv2.threshold(OTSU)
            -> cv2.morphologyEx(CLOSE) -> lấp lỗ (cv2.findContours + drawContours) -> cv2.morphologyEx(OPEN)
            -> cv2.distanceTransform -> hạt giống (cv2.dilate làm lọc cực đại) -> cv2.watershed
            -> đếm vùng watershed (hoặc cv2.connectedComponents nếu không tách) -> lọc vùng nhỏ

Để so sánh công bằng, hai pipeline dùng CHUNG:
  - bước "độ khác màu nền" (cải tiến của TV1, không có hàm OpenCV tương ứng; chỉ là phép tính số học),
  - phần tử cấu trúc disk(r), quy ước "ngoài ảnh là nền" (borderValue=0, đệm viền cho distanceTransform),
  - quy tắc tìm hạt giống và lọc vùng nhỏ, cùng tham số DEFAULT.
=> hai bên chỉ khác nhau ở phần CÀI ĐẶT các phép toán xử lý ảnh.
Ba điểm này tìm ra khi so trên tập chỉnh: nếu để mặc định OpenCV thì chỉ 8/18 ảnh đúng (xem README_TV2.md).
"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.segment import color_difference
from src.pipeline import DEFAULT
from src.morphology import disk


def _se(r):
    """Phần tử cấu trúc hình tròn. Dùng CHUNG disk(r) với bản tự viết (cv2.getStructuringElement
    ELLIPSE cho hình hơi khác) để hai bên chỉ khác nhau ở phần cài đặt phép toán."""
    return disk(r)


BORDER = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)   # ngoài ảnh = nền, giống bản tự viết


def distance_cv(binary):
    """cv2.distanceTransform KHÔNG coi ngoài ảnh là nền -> vật chạm mép ảnh có khoảng cách lớn bất thường.
    Đệm 1 viền 0 để cùng quy ước với bản tự viết."""
    return cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]


def fill_holes_cv(binary):
    """Lấp lỗ: tìm đường viền NGOÀI của từng vật rồi tô kín bên trong."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(binary)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out


def run_opencv(img_rgb, use_separate=True, **kw):
    p = {**DEFAULT, **kw}
    img = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    s = p["max_side"] / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_LINEAR)
    else:
        s = 1.0
    if p["mode"] == "color":
        feat = color_difference(img)
    else:
        feat = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    k = p["blur_size"]
    feat = cv2.GaussianBlur(feat, (k, k), p["sigma"]) if k > 1 else feat
    feat8 = np.clip(np.round(feat), 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(feat8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if p["mode"] == "color":
        binary = (feat8 > t).astype(np.uint8)
    else:  # nền sáng -> vật là phần tối (giống src.segment.segment)
        b = max(1, int(0.05 * min(feat8.shape)))
        border = np.concatenate([feat8[:b].ravel(), feat8[-b:].ravel(), feat8[:, :b].ravel(), feat8[:, -b:].ravel()])
        binary = ((feat8 <= t) if border.mean() > t else (feat8 > t)).astype(np.uint8)
    m = binary
    if p["close_r"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _se(p["close_r"]), **BORDER)
    m = fill_holes_cv(m)
    if p["open_r"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _se(p["open_r"]), **BORDER)
    cleaned = m

    if use_separate and cleaned.any():
        dist = distance_cv(cleaned)
        top = float(dist.max())
        r = max(3, int(p["peak_ratio"] * top))
        local_max = cv2.dilate(dist, np.ones((2 * r + 1, 2 * r + 1), np.uint8), **BORDER)
        peaks = ((dist >= local_max) & (dist >= max(p["min_ratio"] * top, 3))).astype(np.uint8)
        peaks = cv2.dilate(peaks, _se(2))
        n_mk, markers = cv2.connectedComponents(peaks, connectivity=8)
        # nhãn cho watershed: 1 = nền chắc chắn, 2.. = hạt giống, 0 = vùng chưa biết (sẽ được loang)
        mk = np.where(peaks > 0, markers + 1, 0).astype(np.int32)
        mk[cleaned == 0] = 1
        # Ảnh địa hình cho watershed: trong vật = khoảng cách đảo ngược (tâm thấp, gần biên cao, 0..60),
        # nền = 255. Nếu đưa ảnh màu gốc như ví dụ trong tài liệu OpenCV, hoa văn trên mặt xu tạo
        # gradient lớn và nhãn "nền" loang ngược vào chiếm gần hết đồng xu (đã thử: 8-9/18 ảnh đúng).
        topo = np.where(cleaned > 0, 60 * (1 - dist / max(top, 1e-9)), 255).astype(np.uint8)
        ws = cv2.watershed(cv2.merge([topo, topo, topo]), mk)
        regions = np.where(ws > 1, ws - 1, 0).astype(np.int32)   # mỗi vùng watershed = 1 vật
        separated = (regions > 0).astype(np.uint8)
        labels = regions
    else:
        _, labels = cv2.connectedComponents(cleaned, connectivity=8)
        separated = cleaned
    # lọc vùng nhỏ (cùng quy tắc min_area_ratio) và đánh số lại
    areas = np.bincount(labels.ravel()); areas[0] = 0
    if areas.max() == 0:
        return dict(count=0, labels=np.zeros_like(labels), cleaned=cleaned, separated=separated, scale=s)
    keep = np.nonzero(areas >= p["min_area_ratio"] * areas.max())[0]
    new = np.zeros(len(areas), np.int32); new[keep] = np.arange(1, len(keep) + 1)
    return dict(count=len(keep), labels=new[labels], cleaned=cleaned, separated=separated,
                threshold=int(t), scale=s)
