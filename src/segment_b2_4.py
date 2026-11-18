"""TV1 - Bước 4: Tách vật khỏi nền bằng ngưỡng Otsu (tự cài đặt).

Ý tưởng Otsu: ảnh xám có 256 mức sáng. Thử lần lượt từng ngưỡng t = 0..255,
chia điểm ảnh thành 2 nhóm (tối hơn t / sáng hơn t). Ngưỡng tốt nhất là ngưỡng làm
2 nhóm KHÁC NHAU NHIỀU NHẤT, tức "phương sai giữa hai lớp" lớn nhất:
        sigma_B^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2
    w0, w1  : tỉ lệ điểm ảnh của nhóm tối / nhóm sáng
    mu0, mu1: độ sáng trung bình của nhóm tối / nhóm sáng
"""
import numpy as np


def histogram(gray):
    """Đếm số điểm ảnh ở mỗi mức sáng 0..255."""
    g = np.clip(np.round(gray), 0, 255).astype(np.int64)
    return np.bincount(g.ravel(), minlength=256)


def otsu_threshold(gray):
    """Trả về ngưỡng Otsu (số nguyên 0..255)."""
    hist = histogram(gray).astype(np.float64)
    p = hist / hist.sum()                    # xác suất mỗi mức sáng
    levels = np.arange(256)
    w0 = np.cumsum(p)                        # tỉ lệ nhóm tối khi ngưỡng = t
    w1 = 1.0 - w0                            # tỉ lệ nhóm sáng
    m0_sum = np.cumsum(p * levels)           # tổng (xác suất x mức sáng) của nhóm tối
    m_total = m0_sum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0_sum / w0
        mu1 = (m_total - m0_sum) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between)         # ngưỡng làm 1 nhóm rỗng -> 0
    return int(np.argmax(between))


def border_mean(gray, frac=0.05):
    """Độ sáng trung bình ở viền ảnh - coi như độ sáng của NỀN."""
    h, w = gray.shape
    b = max(1, int(frac * min(h, w)))
    edge = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return edge.mean()


def segment(gray, threshold=None):
    """Ảnh xám -> ảnh nhị phân 0/1 (1 = vật, 0 = nền).

    Đồng xu có thể SÁNG hơn nền (nền đen) hoặc TỐI hơn nền (nền trắng).
    Ta xem viền ảnh sáng hay tối để biết đâu là nền, rồi lấy phía còn lại làm vật.
    Trả về (ảnh_nhị_phân, ngưỡng).
    """
    t = otsu_threshold(gray) if threshold is None else threshold
    if border_mean(gray) > t:                 # nền sáng -> vật là phần tối
        binary = gray <= t
    else:                                     # nền tối -> vật là phần sáng
        binary = gray > t
    return binary.astype(np.uint8), t


def background_color(img_rgb, frac=0.05):
    """Màu nền = TRUNG VỊ màu ở viền ảnh (trung vị không bị ảnh hưởng bởi watermark, chữ ở mép)."""
    h, w = img_rgb.shape[:2]
    b = max(1, int(frac * min(h, w)))
    edge = np.concatenate([img_rgb[:b].reshape(-1, 3), img_rgb[-b:].reshape(-1, 3),
                           img_rgb[:, :b].reshape(-1, 3), img_rgb[:, -b:].reshape(-1, 3)])
    return np.median(edge, axis=0)


def color_difference(img_rgb):
    """Ảnh "độ khác màu nền": mỗi điểm = khoảng cách Euclid trong không gian RGB tới màu nền,
    co giãn về 0..255. Đồng xu (dù sáng hay tối, vàng hay bạc) đều khác màu nền -> giá trị lớn.
    Nhờ vậy dùng được cả với nền màu (đỏ, xanh) mà ảnh xám dễ nhầm."""
    img = np.asarray(img_rgb, dtype=np.float64)
    bg = background_color(img)
    diff = np.sqrt(((img - bg) ** 2).sum(axis=2))
    return diff * (255.0 / max(diff.max(), 1e-9))


def segment_color(diff):
    """Otsu trên ảnh độ-khác-màu-nền: vật = điểm khác nền nhiều hơn ngưỡng."""
    t = otsu_threshold(diff)
    return (diff > t).astype(np.uint8), t
