""" Bước 6-7 (THỬ NGHIỆM, dành cho vật DÀI, bề mặt trơn như viên thuốc con nhộng).

Vì sao cần: bản gốc (separate_b6_7) lấy hạt giống là các ĐỈNH của bản đồ khoảng cách,
giả định mỗi vật có một tâm. Viên thuốc dài có cả một "sống núi" dọc thân -> nhiều đỉnh
-> bị cắt đôi; hai viên nằm sát nhau theo chiều dài thì không có chỗ eo -> bị gộp.
Chỉ nhìn hình dạng khối trắng là không đủ, nên ở đây dùng thêm thông tin BIÊN của ảnh màu.

Bước 6' - Độ lớn gradient (Chương 6, phát hiện biên): toán tử Sobel trên từng kênh màu
    (đã làm mịn Gauss), lấy kênh lớn nhất. Chỗ hai viên chạm nhau có viền tối / bóng
    -> gradient lớn; lòng viên thuốc trơn -> gradient nhỏ.
Bước 7' - Watershed trên ảnh gradient (đúng như slide Chương 6: ảnh = địa hình,
    đường đê nằm ở chỗ biên mạnh):
    a) Hạt giống = "lõi" của từng viên: điểm thuộc vật có gradient thấp, co nhẹ để cắt
       các cầu nối mảnh, gán nhãn, bỏ lõi quá nhỏ.
    b) Nước dâng từ hạt giống, điểm gradient THẤP được loang trước.
    c) Gộp lại 2 vùng kề nhau nếu chúng là 2 nửa của CÙNG một viên (vết nối viên 2 màu,
       vệt bóng sáng dọc thân): dùng mô-men bậc 2 (Chương 7) để đo dài, rộng, độ đầy
       của vùng gộp; vùng gộp vẫn có dạng một viên thuốc (đầy, không quá dài, không rộng
       hơn) thì gộp.

Không dùng cho đồng xu: mặt xu có hoa văn nổi -> gradient lớn khắp nơi -> cắt vụn
(tập chỉnh đồng xu chỉ còn 1/18 ảnh đúng). Vì vậy đây là một CHẾ ĐỘ RIÊNG.
"""
from collections import Counter

import numpy as np

from .preprocess_b1_3 import conv2d, gaussian_blur
from .morphology_b5 import erode, disk
from .separate_b6_7 import watershed, label_bfs
from .count_b8 import remove_small

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
SOBEL_Y = SOBEL_X.T


def gradient_mau(small_rgb, blur_size=5, sigma=1.0):
    """Độ lớn gradient Sobel, lấy lớn nhất trên 3 kênh màu."""
    g = np.zeros(small_rgb.shape[:2])
    for ch in range(3):
        b = gaussian_blur(small_rgb[..., ch], blur_size, sigma)
        g = np.maximum(g, np.hypot(conv2d(b, SOBEL_X), conv2d(b, SOBEL_Y)))
    return g


def hat_giong_loi(mask, grad, q=40, co=2, ti_le_loi=0.2):
    """Lõi trơn của từng vật: gradient < phân vị q (tính trong vật), co bán kính `co`."""
    t = np.percentile(grad[mask], q)
    loi = erode((mask & (grad < t)).astype(np.uint8), disk(co)).astype(bool)
    lab, _ = label_bfs(loi)
    lab, _ = remove_small(lab, min_area_ratio=ti_le_loi)
    return lab


def hinh_dang(ys, xs):
    """Dài, rộng (theo hướng chính, từ mô-men bậc 2) và độ đầy = diện tích / (dài x rộng)."""
    if len(xs) < 3:
        return 1.0, 1.0, 1.0
    cov = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    l1, l2 = sorted(np.linalg.eigvalsh(cov))[::-1]
    dai, rong = np.sqrt(12 * max(l1, 1e-9)), np.sqrt(12 * max(l2, 1e-9))
    return dai, rong, len(xs) / (dai * rong)


def _bien_chung(lab):
    c = Counter()
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1], lab[1:])):
        k = (a != b) & (a > 0) & (b > 0)
        for p, q in zip(a[k].tolist(), b[k].tolist()):
            c[(min(p, q), max(p, q))] += 1
    return c


def gop_hai_nua(lab, ti_le_max=4.2, day_min=0.85, bien_min=0.6, rong_max=1.25):
    """Gộp dần từng cặp vùng kề nhau nếu vùng gộp vẫn giống MỘT viên thuốc:
    biên chung dài cỡ bề rộng viên (vết cắt ngang thân), độ đầy cao, dài/rộng <= ti_le_max,
    và không rộng hơn viên lớn hơn quá rong_max lần (2 viên nằm song song thì không gộp)."""
    lab = lab.copy()
    while True:
        pts = {i: np.nonzero(lab == i) for i in np.unique(lab) if i > 0}
        hd = {i: hinh_dang(*pts[i]) for i in pts}
        tot = None
        for (a, b), nb in _bien_chung(lab).items():
            d, r, day = hinh_dang(np.concatenate([pts[a][0], pts[b][0]]),
                                  np.concatenate([pts[a][1], pts[b][1]]))
            if (nb >= bien_min * min(hd[a][1], hd[b][1]) and day >= day_min
                    and d / r <= ti_le_max and r <= rong_max * max(hd[a][1], hd[b][1])):
                if tot is None or day > tot[0]:
                    tot = (day, a, b)
        if tot is None:
            return lab
        lab[lab == tot[2]] = tot[1]


def separate_vat_dai(mask, small_rgb, q=40, co=2, ti_le_loi=0.2, ti_le_max=4.2):
    """Ảnh nhị phân sau bước 5 + ảnh màu đã thu nhỏ -> ảnh nhị phân đã tách (đường cắt = 0).
    Trả thêm ảnh nhãn và ảnh gradient để hiển thị."""
    mask = mask.astype(bool)
    grad = gradient_mau(small_rgb)
    seeds = hat_giong_loi(mask, grad, q, co, ti_le_loi)
    lab = watershed(-grad, seeds, mask)          # gradient THẤP được loang trước
    lab = gop_hai_nua(lab, ti_le_max)
    edge = np.zeros_like(mask)
    dh = (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] > 0) & (lab[:, 1:] > 0)
    dv = (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] > 0) & (lab[1:, :] > 0)
    edge[:, :-1] |= dh; edge[:, 1:] |= dh
    edge[:-1, :] |= dv; edge[1:, :] |= dv
    return ((lab > 0) & ~edge).astype(np.uint8), lab, grad
