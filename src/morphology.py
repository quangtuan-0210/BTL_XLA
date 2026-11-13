"""TV1 - Bước 5: Hình thái học trên ảnh nhị phân (tự cài đặt).

Phần tử cấu trúc (structuring element, SE) là một "con dấu" nhỏ, ví dụ hình tròn 5x5.
    Co (erosion)  : điểm giữ lại là 1 chỉ khi con dấu đặt ở đó NẰM TRỌN trong vật -> vật nhỏ lại
    Giãn (dilation): điểm thành 1 nếu con dấu đặt ở đó CHẠM vào vật -> vật to ra
    Mở (opening)  = co rồi giãn  : xóa chấm trắng nhỏ, cắt cầu nối mảnh giữa hai vật
    Đóng (closing)= giãn rồi co  : lấp khe nứt, lỗ nhỏ bên trong vật
    Lấp lỗ (fill_holes): lấp toàn bộ lỗ bên trong vật (ví dụ lỗ giữa đồng 5 yên)
"""
import numpy as np


def disk(radius):
    """Phần tử cấu trúc hình tròn bán kính r (ma trận (2r+1)x(2r+1) gồm 0/1)."""
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2).astype(np.uint8)


def _shifts(se):
    """Danh sách độ dời (dy, dx) ứng với các ô bằng 1 trong phần tử cấu trúc."""
    r0, c0 = se.shape[0] // 2, se.shape[1] // 2
    return [(i - r0, j - c0) for i, j in zip(*np.nonzero(se))]


def erode(binary, se):
    """Co: AND của các bản dịch ảnh theo từng ô của con dấu. Ngoài ảnh coi là nền (0)."""
    b = binary.astype(bool)
    r = max(se.shape) // 2
    p = np.pad(b, r, constant_values=False)
    h, w = b.shape
    out = np.ones_like(b)
    for dy, dx in _shifts(se):
        out &= p[r + dy:r + dy + h, r + dx:r + dx + w]
    return out.astype(np.uint8)


def dilate(binary, se):
    """Giãn: OR của các bản dịch ảnh theo từng ô của con dấu."""
    b = binary.astype(bool)
    r = max(se.shape) // 2
    p = np.pad(b, r, constant_values=False)
    h, w = b.shape
    out = np.zeros_like(b)
    for dy, dx in _shifts(se):
        out |= p[r - dy:r - dy + h, r - dx:r - dx + w]
    return out.astype(np.uint8)


def opening(binary, se):
    return dilate(erode(binary, se), se)


def closing(binary, se):
    return erode(dilate(binary, se), se)


def fill_holes(binary):
    """Lấp lỗ: gán nhãn các vùng NỀN. Vùng nền nào chạm viền ảnh là nền thật;
    vùng nền nào không chạm viền là nằm kín bên trong vật -> đó là lỗ -> đổi thành vật."""
    from .separate import label_bfs           # dùng lại hàm gán nhãn (tránh import vòng)
    b = binary.astype(bool)
    lab, _ = label_bfs(~b)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = (~b) & ~np.isin(lab, border)
    return (b | holes).astype(np.uint8)


def clean(binary, open_r=2, close_r=3, fill=True):
    """Gộp bước làm sạch: đóng (nối nét đứt) -> lấp lỗ -> mở (xóa vụn nhỏ)."""
    b = binary
    if close_r: b = closing(b, disk(close_r))
    if fill: b = fill_holes(b)
    if open_r: b = opening(b, disk(open_r))
    return b
