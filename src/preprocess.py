"""TV1 - Bước 1-3: Tiền xử lý ảnh (tự cài đặt bằng NumPy, không dùng hàm xử lý ảnh của OpenCV).

Các hàm:
    to_gray        : ảnh màu  -> ảnh xám
    resize_max     : thu nhỏ ảnh để cạnh dài nhất <= max_side (nội suy song tuyến)
    conv2d         : tích chập 2D tổng quát (dùng chung cho mọi bộ lọc)
    gaussian_kernel: sinh ma trận lọc Gauss
    gaussian_blur  : làm mịn bằng lọc Gauss
    median_filter  : lọc trung vị (xóa nhiễu muối tiêu)
    preprocess     : gộp các bước trên - đúng tên hàm đã thống nhất trong doc chia việc
"""
import numpy as np


def to_gray(img):
    """Ảnh màu RGB (H, W, 3) -> ảnh xám (H, W), kiểu float 0..255.

    Mắt người nhạy với màu xanh lá nhất, rồi đỏ, rồi xanh dương,
    nên mỗi kênh có trọng số khác nhau (chuẩn ITU-R BT.601):
        Xám = 0.299*R + 0.587*G + 0.114*B
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:                 # đã là ảnh xám
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def resize_max(img, max_side=512):
    """Thu nhỏ ảnh sao cho cạnh dài nhất = max_side (giữ tỉ lệ). Ảnh đã nhỏ thì giữ nguyên.

    Nội suy song tuyến (bilinear): giá trị một điểm mới = trung bình có trọng số
    của 4 điểm ảnh gốc xung quanh nó. Điểm nào gần hơn thì trọng số lớn hơn.
    Trả về (ảnh_mới, tỉ_lệ).
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img, 1.0
    nh, nw = int(round(h * scale)), int(round(w * scale))
    # vị trí (số thực) trong ảnh gốc ứng với mỗi hàng/cột của ảnh mới
    ys = (np.arange(nh) + 0.5) / scale - 0.5
    xs = (np.arange(nw) + 0.5) / scale - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]      # khoảng cách tới hàng trên
    wx = np.clip(xs - x0, 0, 1)[None, :]      # khoảng cách tới cột trái
    if img.ndim == 3:
        wy, wx = wy[..., None], wx[..., None]
    a = img[y0][:, x0]; b = img[y0][:, x1]    # 2 điểm hàng trên
    c = img[y1][:, x0]; d = img[y1][:, x1]    # 2 điểm hàng dưới
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return top * (1 - wy) + bot * wy, scale


def conv2d(img, kernel):
    """Tích chập 2D: mỗi điểm ảnh mới = tổng (giá trị lân cận x trọng số của kernel).

    Cách làm: đệm viền ảnh (padding kiểu phản chiếu để viền không bị tối),
    rồi với MỖI ô của kernel, dịch cả ảnh đi một đoạn tương ứng, nhân với trọng số
    và cộng dồn. Làm vậy thay vì 2 vòng lặp qua từng pixel nên chạy nhanh hơn rất nhiều,
    nhưng kết quả giống hệt định nghĩa.
    """
    img = np.asarray(img, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)[::-1, ::-1]   # lật kernel = định nghĩa tích chập
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    h, w = img.shape
    out = np.zeros((h, w))
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * padded[i:i + h, j:j + w]
    return out


def gaussian_kernel(size=5, sigma=1.0):
    """Ma trận lọc Gauss kích thước size x size.

    G(x, y) = exp(-(x^2 + y^2) / (2*sigma^2)), rồi chia cho tổng để các trọng số cộng lại = 1
    (nhờ vậy ảnh không bị sáng lên hay tối đi). Điểm ở giữa có trọng số lớn nhất.
    """
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(img, size=5, sigma=1.0):
    """Làm mịn ảnh: thay mỗi điểm bằng trung bình có trọng số Gauss của vùng xung quanh."""
    return conv2d(img, gaussian_kernel(size, sigma))


def median_filter(img, size=3):
    """Lọc trung vị: thay mỗi điểm bằng giá trị ĐỨNG GIỮA của vùng size x size.

    Khác với lọc Gauss (lấy trung bình), lọc trung vị loại bỏ hẳn các chấm nhiễu lẻ
    mà không làm nhòe đường biên của đồng xu.
    """
    img = np.asarray(img, dtype=np.float64)
    r = size // 2
    padded = np.pad(img, r, mode="reflect")
    h, w = img.shape
    # xếp chồng size*size bản dịch của ảnh, rồi lấy trung vị theo chiều xếp chồng
    stack = np.stack([padded[i:i + h, j:j + w] for i in range(size) for j in range(size)])
    return np.median(stack, axis=0)


def preprocess(img_rgb, max_side=512, blur_size=5, sigma=1.0, median_size=0):
    """Gộp bước tiền xử lý: thu nhỏ -> xám hóa -> (lọc trung vị) -> làm mịn Gauss.

    Trả về (ảnh_xám_đã_làm_mịn, tỉ_lệ_thu_nhỏ). tỉ lệ dùng để đổi kích thước về ảnh gốc.
    """
    small, scale = resize_max(img_rgb, max_side)
    gray = to_gray(small)
    if median_size and median_size > 1:
        gray = median_filter(gray, median_size)
    if blur_size and blur_size > 1:
        gray = gaussian_blur(gray, blur_size, sigma)
    return np.clip(gray, 0, 255), scale
