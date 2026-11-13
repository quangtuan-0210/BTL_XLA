"""TV1 - Bước 6-7: Tách các vật dính nhau (tự cài đặt).

Bước 6 - Biến đổi khoảng cách (distance transform):
    Với mỗi điểm thuộc vật, tính khoảng cách tới điểm nền gần nhất.
    Tâm đồng xu xa nền nhất -> giá trị lớn nhất. Hai xu dính nhau giống 2 "ngọn đồi"
    nối nhau bằng một "yên ngựa" thấp ở chỗ tiếp xúc.
    Cài đặt: thuật toán Chamfer 3-4 (Borgefors, 1986), quét ảnh 2 lượt:
    lượt xuôi (trên->dưới, trái->phải) và lượt ngược. Bước ngang/dọc tốn 3, bước chéo tốn 4
    (xấp xỉ 3*căn 2). Kết quả chia 3 -> khoảng cách tính bằng pixel.

Bước 7 - Watershed (phân thủy) có hạt giống:
    a) Hạt giống = các ĐỈNH đồi (cực đại cục bộ của distance transform), mỗi đỉnh = 1 vật.
    b) "Nước dâng": từ các hạt giống, loang dần ra các điểm lân cận theo thứ tự
       điểm CAO trước (gần tâm) - điểm THẤP sau (gần biên). Khi hai vùng nước gặp nhau
       ở chỗ yên ngựa thì dừng -> đó là đường cắt giữa hai đồng xu.
"""
import heapq
from collections import deque
import numpy as np

from .morphology import dilate, disk


# ---------------- Bước 6: distance transform ----------------
def _pass_rows(d, a, b):
    """Một lượt quét từ trên xuống: lấy giá trị từ hàng trên (+a dọc, +b chéo),
    rồi lan từ trái sang phải trong cùng hàng (+a ngang)."""
    h, w = d.shape
    x = np.arange(w) * a
    for y in range(h):
        if y > 0:
            up = d[y - 1]
            cand = up + a
            cand[1:] = np.minimum(cand[1:], up[:-1] + b)     # chéo trên-trái
            cand[:-1] = np.minimum(cand[:-1], up[1:] + b)    # chéo trên-phải
            d[y] = np.minimum(d[y], cand)
        # lan ngang trái->phải: d[x] = min_j<=x (d[j] + a*(x-j))
        d[y] = np.minimum.accumulate(d[y] - x) + x
    return d


def distance_transform(binary):
    """Khoảng cách (pixel, xấp xỉ Euclid) từ mỗi điểm vật tới điểm nền gần nhất. Nền = 0.
    Ngoài ảnh coi là nền."""
    b = np.pad(binary.astype(bool), 1, constant_values=False)
    INF = 1e9
    d = np.where(b, INF, 0.0)
    d = _pass_rows(d, 3.0, 4.0)                       # lượt xuôi
    d = _pass_rows(d[::-1, ::-1].copy(), 3.0, 4.0)[::-1, ::-1]   # lượt ngược (lật ảnh rồi quét lại)
    return d[1:-1, 1:-1] / 3.0


# ---------------- tiện ích ----------------
def max_filter(img, radius):
    """Lọc cực đại cửa sổ vuông (2r+1)x(2r+1). Làm theo hàng rồi theo cột (tách được)."""
    p = np.pad(img, radius, mode="constant", constant_values=-np.inf)
    h, w = img.shape
    rowmax = np.full((h + 2 * radius, w), -np.inf)
    for j in range(2 * radius + 1):
        rowmax = np.maximum(rowmax, p[:, j:j + w])
    out = np.full((h, w), -np.inf)
    for i in range(2 * radius + 1):
        out = np.maximum(out, rowmax[i:i + h])
    return out


def label_bfs(binary):
    """Gán nhãn thành phần liên thông (8 lân cận). Trả về (ảnh_nhãn, số_vùng).

    Thay vì loang từng pixel (chậm trong Python), làm theo "đoạn chạy" (run):
      1) Mỗi hàng, gom các pixel vật liền nhau thành một đoạn [đầu, cuối].
      2) Đoạn ở hàng dưới chạm (kể cả chéo) đoạn ở hàng trên -> cùng một vật (hợp nhất - union-find).
      3) Gán cho mọi đoạn số hiệu vật của nó.
    Ý tưởng giống hệt BFS, nhưng làm trên vài nghìn đoạn thay vì vài trăm nghìn pixel.
    (Bản đơn giản để TV1 dùng; TV2 có bản đầy đủ kèm thống kê.)"""
    b = binary.astype(bool)
    h, w = b.shape
    pad = np.zeros((h, w + 2), dtype=np.int8); pad[:, 1:-1] = b
    dif = np.diff(pad, axis=1)
    sy, sx = np.nonzero(dif == 1)          # điểm bắt đầu đoạn
    ey, ex = np.nonzero(dif == -1)         # điểm kết thúc (sau phần tử cuối)
    nrun = len(sx)
    parent = list(range(nrun))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    row_start = np.searchsorted(sy, np.arange(h + 1))
    for y in range(1, h):
        i0, i1 = row_start[y - 1], row_start[y]          # các đoạn hàng trên
        j0, j1 = row_start[y], row_start[y + 1]          # các đoạn hàng này
        i = i0
        for j in range(j0, j1):
            while i < i1 and ex[i] < sx[j]:               # đoạn trên kết thúc trước (tính cả chéo)
                i += 1
            k = i
            while k < i1 and sx[k] <= ex[j]:              # đoạn trên chồng lấn (tính cả chéo)
                ra, rb = find(j), find(k)
                if ra != rb: parent[max(ra, rb)] = min(ra, rb)
                k += 1
    roots = np.array([find(i) for i in range(nrun)], dtype=np.int64)
    uniq, ids = np.unique(roots, return_inverse=True)
    lab = np.zeros((h, w), dtype=np.int32)
    for r in range(nrun):
        lab[sy[r], sx[r]:ex[r]] = ids[r] + 1
    return lab, len(uniq)


# ---------------- Bước 7: watershed ----------------
def find_markers(dist, peak_ratio=0.4, min_ratio=0.2, min_radius=3):
    """Tìm hạt giống: điểm cao nhất trong vùng lân cận (bán kính = peak_ratio * đỉnh cao nhất)
    và đủ cao (>= min_ratio * đỉnh cao nhất). Đỉnh nằm sát nhau được gộp làm một."""
    top = dist.max()
    if top <= 0:
        return np.zeros(dist.shape, np.int32), 0
    r = max(min_radius, int(peak_ratio * top))
    peaks = (dist >= max_filter(dist, r)) & (dist >= max(min_ratio * top, min_radius))
    peaks = dilate(peaks, disk(2))            # gộp các điểm đỉnh sát nhau (đỉnh bằng phẳng)
    return label_bfs(peaks)


def watershed(dist, markers, mask):
    """Nước dâng từ hạt giống. Điểm có dist cao được loang trước. Trả về ảnh nhãn."""
    h, w = mask.shape
    lab = markers.copy()
    heap, order = [], 0
    for y, x in zip(*np.nonzero(markers)):
        heapq.heappush(heap, (-dist[y, x], order, y, x)); order += 1
    while heap:
        _, _, y, x = heapq.heappop(heap)
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                lab[ny, nx] = lab[y, x]
                heapq.heappush(heap, (-dist[ny, nx], order, ny, nx)); order += 1
    return lab


def separate(binary, peak_ratio=0.4, min_ratio=0.2, return_labels=False):
    """Ảnh nhị phân -> ảnh nhị phân đã tách vật dính (đường cắt giữa 2 vật = 0).
    return_labels=True: trả thêm ảnh nhãn (mỗi vật một số) và ảnh khoảng cách để hiển thị."""
    mask = binary.astype(bool)
    dist = distance_transform(mask)
    markers, _ = find_markers(dist, peak_ratio, min_ratio)
    lab = watershed(dist, markers, mask)
    # đường cắt: điểm có hàng xóm (trên/dưới/trái/phải) thuộc vật KHÁC -> đổi thành nền.
    # Đánh dấu cả 2 phía nên đường cắt dày 2 pixel: bước đếm dùng 8 lân cận
    # (tính cả đường chéo) cũng không "đi xuyên" qua được.
    edge = np.zeros_like(mask)
    dh = (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] > 0) & (lab[:, 1:] > 0)
    dv = (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] > 0) & (lab[1:, :] > 0)
    edge[:, :-1] |= dh; edge[:, 1:] |= dh
    edge[:-1, :] |= dv; edge[1:, :] |= dv
    out = ((lab > 0) & ~edge).astype(np.uint8)
    if return_labels:
        return out, lab, dist
    return out
