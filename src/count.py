"""TV2 - Bước 1-2: Gán nhãn thành phần liên thông (BFS, 8 lân cận) -> lọc vùng nhiễu -> đếm.

Ý tưởng gán nhãn bằng BFS (tìm kiếm theo chiều rộng):
    Duyệt ảnh từ trên xuống, trái sang phải. Gặp một điểm trắng CHƯA có nhãn:
        - tạo nhãn mới (vật thứ n),
        - cho điểm đó vào hàng đợi,
        - lần lượt lấy điểm ở đầu hàng đợi ra, xét 8 điểm xung quanh; điểm nào trắng và
          chưa có nhãn thì gán cùng nhãn và cho vào cuối hàng đợi,
        - hàng đợi rỗng = đã loang hết vật đó.
    Mỗi điểm được cho vào hàng đợi đúng 1 lần -> độ phức tạp O(số điểm ảnh).

8 lân cận (tính cả đường chéo) hay 4 lân cận?
    4 lân cận: chỉ trên/dưới/trái/phải. Một vật có viền răng cưa (chéo) có thể bị đếm thành nhiều mảnh.
    8 lân cận: thêm 4 đường chéo -> viền chéo vẫn liền. Đổi lại đường cắt giữa 2 vật phải dày
    2 pixel (TV1 đã làm ở separate.py), nếu không 2 vật sẽ "đi chéo" qua đường cắt và bị gộp.
"""
from collections import deque

import numpy as np

from .measure import measure_objects

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def label_components(binary, connectivity=8):
    """Gán nhãn liên thông bằng BFS. Trả về (ảnh_nhãn int32, số_vùng). Nền = 0, vật = 1..n.

    Cài đặt: đệm thêm 1 viền 0 quanh ảnh để không phải kiểm tra "ra ngoài ảnh", rồi làm việc
    trên mảng 1 chiều: điểm (y, x) có chỉ số p = y*W + x, hàng xóm là p + dy*W + dx.
    """
    b = np.asarray(binary).astype(bool)
    h, w = b.shape
    W = w + 2
    pad = np.zeros((h + 2, W), dtype=np.uint8)
    pad[1:-1, 1:-1] = b
    todo = bytearray(pad.tobytes())                 # 1 = điểm vật chưa gán nhãn
    steps = [dy * W + dx for dy, dx in (N8 if connectivity == 8 else N4)]
    lab = np.zeros((h + 2) * W, dtype=np.int32)
    n = 0
    for s in np.flatnonzero(pad).tolist():          # thứ tự quét: trên->dưới, trái->phải
        if not todo[s]:
            continue                                # đã thuộc một vật trước đó
        n += 1
        todo[s] = 0
        queue = deque([s])
        members = [s]
        while queue:
            p = queue.popleft()
            for d in steps:
                q = p + d
                if todo[q]:
                    todo[q] = 0
                    queue.append(q)
                    members.append(q)
        lab[members] = n
    return lab.reshape(h + 2, W)[1:-1, 1:-1].copy(), n


def remove_small(labels, min_area=0, min_area_ratio=0.0):
    """Lọc vùng nhiễu: bỏ vùng có diện tích < max(min_area, min_area_ratio * diện tích vùng lớn nhất).

    Dùng tỉ lệ so với vật lớn nhất vì ảnh trong bộ dữ liệu có kích thước rất khác nhau
    (đồng xu có thể to 20 px hoặc 200 px), một ngưỡng pixel cố định không hợp cho mọi ảnh.
    Các vùng còn lại được đánh số lại liên tục 1..k. Trả về (ảnh_nhãn_mới, k).
    """
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    if areas.max() == 0:
        return np.zeros_like(labels), 0
    limit = max(min_area, min_area_ratio * areas.max())
    keep = np.nonzero((areas >= limit) & (areas > 0))[0]
    new_id = np.zeros(len(areas), dtype=np.int32)
    new_id[keep] = np.arange(1, len(keep) + 1)
    return new_id[labels], len(keep)


def count_objects(binary, min_area=0, min_area_ratio=0.05, connectivity=8, scale=1.0,
                  return_labels=False):
    """Hàm đã thống nhất trong doc chia việc: ảnh nhị phân -> (số lượng, thông tin từng vật).

    scale: tỉ lệ thu nhỏ ảnh ở bước tiền xử lý (pipeline trả về) để đổi kích thước về ảnh gốc.
    return_labels=True: trả thêm ảnh nhãn để vẽ kết quả.
    """
    labels, _ = label_components(binary, connectivity)
    labels, n = remove_small(labels, min_area, min_area_ratio)
    objects = measure_objects(labels, scale)
    if return_labels:
        return n, objects, labels
    return n, objects
