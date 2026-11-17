"""TV2 - Bước 3: Đo thông số từng vật từ ảnh nhãn.

Với vật có nhãn i, gọi S_i là tập điểm ảnh của nó (N_i điểm):
    Diện tích        A   = N_i                                (số điểm ảnh)
    Tâm (trọng tâm)  cx  = trung bình x,  cy = trung bình y
    Khung bao                x_min..x_max, y_min..y_max -> rộng = x_max - x_min + 1
    Đường kính tương đương d = 2 * sqrt(A / pi)           (đường kính hình tròn có cùng diện tích)
    Tỉ lệ khung        rộng / cao                          (đồng xu tròn ~ 1)
    Độ đặc (extent)    A / (rộng * cao)                    (hình tròn = pi/4 ~ 0,785)

Hai chỉ số cuối giúp phát hiện vùng "nghi gộp nhiều xu": vùng gộp 2 xu có tỉ lệ khung ~2
và độ đặc thấp hơn hẳn 0,785.

Mọi phép tính làm một lượt cho tất cả vật bằng np.bincount (đếm có trọng số theo nhãn),
không lặp qua từng vật.
"""
import numpy as np


def measure_objects(labels, scale=1.0):
    """Ảnh nhãn (0 = nền) -> list[dict], mỗi dict là một vật.

    Kích thước được đổi về ảnh GỐC (chia cho scale): ảnh nhãn ở kích thước đã thu nhỏ,
    nhưng người dùng / bước tính mệnh giá cần số đo trên ảnh gốc.
    Khóa: id, dien_tich, tam_x, tam_y, x, y, rong, cao, duong_kinh, ti_le_khung, do_dac.
    """
    labels = np.asarray(labels)
    n = int(labels.max()) if labels.size else 0
    if n == 0:
        return []
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    area = np.bincount(lab, minlength=n + 1).astype(np.float64)
    sum_x = np.bincount(lab, weights=xs, minlength=n + 1)
    sum_y = np.bincount(lab, weights=ys, minlength=n + 1)
    big = labels.shape[0] + labels.shape[1]
    x0 = np.full(n + 1, big); x1 = np.full(n + 1, -1)
    y0 = np.full(n + 1, big); y1 = np.full(n + 1, -1)
    np.minimum.at(x0, lab, xs); np.maximum.at(x1, lab, xs)
    np.minimum.at(y0, lab, ys); np.maximum.at(y1, lab, ys)

    objects = []
    for i in range(1, n + 1):
        if area[i] == 0:
            continue
        w_px, h_px = x1[i] - x0[i] + 1, y1[i] - y0[i] + 1
        a = area[i] / scale ** 2
        objects.append(dict(
            id=len(objects) + 1,
            dien_tich=round(float(a), 1),
            tam_x=round(float(sum_x[i] / area[i] / scale), 1),
            tam_y=round(float(sum_y[i] / area[i] / scale), 1),
            x=int(round(x0[i] / scale)), y=int(round(y0[i] / scale)),
            rong=int(round(w_px / scale)), cao=int(round(h_px / scale)),
            duong_kinh=round(float(2 * np.sqrt(a / np.pi)), 1),
            ti_le_khung=round(float(w_px / h_px), 2),
            do_dac=round(float(area[i] / (w_px * h_px)), 3),
        ))
    return objects


COLUMNS_VI = {"id": "Vật", "dien_tich": "Diện tích (px²)", "tam_x": "Tâm x", "tam_y": "Tâm y",
              "x": "Khung x", "y": "Khung y", "rong": "Rộng (px)", "cao": "Cao (px)",
              "duong_kinh": "Đường kính (px)", "ti_le_khung": "Rộng/Cao", "do_dac": "Độ đặc"}


def objects_table(objects):
    """list[dict] -> bảng pandas có tên cột tiếng Việt (để hiển thị / xuất CSV)."""
    import pandas as pd
    return pd.DataFrame(objects, columns=list(COLUMNS_VI)).rename(columns=COLUMNS_VI)


def suspicious(objects, ratio_max=1.4, extent_min=0.65):
    """Vật nghi là nhiều xu gộp lại / bị cắt hỏng: hình dạng khác xa hình tròn."""
    return [o["id"] for o in objects if not (1 / ratio_max <= o["ti_le_khung"] <= ratio_max)
            or o["do_dac"] < extent_min]
