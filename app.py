"""Giao diện demo (TV3) – Ứng dụng phát hiện và đếm đối tượng trong ảnh.

Chạy:  streamlit run app.py
Thuật toán gọi từ src/pipeline.py (tự cài đặt bằng NumPy). Streamlit chỉ lo phần hiển thị.
"""
import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.pipeline import DEFAULT, run
from src.preprocess import resize_max

ROOT = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Đếm đối tượng trong ảnh", layout="wide")


# ---------------- tiện ích hiển thị ----------------
def object_table(labels, scale):
    """Thông tin từng vật: diện tích, tâm, khung bao, đường kính tương đương (đổi về pixel ảnh gốc)."""
    rows = []
    for i in range(1, labels.max() + 1):
        ys, xs = np.nonzero(labels == i)
        if len(ys) == 0:
            continue
        area = len(ys) / scale ** 2
        rows.append(dict(vat=len(rows) + 1, tam_x=int(xs.mean() / scale), tam_y=int(ys.mean() / scale),
                         dien_tich_px=int(area), duong_kinh_px=round(2 * np.sqrt(area / np.pi), 1),
                         rong_px=int((xs.max() - xs.min() + 1) / scale), cao_px=int((ys.max() - ys.min() + 1) / scale)))
    return pd.DataFrame(rows).rename(columns={
        "vat": "Vật", "tam_x": "Tâm x", "tam_y": "Tâm y", "dien_tich_px": "Diện tích (px²)",
        "duong_kinh_px": "Đường kính (px)", "rong_px": "Rộng (px)", "cao_px": "Cao (px)"})


def overlay(img_small, labels):
    """Tô màu nhẹ từng vật, vẽ viền và số thứ tự lên ảnh."""
    rng = np.random.default_rng(7)
    pal = rng.integers(40, 255, (labels.max() + 1, 3))
    base = img_small.astype(np.float64)
    m = labels > 0
    base[m] = 0.55 * base[m] + 0.45 * pal[labels[m]]
    edge = np.zeros_like(m)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    base[edge & (m | np.roll(m, 1, 0) | np.roll(m, 1, 1))] = (255, 40, 40)
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    k = 0
    for i in range(1, labels.max() + 1):
        ys, xs = np.nonzero(labels == i)
        if len(ys) == 0:
            continue
        k += 1
        cx, cy = xs.mean(), ys.mean()
        dr.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=(255, 255, 255), outline=(20, 20, 20))
        dr.text((cx, cy), str(k), fill=(0, 0, 0), anchor="mm")
    return im


def to_uint8(a):
    a = np.asarray(a, dtype=np.float64)
    if a.max() <= 1:
        a = a * 255
    elif a.max() > 255:
        a = a * 255 / a.max()
    return np.clip(a, 0, 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def run_cached(img_bytes, params, use_separate):
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    t = time.time()
    res = run(img, use_separate=use_separate, **params)
    res["seconds"] = time.time() - t
    res["small"] = resize_max(img, params["max_side"])[0].astype(np.uint8)
    return res


# ---------------- thanh bên: chọn ảnh + tham số ----------------
st.sidebar.header("1. Chọn ảnh")
src = st.sidebar.radio("Nguồn ảnh", ["Tải ảnh lên", "Ảnh mẫu trong bộ dữ liệu"])
img_bytes, truth = None, None
if src == "Tải ảnh lên":
    up = st.sidebar.file_uploader("Ảnh JPG / PNG", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if up:
        img_bytes = up.getvalue()
else:
    lab = pd.read_csv(os.path.join(ROOT, "data", "labels.csv"))
    level = st.sidebar.selectbox("Mức độ", ["de", "vua", "kho"], format_func={"de": "Dễ", "vua": "Vừa", "kho": "Khó"}.get)
    sub = lab[lab.muc_do == level]
    f = st.sidebar.selectbox("Ảnh", sub.file.tolist())
    with open(os.path.join(ROOT, "data", "images", f), "rb") as fh:
        img_bytes = fh.read()
    truth = int(sub[sub.file == f].so_luong.iloc[0])

st.sidebar.header("2. Tham số")
p = dict(DEFAULT)
p["mode"] = st.sidebar.radio("Cách tách nền", ["color", "gray"],
                             format_func={"color": "Độ khác màu nền (khuyên dùng)", "gray": "Ảnh xám (Otsu cổ điển)"}.get)
use_sep = st.sidebar.checkbox("Tách vật dính nhau (watershed)", value=True)
with st.sidebar.expander("Tham số nâng cao"):
    p["max_side"] = st.slider("Thu nhỏ ảnh về cạnh dài (px)", 256, 1024, DEFAULT["max_side"], 64)
    p["blur_size"] = st.select_slider("Kích thước lọc Gauss", [1, 3, 5, 7, 9], DEFAULT["blur_size"])
    p["median_size"] = st.select_slider("Lọc trung vị (0 = tắt)", [0, 3, 5], DEFAULT["median_size"])
    p["close_r"] = st.slider("Bán kính phép đóng", 0, 8, DEFAULT["close_r"])
    p["open_r"] = st.slider("Bán kính phép mở", 0, 8, DEFAULT["open_r"])
    p["peak_ratio"] = st.slider("Cửa sổ tìm đỉnh (tỉ lệ)", 0.1, 0.8, DEFAULT["peak_ratio"], 0.05)
    p["min_ratio"] = st.slider("Độ cao tối thiểu của đỉnh (tỉ lệ)", 0.05, 0.6, DEFAULT["min_ratio"], 0.05)
    p["min_area_ratio"] = st.slider("Bỏ vùng nhỏ hơn (tỉ lệ vật lớn nhất)", 0.0, 0.3, DEFAULT["min_area_ratio"], 0.01)

# ---------------- nội dung chính ----------------
st.title("Ứng dụng phát hiện và đếm đối tượng trong ảnh")
st.caption("Thuật toán xử lý ảnh tự cài đặt bằng NumPy: tiền xử lý → Otsu → hình thái học → watershed → đếm.")

if img_bytes is None:
    st.info("Chọn một ảnh ở thanh bên trái để bắt đầu. Ứng dụng hợp nhất với vật tách biệt "
            "(đồng xu, viên thuốc, cúc áo, tế bào) trên nền tương đối đồng nhất.")
    st.stop()

with st.spinner("Đang xử lý..."):
    res = run_cached(img_bytes, p, use_sep)

n = res["count"]
c1, c2, c3 = st.columns(3)
c1.metric("Số vật đếm được", n)
if truth is not None:
    c2.metric("Đáp án", truth, delta=n - truth, delta_color="inverse" if n != truth else "off")
c3.metric("Thời gian xử lý", f"{res['seconds']:.2f} s")
if truth is not None:
    (st.success if n == truth else st.warning)("Đếm đúng." if n == truth else f"Lệch {n - truth:+d} so với đáp án.")

left, right = st.columns(2)
left.image(res["small"], caption="Ảnh gốc", width="stretch")
ov = overlay(res["small"], res["labels"])
right.image(ov, caption=f"Kết quả: {n} vật", width="stretch")
buf = io.BytesIO(); ov.save(buf, format="PNG")
right.download_button("Tải ảnh kết quả", buf.getvalue(), "ket_qua.png", "image/png")

st.subheader("Các bước xử lý")
steps = [("Xám hóa + lọc Gauss", res["gray"]),
         (f"Tách nền (ngưỡng {res['threshold']})", res["binary"]),
         ("Hình thái học", res["cleaned"])]
if res["dist"] is not None:
    steps += [("Biến đổi khoảng cách", res["dist"] * 255 / max(res["dist"].max(), 1e-9)), ("Sau watershed", res["separated"])]
cols = st.columns(len(steps))
for col, (name, im) in zip(cols, steps):
    col.image(to_uint8(im), caption=name, width="stretch", clamp=True)

st.subheader("Thông tin từng vật")
tbl = object_table(res["labels"], res["scale"])
st.dataframe(tbl, hide_index=True, width="stretch")
st.download_button("Tải bảng (CSV)", tbl.to_csv(index=False).encode("utf-8-sig"), "thong_tin_vat.csv", "text/csv")
