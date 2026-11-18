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
from PIL import Image

from src.pipeline import DEFAULT, run
from src.preprocess import resize_max
from src.measure import objects_table, suspicious      # TV2: bảng thông số từng vật
from src.visualize import draw_objects                  # TV2: vẽ kết quả

ROOT = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Đếm đối tượng trong ảnh", layout="wide")


# ---------------- tiện ích hiển thị ----------------


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

st.sidebar.header("2. Tùy chọn")
p = dict(DEFAULT)
p["mode"] = st.sidebar.radio("Cách tách nền", ["color", "gray"],
                             format_func={"color": "Độ khác màu nền (khuyên dùng)", "gray": "Ảnh xám (Otsu cổ điển)"}.get)
use_sep = st.sidebar.checkbox("Tách vật dính nhau (watershed)", value=True)

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
nghi = suspicious(res["objects"])
ov = draw_objects(res["small"], res["labels"], suspicious_ids=nghi)
right.image(ov, caption=f"Kết quả: {n} vật" + (f" (số đỏ = nghi gộp nhiều vật: {nghi})" if nghi else ""),
            width="stretch")
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
tbl = objects_table(res["objects"])
st.dataframe(tbl, hide_index=True, width="stretch")
st.caption("Đơn vị: px = điểm ảnh, đo trên ảnh đã thu nhỏ (cạnh dài 512 px); tọa độ tính từ góc trên-trái. "
           "Rộng/Cao là tỉ số (≈ 1 với vật tròn); Độ đặc = diện tích / (rộng × cao), từ 0 đến 1 (hình tròn ≈ 0,785).")
st.download_button("Tải bảng (CSV)", tbl.to_csv(index=False).encode("utf-8-sig"), "thong_tin_vat.csv", "text/csv")
