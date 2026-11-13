"""Buoc 4: phan tich dac diem du lieu cho bao cao.
Dau ra: data_tools/out/report/ (bang .csv + bieu do .png + tom_tat.md)
"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "data_tools/out/report"); os.makedirs(R, exist_ok=True)
allimg = pd.read_csv(os.path.join(ROOT, "data_tools/out/all_images.csv"))
pool = pd.read_csv(os.path.join(ROOT, "data_tools/out/filtered_pool.csv"))
main = pd.read_csv(os.path.join(ROOT, "data/labels.csv"))
ORDER = ["de", "vua", "kho"]; NAME = {"de": "Dễ", "vua": "Vừa", "kho": "Khó"}

# Bang 1: qua trinh loc
loc = pd.DataFrame([
    ["Ảnh trong file CSV gốc", 1444],
    ["Sau khi bỏ ảnh trùng", len(allimg)],
    ["Sau khi lọc (2–30 xu, cạnh ngắn ≥150px)", len(pool)],
    ["Bộ chính (30 ảnh/mức)", len(main)],
], columns=["Bước", "Số ảnh"])
loc.to_csv(f"{R}/bang1_qua_trinh_loc.csv", index=False)

# Bang 2: thong ke bo chinh theo muc do
g = main.groupby("muc_do")
b2 = pd.DataFrame({
    "Số ảnh": g.size(),
    "Tập chỉnh": g.apply(lambda x: (x.tap == "chinh").sum()),
    "Tập kiểm tra": g.apply(lambda x: (x.tap == "kiem_tra").sum()),
    "Tổng số xu": g.so_luong.sum(),
    "Xu/ảnh TB": g.so_luong.mean().round(1),
    "Xu/ảnh min–max": g.so_luong.agg(lambda s: f"{s.min()}–{s.max()}"),
    "Độ sáng TB": g.do_sang_tb.mean().round(1),
    "Độ phức tạp nền TB": g.nen_gradient.mean().round(2),
    "Chênh sáng 4 góc TB": g.chenh_sang_goc.mean().round(1),
    "Kích thước TB (px)": g.apply(lambda x: f"{int(x.rong.mean())}×{int(x.cao.mean())}"),
}).reindex(ORDER).rename(index=NAME)
b2.to_csv(f"{R}/bang2_thong_ke_bo_chinh.csv")

# Bang 3: theo loai xu (nuoc)
b3 = main.groupby("thu_muc").agg(so_anh=("file", "size"), tong_xu=("so_luong", "sum")).sort_values("so_anh", ascending=False)
b3.to_csv(f"{R}/bang3_theo_loai_xu.csv")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = {"de": "#4C9F70", "vua": "#E0A030", "kho": "#C8553D"}

# Hinh 1: phan bo so xu moi anh theo muc do
fig, ax = plt.subplots(figsize=(7, 3.6))
bins = range(2, main.so_luong.max() + 2)
ax.hist([main[main.muc_do == lv].so_luong for lv in ORDER], bins=bins, stacked=True,
        color=[C[l] for l in ORDER], label=[NAME[l] for l in ORDER], edgecolor="white")
ax.set_xlabel("Số đồng xu trong ảnh"); ax.set_ylabel("Số ảnh"); ax.legend(frameon=False)
ax.set_title("Phân bố số xu mỗi ảnh (bộ chính, 90 ảnh)", loc="left")
fig.tight_layout(); fig.savefig(f"{R}/hinh1_phan_bo_so_xu.png", dpi=150); plt.close(fig)

# Hinh 2: do sang va do phuc tap nen theo muc do
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, col, t in zip(axs, ["do_sang_tb", "nen_gradient", "chenh_sang_goc"],
                      ["Độ sáng trung bình (0–255)", "Độ phức tạp nền (gradient viền)", "Chênh sáng giữa 4 góc"]):
    ax.boxplot([main[main.muc_do == lv][col] for lv in ORDER], tick_labels=[NAME[l] for l in ORDER],
               patch_artist=True, boxprops=dict(facecolor="#dde7f0"))
    ax.set_title(t, fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(f"{R}/hinh2_dac_trung_theo_muc.png", dpi=150); plt.close(fig)

# Hinh 3: kich thuoc anh
fig, ax = plt.subplots(figsize=(6, 4))
for lv in ORDER:
    s = main[main.muc_do == lv]; ax.scatter(s.rong, s.cao, s=18, color=C[lv], label=NAME[lv], alpha=.8)
ax.set_xlabel("Chiều rộng (px)"); ax.set_ylabel("Chiều cao (px)"); ax.legend(frameon=False)
ax.set_title("Kích thước ảnh rất khác nhau → cần chuẩn hóa kích thước", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/hinh3_kich_thuoc_anh.png", dpi=150); plt.close(fig)

# Hinh 4: anh vi du moi muc (4 anh/muc)
fig, axs = plt.subplots(3, 4, figsize=(10, 7.5))
for i, lv in enumerate(ORDER):
    s = main[main.muc_do == lv].sample(4, random_state=7)
    for j, r in enumerate(s.itertuples()):
        axs[i, j].imshow(Image.open(os.path.join(ROOT, "data/images", r.file)).convert("RGB"))
        axs[i, j].set_title(f"{NAME[lv]} · {r.so_luong} xu", fontsize=9); axs[i, j].axis("off")
fig.tight_layout(); fig.savefig(f"{R}/hinh4_anh_vi_du.png", dpi=130); plt.close(fig)

with open(f"{R}/tom_tat.md", "w", encoding="utf-8") as f:
    f.write("# Tóm tắt dữ liệu\n\n" + loc.to_markdown(index=False) + "\n\n" + b2.to_markdown() + "\n")
print(loc.to_string(index=False)); print(b2.to_string())
