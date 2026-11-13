"""Buoc 2 + 3: gan muc do, chon bo chinh, chia tap chinh/kiem tra.
Dau ra: data/labels.csv  (bo chinh, Twin duyet lai)
        data_tools/out/filtered_pool.csv (tat ca anh dat tieu chi loc)
"""
import os, shutil, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
PER_LEVEL = 30          # so anh moi muc trong bo chinh
TUNE_RATIO = 0.2        # 20% tap chinh tham so

d = pd.read_csv(os.path.join(ROOT, "data_tools/out/all_images.csv"))

# ---- Loc ----
ly_do = np.where(d.so_luong < 2, "chi 1 dong xu",
        np.where(d.so_luong > 30, "qua nhieu xu (de chong len nhau)",
        np.where(np.minimum(d.rong, d.cao) < 150, "anh qua nho", "")))
d["ly_do_loai"] = ly_do
pool = d[d.ly_do_loai == ""].copy()

# ---- Muc do (tu dong, Twin duyet lai) ----
diem = ((pool.nen_gradient > 5).astype(int)          # nen co van / chi tiet
        + (pool.nen_gradient > 9.5).astype(int)      # nen rat lon xon
        + (pool.chenh_sang_goc > 35).astype(int)     # anh sang khong deu
        + (pool.so_luong > 12).astype(int))          # nhieu xu -> de dinh nhau
pool["diem_kho"] = diem
pool["muc_do"] = np.select([diem == 0, diem == 1], ["de", "vua"], "kho")
pool.to_csv(os.path.join(ROOT, "data_tools/out/filtered_pool.csv"), index=False)

# ---- Chon bo chinh: moi muc PER_LEVEL anh, trai deu cac thu muc (nuoc) ----
rng = np.random.default_rng(SEED)
chon = []
for lv in ["de", "vua", "kho"]:
    g = pool[pool.muc_do == lv]
    g = g.sample(frac=1, random_state=SEED)
    # round-robin theo thu_muc de da dang loai xu
    by = [x for _, x in g.groupby("thu_muc")]
    picked, i = [], 0
    while len(picked) < min(PER_LEVEL, len(g)):
        grp = by[i % len(by)]
        k = i // len(by)
        if k < len(grp): picked.append(grp.iloc[k])
        i += 1
        if i > 10000: break
    chon.append(pd.DataFrame(picked))
main = pd.concat(chon)

# ---- Chia tap: 20% chinh / 80% kiem tra, chia deu theo muc do ----
main["tap"] = "kiem_tra"
for lv, g in main.groupby("muc_do"):
    n_tune = int(round(len(g) * TUNE_RATIO))
    idx = g.sample(n_tune, random_state=SEED).index
    main.loc[idx, "tap"] = "chinh"

# ---- Copy anh vao data/images de repo gon gang ----
img_out = os.path.join(ROOT, "data", "images")
os.makedirs(img_out, exist_ok=True)
for r in main.itertuples():
    dst = f"{r.thu_muc.replace('_coins','')}_{r.ten_anh}"
    shutil.copy2(os.path.join(ROOT, r.duong_dan), os.path.join(img_out, dst))
    main.loc[r.Index, "file"] = dst

main["nguon"] = "Kaggle Count Coins (balabaskar)"
main["da_duyet"] = "chua"   # Twin doi thanh 'roi' sau khi xem anh
main["ghi_chu"] = ""
cols = ["file", "so_luong", "muc_do", "tap", "nguon", "thu_muc", "ten_anh",
        "rong", "cao", "do_sang_tb", "do_tuong_phan", "nen_gradient", "chenh_sang_goc",
        "diem_kho", "da_duyet", "ghi_chu"]
main = main[cols].sort_values(["muc_do", "tap", "file"])
main.to_csv(os.path.join(ROOT, "data", "labels.csv"), index=False, encoding="utf-8")

print("Sau loc:", len(pool), "/", len(d))
print(d.ly_do_loai.replace("", "giu lai").value_counts().to_string())
print("Muc do trong pool:", pool.muc_do.value_counts().to_dict())
print("Bo chinh:", len(main)); print(pd.crosstab(main.muc_do, main.tap))
print("So luong xu moi muc:", main.groupby("muc_do").so_luong.describe()[["mean","min","max"]].round(1).to_dict())
