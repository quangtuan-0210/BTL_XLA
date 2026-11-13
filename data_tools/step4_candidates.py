"""Buoc 5: lay anh ung vien bu vao cho anh bi loai, bo anh gan-trung (aHash)."""
import os, numpy as np, pandas as pd
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pool = pd.read_csv(os.path.join(ROOT, "data_tools/out/filtered_pool.csv"))
main = pd.read_csv(os.path.join(ROOT, "data/labels.csv"))
def ahash(p):
    a = np.asarray(Image.open(os.path.join(ROOT, p)).convert("L").resize((16, 16)), dtype=float)
    return (a > a.mean()).ravel()
pool["h"] = pool.duong_dan.map(ahash)
used = set(main.ten_anh)
seen = [h for n, h in zip(pool.ten_anh, pool.h) if n in used]
keep = []
for r in pool[~pool.ten_anh.isin(used)].sample(frac=1, random_state=3).itertuples():
    if all((r.h != s).sum() > 20 for s in seen):
        keep.append(r.Index); seen.append(r.h)
rest = pool.loc[keep]
cand = pd.concat([rest[rest.muc_do == "kho"].head(26), rest[rest.muc_do == "vua"].head(22), rest[rest.muc_do == "de"].head(16)])
cand = cand.drop(columns="h").reset_index(drop=True); cand.insert(0, "stt", range(1, len(cand) + 1))
cand.to_csv(os.path.join(ROOT, "data_tools/out/candidates.csv"), index=False)
T, cols = 220, 8; rows = (len(cand) + cols - 1) // cols
from PIL import ImageDraw
sh = Image.new("RGB", (cols * T, rows * (T + 18)), "white"); dr = ImageDraw.Draw(sh)
for i, r in cand.iterrows():
    im = Image.open(os.path.join(ROOT, r.duong_dan)).convert("RGB"); im.thumbnail((T - 6, T - 6))
    x, y = (i % cols) * T, (i // cols) * (T + 18); sh.paste(im, (x + 3, y + 3))
    dr.text((x + 4, y + T + 2), f"#{r.stt}  n={r.so_luong}", fill="black")
sh.save(os.path.join(ROOT, "data_tools/out/ung_vien.png")); print(len(rest), "anh con lai (sau bo gan-trung)", len(cand), "ung vien")
