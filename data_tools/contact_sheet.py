"""Ve anh ghep (contact sheet) de duyet nhanh: moi o = 1 anh + ten + so luong + cac chi so.
Cach dung: python contact_sheet.py <csv> <file_png_ra> [so_anh] [cot_loc=gia_tri]
"""
import sys, os, pandas as pd
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_in, png_out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
d = pd.read_csv(csv_in)
for a in sys.argv[4:]:
    k, v = a.split("=")
    d = d[d[k].astype(str) == v]
d = d.sample(min(n, len(d)), random_state=1) if len(d) > n else d
T, cols = 220, 8
rows = (len(d) + cols - 1) // cols
sheet = Image.new("RGB", (cols * T, rows * (T + 30)), "white")
dr = ImageDraw.Draw(sheet)
for i, r in enumerate(d.itertuples()):
    im = Image.open(os.path.join(ROOT, r.duong_dan)).convert("RGB")
    im.thumbnail((T - 6, T - 6))
    x, y = (i % cols) * T, (i // cols) * (T + 30)
    sheet.paste(im, (x + 3, y + 3))
    lab = f"{r.ten_anh[:6]} n={r.so_luong}"
    if hasattr(r, "muc_do"): lab += f" {r.muc_do}"
    dr.text((x + 4, y + T + 2), lab, fill="black")
    dr.text((x + 4, y + T + 15), f"g={r.nen_gradient} s={r.chenh_sang_goc}", fill="gray")
sheet.save(png_out)
print("saved", png_out, len(d))
