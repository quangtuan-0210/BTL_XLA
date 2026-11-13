"""Buoc 6: ap dung ket qua duyet bang mat (05/11/2026).
Bo anh xau, sua muc do, them anh bu, chia lai tap chinh/kiem tra.
Ghi lai moi quyet dinh vao data_tools/out/review_log.csv
"""
import os, shutil, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, TUNE = 42, 0.2
main = pd.read_csv(os.path.join(ROOT, "data/labels.csv"))
cand = pd.read_csv(os.path.join(ROOT, "data_tools/out/candidates.csv"))

BO = {  # ma 6 ky tu -> ly do
 "e514e1": "xu qua nho, kem chu", "200798": "xu bi cat o mep anh", "1d5fb7": "anh chup trang web",
 "9b1df4": "trung voi 767455", "d8e50e": "hinh minh hoa co chu", "3c1e9b": "tay cam xu",
 "6e069d": "trung voi 50c2d6", "0d493d": "anh ghep", "89c6f2": "anh quang cao co chu",
 "716dcf": "anh 3D, xu nghieng", "abd33e": "co nhan gia che xu", "f560ce": "xu xep chong, kho dem",
 "78b584": "anh ghep 2 khung", "c6aeae": "nguoi cam the xu", "efc291": "xu xep chong",
 "bfc5c8": "anh ghep nhieu khung", "f8acdd": "luoi xu co chu", "f7b627": "xu bi cat o mep",
 "9f842d": "album, co vach mau", "77d310": "xu lan trong tien giay", "2bf5e8": "hinh minh hoa co chu",
 "766f50": "poster", "1175dc": "trang sach", "3faaad": "xu bi cat o mep",
 "62e09c": "hop trung bay, xu qua nho", "9a6e4e": "anh do hoa",
}
MUC = {
 "de":  "329024 7ea2d9 767455 6b8399 7f1965 2f1a25 630daf 5f344d d954b3 e5b36a 432ce0 d0a8b1 d69749 "
        "0871f0 43a864 5a5bc5 d1fc6b 15e766 9d481c 32f018 5a93f5 c2d044 e42293",
 "vua": "ab19a4 50c2d6 4ac89e 82abf4 d5276b 557dc9 c5cf6a 23560e bb8b54 7dc2f1 82ace9 06742c 698f85 "
        "f7974d 069f87 e7b6e3 b098de 625ce7 e982d8 2f20e6 bb4ab4",
 "kho": "3a31d6 4a00c6 857e44 6e1261 14080c 59cd7e a11e78 21ad30 dce60d 48dc01 dee37d f48696 162ed8 "
        "d61f49 069601 24fc83 80840f ca589b 01685f e16bf3",
}
BU = {"de": [23, 41, 48, 49, 53, 56, 59], "vua": [7, 11, 12, 16, 22, 28, 31, 42, 46],
      "kho": [10, 14, 24, 25, 27, 29, 36, 44, 60, 64]}
LY_DO_MUC = {"de": "xu roi nhau, nen tron", "vua": "xu cham nhau hoac nen mau/co bong",
             "kho": "xu chong nhau hoac nen van (go, vai, giay)"}

main["ma"] = main.ten_anh.str[:6]
log = []
for ma, ly in BO.items():
    r = main[main.ma == ma].iloc[0]; log.append([r.file, r.muc_do, "bo", ly])
for lv, s in MUC.items():
    for ma in s.split():
        r = main[main.ma == ma].iloc[0]
        log.append([r.file, r.muc_do, lv, "giu" if r.muc_do == lv else "doi muc: " + LY_DO_MUC[lv]])
assert len(log) == len(main), (len(log), len(main))

# anh bi bo -> chuyen sang data/_bo (khong xoa)
os.makedirs(os.path.join(ROOT, "data/_bo"), exist_ok=True)
bo_files = set(main[main.ma.isin(BO)].file)
for f in bo_files:
    p = os.path.join(ROOT, "data/images", f)
    if os.path.exists(p): shutil.move(p, os.path.join(ROOT, "data/_bo", f))
newmain = main[~main.ma.isin(BO)].copy()
lvmap = {ma: lv for lv, s in MUC.items() for ma in s.split()}
newmain["muc_do"] = newmain.ma.map(lvmap)

# them anh bu
add = []
for lv, ids in BU.items():
    for i in ids:
        r = cand[cand.stt == i].iloc[0].to_dict()
        f = f"{r['thu_muc'].replace('_coins', '')}_{r['ten_anh']}"
        shutil.copy2(os.path.join(ROOT, r["duong_dan"]), os.path.join(ROOT, "data/images", f))
        r.update(file=f, muc_do=lv, nguon="Kaggle Count Coins (balabaskar)")
        add.append(r); log.append([f, "(moi)", lv, "anh bu: " + LY_DO_MUC[lv]])
newmain = pd.concat([newmain, pd.DataFrame(add)], ignore_index=True)

# chia lai tap
newmain["tap"] = "kiem_tra"
for lv, g in newmain.groupby("muc_do"):
    newmain.loc[g.sample(int(round(len(g) * TUNE)), random_state=SEED).index, "tap"] = "chinh"
newmain["da_duyet"] = "roi"
newmain["ghi_chu"] = ""
cols = ["file", "so_luong", "muc_do", "tap", "nguon", "thu_muc", "ten_anh", "rong", "cao", "do_sang_tb",
        "do_tuong_phan", "nen_gradient", "chenh_sang_goc", "diem_kho", "da_duyet", "ghi_chu"]
newmain = newmain[cols].sort_values(["muc_do", "tap", "file"])
newmain.to_csv(os.path.join(ROOT, "data/labels.csv"), index=False, encoding="utf-8")
pd.DataFrame(log, columns=["file", "muc_cu", "quyet_dinh", "ly_do"]).to_csv(
    os.path.join(ROOT, "data_tools/out/review_log.csv"), index=False, encoding="utf-8")
print(pd.crosstab(newmain.muc_do, newmain.tap, margins=True))
print("anh trong images/:", len(os.listdir(os.path.join(ROOT, "data/images"))), "| anh trong _bo/:", len(os.listdir(os.path.join(ROOT, "data/_bo"))))
