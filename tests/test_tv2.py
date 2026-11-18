"""Kiểm thử phần TV2 bằng ảnh tổng hợp nhỏ (biết trước đáp án).
Chạy từ thư mục gốc dự án:  python tests/test_tv2.py
"""
import os, sys, tempfile
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.count_b8 import label_components, remove_small, count_objects
from src.measure_b8 import measure_objects, suspicious
from src.evaluate import evaluate
from src.visualize import draw_objects

OK = 0
N = 0
def check(name, cond):
    global OK, N
    N += 1
    print(("  PASS " if cond else "  FAIL ") + name)
    OK += bool(cond)


def circle(h, w, cy, cx, r):
    y, x = np.mgrid[:h, :w]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


print("Bước 1 - Gán nhãn BFS")
diag = np.zeros((5, 5), np.uint8); diag[1, 1] = diag[2, 2] = 1
check("2 pixel chéo: 8 lân cận -> 1 vật", label_components(diag, 8)[1] == 1)
check("2 pixel chéo: 4 lân cận -> 2 vật", label_components(diag, 4)[1] == 2)
check("ảnh toàn nền -> 0 vật", label_components(np.zeros((10, 10)))[1] == 0)
full = np.ones((7, 9), np.uint8)
lab, n = label_components(full)
check("ảnh toàn vật (chạm cả 4 mép) -> 1 vật, mọi điểm nhãn 1", n == 1 and np.all(lab == 1))
u = np.zeros((10, 10), np.uint8); u[2:8, 2] = u[2:8, 7] = u[7, 2:8] = 1   # chữ U: 2 nhánh nối ở đáy
check("chữ U (2 nhánh nối ở dưới) -> 1 vật", label_components(u)[1] == 1)
img = (circle(60, 120, 30, 25, 12) | circle(60, 120, 30, 90, 15)).astype(np.uint8)
lab, n = label_components(img)
# quét từ trên xuống: xu phải (r=15) có điểm cao nhất ở hàng 15, xu trái (r=12) ở hàng 18 -> xu phải gặp trước
check("2 hình tròn rời -> 2 vật, nhãn theo thứ tự quét (vật gặp trước = 1)", n == 2 and lab[30, 90] == 1 and lab[30, 25] == 2)

print("Bước 2 - Lọc vùng nhiễu")
noisy = img.copy(); noisy[5, 5] = 1; noisy[50, 60:62] = 1
lab, n = label_components(noisy)
lab2, k = remove_small(lab, min_area_ratio=0.05)
check(f"2 xu + 2 chấm nhiễu: trước lọc {n} vùng, sau lọc {k} vùng", n == 4 and k == 2)
check("sau lọc nhãn được đánh số lại liên tục 1..k", set(np.unique(lab2)) == {0, 1, 2})
check("lọc theo diện tích tuyệt đối (min_area=500) -> chỉ còn xu to", remove_small(lab, min_area=500)[1] == 1)

print("Bước 3 - Đo thông số")
sq = np.zeros((40, 50), np.int32); sq[5:15, 10:30] = 1            # chữ nhật 10 x 20
o = measure_objects(sq)[0]
check("diện tích chữ nhật 10x20 = 200", o["dien_tich"] == 200)
check("khung bao x=10, y=5, rộng 20, cao 10", (o["x"], o["y"], o["rong"], o["cao"]) == (10, 5, 20, 10))
check("tâm = (19.5, 9.5)", (o["tam_x"], o["tam_y"]) == (19.5, 9.5))
check("độ đặc chữ nhật = 1, tỉ lệ khung = 2", o["do_dac"] == 1 and o["ti_le_khung"] == 2)
c = measure_objects(circle(101, 101, 50, 50, 40).astype(np.int32))[0]
check(f"đường kính tương đương hình tròn r=40 ~ 80 (được {c['duong_kinh']})", abs(c["duong_kinh"] - 80) < 1)
check(f"độ đặc hình tròn ~ pi/4 = 0.785 (được {c['do_dac']})", abs(c["do_dac"] - np.pi / 4) < 0.02)
o2 = measure_objects(sq, scale=0.5)[0]
check("scale=0.5 (ảnh đã thu nhỏ 1/2) -> diện tích gốc x4, khung x2", o2["dien_tich"] == 800 and o2["rong"] == 40)
two = (circle(60, 120, 30, 30, 20) | circle(60, 120, 30, 65, 20)).astype(np.int32)
check("2 xu dính chưa tách -> bị đánh dấu nghi gộp", suspicious(measure_objects(two)) == [1])
check("1 xu tròn -> không nghi", suspicious([c]) == [])

print("count_objects (hàm thống nhất)")
n, objs = count_objects(noisy)
check("trả về (2, list 2 dict)", n == 2 and len(objs) == 2 and objs[0]["id"] == 1)
n, objs, lab = count_objects(noisy, return_labels=True)
check("return_labels=True trả thêm ảnh nhãn cùng kích thước", lab.shape == noisy.shape and lab.max() == 2)

print("Bước 4 - Vẽ kết quả")
im = draw_objects(np.full((60, 120, 3), 200, np.uint8), lab)
check("ảnh kết quả cùng kích thước ảnh vào", im.size == (120, 60))

print("Bước 5 - Đánh giá")
lab_csv = pd.DataFrame(dict(file=["a", "b", "c", "d"], so_luong=[2, 4, 5, 10], muc_do=["de", "de", "vua", "kho"]))
pred = pd.DataFrame(dict(file=["a", "b", "c", "d"], dem=[2, 4, 7, 7]))
with tempfile.TemporaryDirectory() as tmp:
    p1, p2 = os.path.join(tmp, "pred.csv"), os.path.join(tmp, "lab.csv")
    pred.to_csv(p1, index=False); lab_csv.to_csv(p2, index=False)
    r = evaluate(p1, p2)
check("tỉ lệ đúng = 2/4 = 50%", r["tong"]["ti_le_dung"] == 0.5)
check("MAE = (0+0+2+3)/4 = 1.25", r["tong"]["mae"] == 1.25)
check("sai lệch TB = (0+0+2-3)/4 = -0.25", r["tong"]["sai_lech_tb"] == -0.25)
check("đếm thiếu 1 ảnh, thừa 1 ảnh", r["tong"]["so_anh_dem_thieu"] == 1 and r["tong"]["so_anh_dem_thua"] == 1)
check("mức Dễ: đúng 100%, mức Khó: MAE 3", r["theo_muc"]["de"]["ti_le_dung"] == 1 and r["theo_muc"]["kho"]["mae"] == 3)
try:
    evaluate(pred.assign(file=["a", "b", "c", "x"]), lab_csv); bad = False
except ValueError:
    bad = True
check("ảnh không có trong đáp án -> báo lỗi (không âm thầm bỏ qua)", bad)

print(f"\nKết quả: {OK}/{N} bài đạt")
sys.exit(0 if OK == N else 1)
