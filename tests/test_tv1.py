"""Kiểm thử phần TV1 bằng ảnh tổng hợp nhỏ (biết trước đáp án).
Chạy từ thư mục gốc dự án:  python tests/test_tv1.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess_b1_3 import to_gray, gaussian_kernel, conv2d
from src.segment_b2_4 import otsu_threshold, segment
from src.morphology_b5 import disk, erode, dilate, closing, fill_holes
from src.separate_b6_7 import distance_transform, label_bfs, separate

OK = 0
N = 0
def check(name, cond):
    global OK, N
    N += 1
    print(("  PASS " if cond else "  FAIL ") + name)
    OK += bool(cond)
    return cond


def hai_xu(khoang_cach):
    """Hai hình tròn bán kính 30, tâm cách nhau `khoang_cach` px (< 60 là dính nhau)."""
    y, x = np.mgrid[:100, :180]
    a = (x - 50) ** 2 + (y - 50) ** 2 <= 30 ** 2
    b = (x - 50 - khoang_cach) ** 2 + (y - 50) ** 2 <= 30 ** 2
    return (a | b).astype(np.uint8)


print("Bước 1 - Xám hóa")
check("đỏ thuần (255,0,0) -> 76.245", np.isclose(to_gray(np.array([[[255, 0, 0]]]))[0, 0], 76.245))
check("trắng -> 255", np.isclose(to_gray(np.full((2, 2, 3), 255))[0, 0], 255))

print("Bước 2-3 - Tích chập, Gauss")
img = np.random.default_rng(0).integers(0, 256, (20, 30)).astype(float)
I = np.zeros((3, 3)); I[1, 1] = 1
check("conv2d với kernel đơn vị giữ nguyên ảnh", np.allclose(conv2d(img, I), img))
k = gaussian_kernel(5, 1.0)
check("kernel Gauss tổng = 1", np.isclose(k.sum(), 1))
check("kernel Gauss đối xứng, tâm lớn nhất", np.allclose(k, k.T) and k[2, 2] == k.max())

print("Bước 4 - Otsu")
g = np.full((50, 100), 200.0); g[:, :50] = 50
t = otsu_threshold(g)
check(f"ảnh 2 mức 50/200 -> ngưỡng nằm giữa (t={t})", 50 <= t < 200)
g = np.full((60, 60), 200.0); yy, xx = np.mgrid[:60, :60]; m = (xx - 30) ** 2 + (yy - 30) ** 2 <= 15 ** 2
g[m] = 50
b, _ = segment(g)
check("nền sáng, xu tối -> xu = 1, nền = 0", np.array_equal(b.astype(bool), m))
g2 = 250 - g
b2, _ = segment(g2)
check("nền tối, xu sáng -> xu = 1, nền = 0", np.array_equal(b2.astype(bool), m))

print("Bước 5 - Hình thái học")
sq = np.zeros((30, 30), np.uint8); sq[10:20, 10:20] = 1
check("co bằng ô 3x3 -> vuông 10x10 còn 8x8", erode(sq, np.ones((3, 3), np.uint8)).sum() == 64)
check("giãn bằng ô 3x3 -> vuông 10x10 thành 12x12", dilate(sq, np.ones((3, 3), np.uint8)).sum() == 144)
gap = sq.copy(); gap[:, 15] = 0
check("đóng lấp khe hở 1 pixel", closing(gap, disk(1))[11:19, 15].all())
ring = sq.copy(); ring[13:17, 13:17] = 0
check("lấp lỗ: vòng có lỗ -> vuông đặc", np.array_equal(fill_holes(ring), sq))

print("Bước 6 - Biến đổi khoảng cách")
sq = np.zeros((41, 41), np.uint8); sq[10:31, 10:31] = 1
d = distance_transform(sq)
check(f"tâm vuông 21x21 cách nền 11 px (được {d[20, 20]:.2f})", abs(d[20, 20] - 11) < 0.5)
check("nền có khoảng cách = 0", np.all(d[sq == 0] == 0))
y, x = np.mgrid[:61, :61]; tron = ((x - 30) ** 2 + (y - 30) ** 2 <= 20 ** 2).astype(np.uint8)
dt = distance_transform(tron)
check(f"đường tròn bán kính 20: tâm cách nền ~20 px (được {dt[30, 30]:.0f})", abs(dt[30, 30] - 20) <= 2)

print("Bước 7 - Gán nhãn + watershed")
diag = np.zeros((5, 5), np.uint8); diag[1, 1] = diag[2, 2] = 1
check("2 pixel chéo nhau = 1 vật (8 lân cận)", label_bfs(diag)[1] == 1)
check("2 xu rời nhau -> đếm 2", label_bfs(hai_xu(70))[1] == 2)
dinh = hai_xu(50)
n_truoc = label_bfs(dinh)[1]
n_sau = label_bfs(separate(dinh))[1]
check(f"2 xu dính nhau: trước tách = 1 (được {n_truoc}), sau tách = 2 (được {n_sau})", n_truoc == 1 and n_sau == 2)
y, x = np.mgrid[:100, :200]
ba = sum(((x - cx) ** 2 + (y - 50) ** 2 <= 30 ** 2) for cx in (45, 95, 145)) > 0
n3 = label_bfs(separate(ba.astype(np.uint8)))[1]
check(f"3 xu dính thành chuỗi -> 3 (được {n3})", n3 == 3)

print(f"\nKết quả: {OK}/{N} bài đạt")
sys.exit(0 if OK == N else 1)
