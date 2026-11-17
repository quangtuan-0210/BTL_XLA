"""TV2 - Bước 5-6: chạy các cấu hình pipeline trên một tập ảnh, lưu dự đoán và bảng đánh giá.

Cách dùng (chạy từ thư mục gốc dự án):
    python src/run_tv2_eval.py chinh        tập chỉnh (18 ảnh) - được chạy nhiều lần
    python src/run_tv2_eval.py kiem_tra     tập kiểm tra (72 ảnh) - CHỈ CHẠY 1 LẦN khi đã chốt tham số

Cấu hình so sánh (cùng tham số DEFAULT trong src/pipeline.py):
    tu_viet            pipeline chính: độ khác màu nền + watershed (tự cài đặt)
    tu_viet_khong_tach như trên nhưng bỏ bước watershed       -> đo tác dụng của bước tách vật dính
    tu_viet_xam        Otsu trên ảnh xám thay vì độ khác màu   -> đo tác dụng của cải tiến tách nền
    opencv             cùng thiết kế nhưng dùng hàm OpenCV     -> so cài đặt tự viết với thư viện

Kết quả trong results/tv2/:
    du_doan_<tap>.csv      mỗi ảnh một dòng: số thật + số đếm và thời gian của từng cấu hình
    pred_<tap>_<cau_hinh>.csv  (file, dem, giay) - đầu vào của evaluate()
    bang_ket_qua_<tap>.csv bảng chỉ số theo cấu hình x mức độ
    vat_<tap>.csv          thông số từng vật (cấu hình tu_viet)
"""
import os, sys, time
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run
from src.evaluate import evaluate, result_table, LEVEL_VI
from baseline_opencv.pipeline_opencv import run_opencv

CONFIGS = {
    "tu_viet": lambda img: run(img),
    "tu_viet_khong_tach": lambda img: run(img, use_separate=False),
    "tu_viet_xam": lambda img: run(img, mode="gray"),
    "opencv": lambda img: run_opencv(img),
}
OUT = "results/tv2"


def main(split):
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv("data/labels.csv")
    if split != "all":
        lab = lab[lab.tap == split]
    rows, objs = [], []
    for k, r in enumerate(lab.itertuples(), 1):
        img = np.array(Image.open("data/images/" + r.file).convert("RGB"))
        row = dict(file=r.file, muc_do=r.muc_do, tap=r.tap, that=r.so_luong)
        for name, f in CONFIGS.items():
            t = time.perf_counter(); res = f(img); dt = time.perf_counter() - t
            row["dem_" + name] = res["count"]; row["giay_" + name] = round(dt, 3)
            if name == "tu_viet":
                objs += [dict(file=r.file, **o) for o in res["objects"]]
        rows.append(row)
        print(f"\r{k}/{len(lab)} {r.file:28s}", end="", flush=True)
    print()
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/du_doan_{split}.csv", index=False)
    pd.DataFrame(objs).to_csv(f"{OUT}/vat_{split}.csv", index=False)

    results = {}
    for name in CONFIGS:
        pred = d[["file"]].assign(dem=d["dem_" + name], giay=d["giay_" + name])
        path = f"{OUT}/pred_{split}_{name}.csv"
        pred.to_csv(path, index=False)
        results[name] = evaluate(path, "data/labels.csv")
    t = result_table(results)
    t.to_csv(f"{OUT}/bang_ket_qua_{split}.csv", index=False)

    show = t.assign(muc_do=t.muc_do.map(LEVEL_VI))[["cau_hinh", "muc_do", "so_anh", "ti_le_dung_pct", "mae",
                                                    "sai_lech_tb", "so_anh_dem_thieu", "so_anh_dem_thua", "giay_tb"]]
    print(f"== Tập '{split}' ({len(d)} ảnh) ==")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "chinh")
