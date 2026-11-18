"""Vẽ kết quả - tô màu từng vật, vẽ viền, đánh số lên ảnh gốc."""
import numpy as np
from PIL import Image, ImageDraw


def palette(n, seed=7):
    """n+1 màu ngẫu nhiên (màu 0 là nền, không dùng)."""
    return np.random.default_rng(seed).integers(40, 255, (n + 1, 3))


def draw_objects(img_rgb, labels, alpha=0.45, numbers=True, suspicious_ids=()):
    """Tô màu nhẹ từng vật, viền đỏ, số thứ tự ở tâm. img_rgb và labels cùng kích thước.
    suspicious_ids: vật nghi gộp nhiều xu -> số in màu đỏ. Trả về ảnh PIL."""
    labels = np.asarray(labels)
    n = int(labels.max())
    base = np.asarray(img_rgb, dtype=np.float64).copy()
    if base.ndim == 2:
        base = np.stack([base] * 3, -1)
    m = labels > 0
    if n:
        pal = palette(n)
        base[m] = (1 - alpha) * base[m] + alpha * pal[labels[m]]
    edge = np.zeros_like(m)                     # viền = điểm có hàng xóm mang nhãn khác
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    base[edge] = (255, 40, 40)
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if not (numbers and n):
        return im
    dr = ImageDraw.Draw(im)
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    cnt = np.bincount(lab, minlength=n + 1)
    cx = np.bincount(lab, xs, n + 1); cy = np.bincount(lab, ys, n + 1)
    r = max(7, min(11, int(min(labels.shape) / 40)))
    for i in range(1, n + 1):
        if cnt[i] == 0:
            continue
        x, y = cx[i] / cnt[i], cy[i] / cnt[i]
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=(20, 20, 20))
        dr.text((x, y), str(i), fill=(220, 0, 0) if i in suspicious_ids else (0, 0, 0), anchor="mm")
    return im


def _font(size=13):
    """Phông có dấu tiếng Việt: DejaVuSans đi kèm matplotlib (phông mặc định của PIL không có dấu)."""
    from PIL import ImageFont
    try:
        import os, matplotlib
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def caption(im, text, color=(0, 0, 0), h=20, size=13):
    """Thêm một dòng chữ phía trên ảnh (dùng cho bảng ảnh minh họa)."""
    out = Image.new("RGB", (im.width, im.height + h), "white")
    out.paste(im, (0, h))
    ImageDraw.Draw(out).text((3, 2), text, fill=color, font=_font(size))
    return out


def grid(images, cols, pad=6, bg="white"):
    """Ghép nhiều ảnh PIL thành lưới (mọi ô cùng kích thước ô lớn nhất)."""
    cw = max(i.width for i in images); ch = max(i.height for i in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad) + pad), bg)
    for k, im in enumerate(images):
        sheet.paste(im, (pad + (k % cols) * (cw + pad), pad + (k // cols) * (ch + pad)))
    return sheet
