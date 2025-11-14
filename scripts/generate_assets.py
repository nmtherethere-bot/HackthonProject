import os
from PIL import Image, ImageDraw, ImageFont


def try_load_font(size: int) -> ImageFont.FreeTypeFont:
    """Attempt to load a reasonable system font; fall back to Pillow default."""
    candidates = [
        # Common Windows fonts
        r"C:\\Windows\\Fonts\\segoeui.ttf",
        r"C:\\Windows\\Fonts\\arial.ttf",
        r"C:\\Windows\\Fonts\\calibri.ttf",
        # Fallback names (if Pillow can resolve from fonts dir)
        "segoeui.ttf",
        "arial.ttf",
        "calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def linear_gradient(width: int, height: int, start_rgb=(12, 16, 22), end_rgb=(36, 88, 200)) -> Image.Image:
    """Create a vertical linear gradient from start_rgb (top) to end_rgb (bottom)."""
    top = Image.new("RGB", (width, height), start_rgb)
    bottom = Image.new("RGB", (width, height), end_rgb)
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)
    for y in range(height):
        # 0..255 across height
        val = int(255 * (y / max(1, height - 1)))
        draw.line([(0, y), (width, y)], fill=val)
    return Image.composite(bottom, top, mask)


def measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Measure text size; use textbbox if available for better accuracy."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        return draw.textsize(text, font=font)


def draw_badge(draw: ImageDraw.ImageDraw, xy, text: str, fill=(58, 130, 246), font=None, padding=(16, 8), radius=12):
    x, y = xy
    tw, th = measure(draw, text, font)
    bw = tw + padding[0] * 2
    bh = th + padding[1] * 2
    rect = [x, y, x + bw, y + bh]
    # rounded_rectangle is available in modern Pillow; fallback if needed
    try:
        draw.rounded_rectangle(rect, radius=radius, fill=fill)
    except Exception:
        draw.rectangle(rect, fill=fill)
    draw.text((x + padding[0], y + padding[1]), text, fill=(255, 255, 255), font=font)
    return bw, bh


def generate_card():
    W, H = 1200, 630  # Social/OG card size
    img = linear_gradient(W, H, start_rgb=(10, 12, 16), end_rgb=(30, 84, 190))
    draw = ImageDraw.Draw(img)

    title_font = try_load_font(88)
    subtitle_font = try_load_font(36)
    badge_font = try_load_font(28)

    margin_x = 80
    base_y = int(H * 0.28)

    title = "ExplainAI"
    subtitle = "Transparent Reasoning with SFT+GRPO on TPU"

    # Title
    draw.text((margin_x, base_y), title, font=title_font, fill=(255, 255, 255))
    # Subtitle
    draw.text((margin_x, base_y + 110), subtitle, font=subtitle_font, fill=(224, 234, 255))

    # Badges row
    bx = margin_x
    by = base_y + 170
    badges = [
        ("SFT", (58, 130, 246)),
        ("GRPO", (130, 58, 246)),
        ("TPU v4-8/16", (46, 184, 113)),
    ]
    for label, color in badges:
        bw, _ = draw_badge(draw, (bx, by), label, fill=color, font=badge_font)
        bx += bw + 16

    # Footer tag
    footer = "Reproducible configs • Verifiable outputs • Tunix-ready"
    fw, fh = measure(draw, footer, subtitle_font)
    draw.text((W - fw - 80, H - fh - 60), footer, font=subtitle_font, fill=(210, 220, 240))

    os.makedirs("assets", exist_ok=True)
    out_path = os.path.join("assets", "explainai_card.png")
    img.save(out_path)
    return out_path


def generate_thumbnail():
    W, H = 512, 512  # Square thumbnail
    img = linear_gradient(W, H, start_rgb=(12, 16, 22), end_rgb=(36, 88, 200))
    draw = ImageDraw.Draw(img)

    title_font = try_load_font(64)
    sub_font = try_load_font(28)
    badge_font = try_load_font(24)

    # Centered title
    title = "ExplainAI"
    tw, th = measure(draw, title, title_font)
    tx = (W - tw) // 2
    ty = int(H * 0.35) - th // 2
    draw.text((tx, ty), title, font=title_font, fill=(255, 255, 255))

    subtitle = "SFT+GRPO on TPU"
    sw, sh = measure(draw, subtitle, sub_font)
    sx = (W - sw) // 2
    sy = ty + th + 20
    draw.text((sx, sy), subtitle, font=sub_font, fill=(224, 234, 255))

    # Badges centered
    bpad = 12
    labels = [
        ("SFT", (58, 130, 246)),
        ("GRPO", (130, 58, 246)),
        ("TPU", (46, 184, 113)),
    ]
    total_w = 0
    sizes = []
    for l, c in labels:
        # measure before drawing to compute total
        tw2, th2 = measure(draw, l, badge_font)
        bw2 = tw2 + 16 * 2
        sizes.append((bw2, th2 + 8 * 2, l, c))
        total_w += bw2
    total_w += bpad * (len(labels) - 1)
    start_x = (W - total_w) // 2
    by = sy + sh + 30
    x = start_x
    for bw2, bh2, l, c in sizes:
        draw_badge(draw, (x, by), l, fill=c, font=badge_font)
        x += bw2 + bpad

    os.makedirs("assets", exist_ok=True)
    out_path = os.path.join("assets", "explainai_thumbnail.png")
    img.save(out_path)
    return out_path


def main():
    card_path = generate_card()
    thumb_path = generate_thumbnail()
    print(f"Card saved to: {card_path}")
    print(f"Thumbnail saved to: {thumb_path}")


if __name__ == "__main__":
    main()