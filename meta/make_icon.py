"""Cree l'icone 1024x1024 de l'app Meta (meta/icone_clip_studio.png)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 1024
out = Path(__file__).with_name("icone_clip_studio.png")
img = Image.new("RGB", (S, S))
d = ImageDraw.Draw(img)
for y in range(S):  # degrade violet -> rose facon "Vice City"
    t = y / S
    d.line([(0, y), (S, y)], fill=(int(90 + 160 * t), int(30 + 20 * t), int(160 - 40 * t)))
glow = Image.new("L", (S, S), 0)
ImageDraw.Draw(glow).ellipse((180, 180, 844, 844), fill=180)
img = Image.composite(Image.new("RGB", (S, S), (255, 120, 200)), img, glow.filter(ImageFilter.GaussianBlur(120)))
d = ImageDraw.Draw(img)
# icone "lecture" blanche dans un cadre vertical 9:16
d.rounded_rectangle((322, 172, 702, 852), radius=60, outline="white", width=34)
d.polygon([(450, 380), (450, 644), (640, 512)], fill="white")


def font(size):
    for f in ("C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


f = font(120)
txt = "CLIP"
w = d.textlength(txt, font=f)
d.text(((S - w) / 2, 880), txt, font=f, fill="white", stroke_width=6, stroke_fill=(60, 10, 80))
img.save(out)
print(out)
