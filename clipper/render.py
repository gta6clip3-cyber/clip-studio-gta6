"""Montage final avec FFmpeg : 9:16, facecam en haut, jeu en bas, zoom, sous-titres, sons."""
from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import config
from .layout import Layout
from .sfx import ensure_sfx
from .utils import log, probe_fps, run, tool


def _encoder_args(cfg: dict) -> list[str]:
    if cfg.get("use_nvenc", True):
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "22", "-b:v", "8M",
                "-maxrate", "11M", "-bufsize", "16M", "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]


def render_clip(src: Path, offset: float, duration: float, layout: Layout, ass_file: Path,
                punch: float | None, out: Path, music: Path | None) -> Path:
    cfg = config.load()
    W, H = cfg["width"], cfg["height"]
    src, out, ass_file = Path(src).resolve(), Path(out).resolve(), Path(ass_file).resolve()
    fps = probe_fps(src)
    sfx = ensure_sfx()
    gx, gy, gw, gh = layout.game
    D = f"{duration:.3f}"

    if punch is not None and 0.5 < punch < duration - 0.5:
        P = f"{punch:.3f}"
        zexpr = f"1+0.14*clip((it-{P})/0.18,0,1)*clip(({P}+1.7-it)/0.35,0,1)"
    else:
        zexpr = "1"
    zoom = (f"zoompan=z='{zexpr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:"
            f"s={layout.fg_w}x{layout.fg_h}:fps={fps:g}")
    PH = layout.panel_h
    # panneau jeu = zone de jeu cadree, posee sur une version floutee d'elle-meme si elle ne remplit pas tout
    game = (f"[vb]crop={gw}:{gh}:{gx}:{gy},split=2[g1][g2];"
            f"[g1]scale={W}:{PH}:force_original_aspect_ratio=increase,crop={W}:{PH},"
            f"gblur=sigma=26,eq=brightness=-0.15,setsar=1[gbg];"
            f"[g2]{zoom},setsar=1[gfg];"
            f"[gbg][gfg]overlay=(W-w)/2:(H-h)/2[game];")
    if layout.has_cam:
        cx, cy, cw, ch = layout.cam
        vf = (f"[0:v]split=2[va][vb];"
              f"[va]crop={cw}:{ch}:{cx}:{cy},scale={W}:{layout.cam_h}:flags=lanczos,setsar=1[cam];"
              + game +
              f"[cam][game]vstack=inputs=2,format=yuv420p[stack];")
    else:
        vf = "[0:v]null[vb];" + game + "[game]format=yuv420p[stack];"
    vf += f"[stack]ass={ass_file.name}[vout]"

    sv = cfg["sfx_volume"]
    inputs = ["-ss", f"{offset:.3f}", "-t", D, "-i", str(src),
              "-i", str(sfx["whoosh"]), "-i", str(sfx["boom"])]
    af = "[0:a]aresample=48000,aformat=channel_layouts=stereo"
    mix = ["[voice]", "[s1]"]
    if music:
        af += ",asplit=2[voice][key];"
    else:
        af += "[voice];"
    af += f"[1:a]volume={sv * 0.8:.2f}[s1];"
    if punch is not None and 0.5 < punch < duration - 0.5:
        ms = int(max(0, punch - 0.05) * 1000)
        af += f"[2:a]adelay={ms}|{ms},volume={sv:.2f}[s2];"
        mix.append("[s2]")
    if music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        mi = 3
        af += (f"[{mi}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{D},"
               f"volume={cfg['music_volume']:.2f}[mus];"
               f"[mus][key]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=350[mduck];")
        mix.append("[mduck]")
    af += (f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0,"
           f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")

    cmd = [tool("ffmpeg"), "-y", "-v", "error", *inputs,
           "-filter_complex", vf + ";" + af, "-map", "[vout]", "-map", "[aout]",
           *_encoder_args(cfg), "-pix_fmt", "yuv420p", "-r", f"{fps:g}",
           "-c:a", "aac", "-b:a", "192k", "-t", D, "-movflags", "+faststart", str(out)]
    try:
        run(cmd, cwd=ass_file.parent)
    except RuntimeError as e:
        if "nvenc" in str(e).lower() or "No capable devices" in str(e):
            log.warning("NVENC indisponible, encodage CPU")
            cfg2 = dict(cfg, use_nvenc=False)
            enc = _encoder_args(cfg2)
            i = cmd.index("-c:v")
            cmd = cmd[:i] + enc + cmd[cmd.index("-pix_fmt"):]
            run(cmd, cwd=ass_file.parent)
        else:
            raise
    return out


# ---------------------------------------------------------------- miniature

def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for f in (name, "impact.ttf", "ariblk.ttf", "arialbd.ttf"):
        for p in (f, f"C:/Windows/Fonts/{f}"):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    try:  # Linux : la police grasse du systeme (fontconfig)
        return ImageFont.truetype(run(["fc-match", "-f", "%{file}", "sans:bold"]).stdout, size)
    except (OSError, RuntimeError):
        return ImageFont.load_default(size)


def _grab(src: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Image de couverture introuvable")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def make_cover(src: Path, t: float, layout: Layout, title: str, out: Path) -> Path:
    cfg = config.load()
    W, H = cfg["width"], cfg["height"]
    frame = Image.fromarray(_grab(src, t))
    gx, gy, gw, gh = layout.game
    canvas = Image.new("RGB", (W, H), "black")
    crop = frame.crop((gx, gy, gx + gw, gy + gh))
    PH = layout.panel_h
    s = max(W / gw, PH / gh)
    bg = crop.resize((int(gw * s) + 1, int(gh * s) + 1), Image.LANCZOS)
    bg = bg.crop(((bg.width - W) // 2, (bg.height - PH) // 2, (bg.width - W) // 2 + W, (bg.height - PH) // 2 + PH))
    bg = Image.eval(bg.filter(ImageFilter.GaussianBlur(26)), lambda v: int(v * 0.8))
    canvas.paste(bg, (0, layout.panel_y))
    canvas.paste(crop.resize((layout.fg_w, layout.fg_h), Image.LANCZOS), ((W - layout.fg_w) // 2, layout.fg_y))
    if layout.has_cam:
        cx, cy, cw, ch = layout.cam
        cam = frame.crop((cx, cy, cx + cw, cy + ch)).resize((W, layout.cam_h), Image.LANCZOS)
        canvas.paste(cam, (0, 0))
    # bandeau sombre derriere le titre
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    top = layout.panel_y + 40 if layout.has_cam else max(60, layout.fg_y - 420)
    for i in range(520):
        sd.line([(0, top + i), (W, top + i)], fill=int(170 * (1 - i / 520)))
    canvas = Image.composite(Image.new("RGB", (W, H), "black"), canvas, shade)
    d = ImageDraw.Draw(canvas)
    font = _font(cfg["title_font"].lower() + ".ttf", 118)
    lines = textwrap.wrap(title.upper(), width=16)[:4]
    y = top + 30
    for k, line in enumerate(lines):
        color = (255, 214, 0) if k % 2 == 0 else (255, 255, 255)
        tw = d.textlength(line, font=font)
        d.text(((W - tw) / 2, y), line, font=font, fill=color, stroke_width=9, stroke_fill="black")
        y += 128
    canvas.save(out, "JPEG", quality=92)
    return out
