"""Enregistre la video de demonstration pour la validation TikTok (Content Posting API).

python meta/record_tiktok_demo.py   (Clip Studio doit tourner sur http://127.0.0.1:8766)
-> meta/demo_tiktok_clip_studio.mp4. ATTENTION : publie vraiment un clip, en « Moi uniquement »."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright  # noqa: E402

from clipper import config  # noqa: E402
from clipper.utils import tool  # noqa: E402

APP = "http://127.0.0.1:8766"
OUT = ROOT / "meta" / "demo_tiktok_clip_studio.mp4"
TMP = ROOT / "work" / "demo_rec"
W, H = 1366, 820

OVERLAY = """
(() => {
  if (window.__demo) return; window.__demo = 1;
  const add = () => {
    const c = document.createElement('div'); c.id = '__cur';
    c.style.cssText = 'position:fixed;z-index:2147483647;width:22px;height:22px;border-radius:50%;pointer-events:none;' +
      'background:rgba(255,255,255,.85);border:3px solid #fe2c55;box-shadow:0 0 12px rgba(0,0,0,.6);left:-50px;top:-50px;transition:transform .12s';
    document.body.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = (e.clientX - 11) + 'px'; c.style.top = (e.clientY - 11) + 'px'; }, true);
    document.addEventListener('mousedown', () => c.style.transform = 'scale(.6)', true);
    document.addEventListener('mouseup', () => c.style.transform = 'scale(1)', true);
    const b = document.createElement('div'); b.id = '__cap';
    b.style.cssText = 'position:fixed;z-index:2147483646;left:50%;bottom:22px;transform:translateX(-50%);max-width:88%;' +
      'background:rgba(10,8,18,.92);color:#fff;font:600 19px/1.4 Segoe UI,Arial,sans-serif;padding:12px 20px;border-radius:12px;' +
      'border:2px solid #fe2c55;box-shadow:0 6px 30px rgba(0,0,0,.6);text-align:center;display:none';
    document.body.appendChild(b);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""


def cap(page, text: str, wait: float = 3.5) -> None:
    page.evaluate("t => { const b = document.getElementById('__cap'); if (b) { b.textContent = t; b.style.display = t ? 'block' : 'none'; } }", text)
    time.sleep(wait)


def move_click(page, loc, wait: float = 0.6, click: bool = True) -> None:
    loc.scroll_into_view_if_needed()
    box = loc.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=28)
    time.sleep(0.35)
    if click:
        loc.click()
    time.sleep(wait)


def pick_clip(exclude: str = "") -> dict:
    """Le clip le plus court de la file TikTok (pas encore publie sur TikTok)."""
    clips = []
    for d in config.OUTPUT.iterdir():
        m = d / "meta.json"
        if m.exists() and (d / "clip.mp4").exists() and d.name != exclude:
            meta = json.loads(m.read_text(encoding="utf-8"))
            if not meta.get("published", {}).get("tiktok", {}).get("ok"):
                clips.append({**meta, "folder": d.name})
    return min(clips, key=lambda c: c.get("duration") or 999)


def main() -> None:
    cfg = config.load()
    clip = pick_clip()
    print("clip choisi :", clip["folder"], clip.get("duration"), "s")
    TMP.mkdir(parents=True, exist_ok=True)
    auth = "https://www.tiktok.com/v2/auth/authorize/?" + urlencode({
        "client_key": cfg["tiktok_client_key"], "response_type": "code", "state": "demo",
        "scope": "user.info.basic,video.publish,video.upload,user.info.stats,video.list",
        "redirect_uri": cfg.get("tiktok_redirect_uri", "")})
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(TMP),
                                  record_video_size={"width": W, "height": H}, locale="fr-FR")
        ctx.add_init_script(OVERLAY)
        page = ctx.new_page()
        page.on("dialog", lambda d: d.accept())

        page.goto(APP)
        page.wait_for_timeout(2500)
        cap(page, "Clip Studio – demo of the TikTok integration (Login Kit + Content Posting API)", 4.5)

        cap(page, "Step 1 – The user logs in with TikTok and authorizes Clip Studio (Login Kit, OAuth).", 1.5)
        page.goto(auth)
        page.wait_for_timeout(4500)
        cap(page, "Step 1 – TikTok's own login / consent page: the user signs in and approves the requested scopes.", 5)
        page.goto(APP)
        page.wait_for_timeout(2000)

        move_click(page, page.locator('#nav button[data-tab="settings"]'), 2.5)
        cap(page, "Step 2 – The TikTok account is connected (user.info.basic: @username shown).", 4)
        move_click(page, page.locator('#nav button[data-tab="dash"]'), 2.5)
        tt_card = page.locator(".plat").nth(1)
        move_click(page, tt_card, 0.5, click=False)
        cap(page, "Step 2 – Analytics dashboard: the user's own follower count (user.info.stats) and their public videos' "
                  "views/likes (video.list), refreshed every 15 min.", 6)
        move_click(page, page.locator('#nav button[data-tab="videos"]'), 2)
        cap(page, "Step 2 – Per-video analytics table (video.list). Only the connected user's own data is read.", 4)

        move_click(page, page.locator('#nav button[data-tab="clips"]'), 2.5)
        cap(page, "Step 3 – The user picks one of their clips and clicks « TikTok » to open the posting screen.", 2.5)
        move_click(page, page.locator(f"button.tt[onclick*=\"{clip['folder']}\"]").first, 3.5)

        cap(page, "Step 4 – Creator info from creator_info/query: avatar and nickname of the account that will post.", 4.5)
        cap(page, "Step 4 – Preview of the video. The caption is shown and editable by the user.", 1.5)
        move_click(page, page.locator("#ttcap"), 0.4)
        page.keyboard.press("End")
        page.keyboard.type(" #clipstudio", delay=90)
        time.sleep(1.2)

        cap(page, "Step 5 – Privacy has NO default: the user must choose it (options returned by TikTok).", 2.5)
        move_click(page, page.locator("#ttpriv"), 0.8, click=False)
        page.locator("#ttpriv").select_option("SELF_ONLY")
        time.sleep(2)

        cap(page, "Step 6 – Comments / Duet / Stitch are off by default. Duet & Stitch are greyed: disabled in the creator's settings.", 2.5)
        move_click(page, page.locator("label.chk:has(#tt_comment)"), 2)

        cap(page, "Step 7 – Commercial content disclosure (off by default).", 2)
        move_click(page, page.locator("#ttcom"), 2)
        cap(page, "Step 7 – « Ta marque » = Your brand → labelled « Promotional content ».", 1.5)
        move_click(page, page.locator("label.chk:has(#ttorg)"), 2.5)
        cap(page, "Step 7 – « Contenu de marque » = Branded content → « Only me » is no longer allowed, Branded Content Policy added.", 1.5)
        move_click(page, page.locator("label.chk:has(#ttbrand)"), 4)
        move_click(page, page.locator("label.chk:has(#ttbrand)"), 0.8)
        move_click(page, page.locator("label.chk:has(#ttorg)"), 0.8)
        move_click(page, page.locator("#ttcom"), 1.5)
        page.locator("#ttpriv").select_option("SELF_ONLY")
        time.sleep(1)

        cap(page, "Step 8 – Music Usage Confirmation is shown before posting. The user explicitly clicks « Publier ».", 3.5)
        move_click(page, page.locator("#ttgo"), 1)
        cap(page, "Step 8 – Uploading the video to TikTok (Direct Post)…", 0.5)
        page.wait_for_selector("#ttmsg .msg.ok, #ttmsg .msg.err", timeout=360000)
        ok = page.locator("#ttmsg .msg.ok").count() > 0
        cap(page, "Step 9 – Done. The user is told TikTok may need a few minutes to process the video." if ok
            else "Step 9 – TikTok returned an error, shown to the user.", 5)
        print("publication :", "OK" if ok else page.locator("#ttmsg").inner_text())

        move_click(page, page.locator("#ttm .close"), 1.5)
        draft = pick_clip(exclude=clip["folder"])
        print("brouillon :", draft["folder"])
        cap(page, "Step 10 – Upload to TikTok as a draft (video.upload): the user clicks « Brouillon TikTok » on a clip.", 3)
        move_click(page, page.locator(f"button[onclick*=\"ttDraft('{draft['folder']}'\"]").first, 1)
        cap(page, "Step 10 – Sending the video to the user's TikTok inbox…", 0.5)
        page.wait_for_selector("body > .msg.ok, body > .msg.err", timeout=360000)  # la notification, pas l'ancien message
        okd = page.locator("body > .msg.ok").count() > 0
        cap(page, "Step 10 – Done: the draft is in the user's TikTok inbox. They edit and post it themselves in the TikTok app."
            if okd else "Step 10 – TikTok returned an error, shown to the user.", 6)
        print("brouillon :", "OK" if okd else page.locator("body > .msg").first.inner_text())
        cap(page, "Every TikTok post is chosen and confirmed by the user – Clip Studio never posts to TikTok automatically.", 5)
        cap(page, "", 0.8)
        video = page.video.path()
        ctx.close()
        browser.close()
    subprocess.run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-c:v", "libx264", "-preset", "medium",
                    "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], check=True)
    for f in TMP.iterdir():
        f.unlink()
    TMP.rmdir()
    print("video :", OUT)


if __name__ == "__main__":
    main()
