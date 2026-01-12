import os
import json
import time
import logging
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
from pdf2image import convert_from_path
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from threading import Lock
from datetime import datetime
import contextlib

RESTART_INTERVAL_SECONDS = 6 * 3600
DEBUG_LOG = "pickingslip_debug.log"

def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    line = f"{now()} - {msg}"
    print(line)
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass

# setup simple logging file too
try:
    logging.basicConfig(filename=DEBUG_LOG, level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
except Exception:
    pass

lock = Lock()
last_seen = {}

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("dpi", 300)
    cfg.setdefault("coords_dpi", 600)
    cfg.setdefault("stable_seconds", 2)
    cfg.setdefault("max_wait", 30)
    cfg.setdefault("workers", max(1, (os.cpu_count() or 2) - 1))
    cfg.setdefault("file_age_seconds", 2)
    return cfg

def wait_for_stable(path, stable_seconds, max_wait):
    start = time.time()
    prev = -1
    while time.time() - start < max_wait:
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == prev:
            time.sleep(stable_seconds)
            try:
                if os.path.getsize(path) == size:
                    return True
            except OSError:
                return False
        prev = size
        time.sleep(0.5)
    return False

def pdf_to_png(pdf_path, dpi):
    try:
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        if not pages:
            return None
        tmp = tempfile.NamedTemporaryFile(prefix="ocr_", suffix=".png", delete=False)
        pages[0].save(tmp.name, format="PNG")
        tmp.close()
        return tmp.name
    except Exception as e:
        log(f"pdf_to_png error {e}")
        return None

def clamp(a, low, high):
    return max(low, min(a, high))

def scale_coords(coords, src_dpi, dst_dpi):
    if not coords or len(coords) != 4:
        return None
    scale = dst_dpi / float(src_dpi)
    x0, y0, x1, y1 = coords
    return [int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)]

def crop_with_coords(img, coords):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = coords
    x0 = clamp(x0, 0, w - 1)
    x1 = clamp(x1, 1, w)
    y0 = clamp(y0, 0, h - 1)
    y1 = clamp(y1, 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 30:
        gray = cv2.equalizeHist(gray)
    return gray

def safe_decode_image(img):
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                return decode(img) or []
    except Exception:
        return []

def try_decode(img):
    if len(img.shape) == 2:
        target = img
    else:
        target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    proc = cv2.filter2D(target, -1, kernel)
    variants = [proc,
                cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 15, 9)]
    for v in variants:
        decoded_objs = safe_decode_image(v)
        if decoded_objs:
            return [d.data.decode("utf-8", errors="ignore").strip() for d in decoded_objs]
    return []

def decode_pipeline(png_path, coords_scaled):
    img = cv2.imread(png_path)
    if img is None:
        return None
    candidates = []
    if coords_scaled:
        crop = crop_with_coords(img, coords_scaled)
        if crop is not None:
            candidates.append(crop)
    candidates.append(img)
    added = []
    for c in candidates:
        h = c.shape[0]
        if h < 250:
            scale = int(400 / max(1, h))
            nw = int(c.shape[1] * scale)
            resized = cv2.resize(c, (nw, 400), interpolation=cv2.INTER_LINEAR)
            added.append(resized)
    candidates.extend(added)
    for cand in candidates:
        gray = preprocess_gray(cand)
        res = try_decode(gray)
        if res:
            return res
        r = cand
        for _ in range(3):
            r = cv2.rotate(r, cv2.ROTATE_90_CLOCKWISE)
            gray = preprocess_gray(r)
            res = try_decode(gray)
            if res:
                return res
    return None

def safe_move(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dest = os.path.join(dst_dir, base)
    i = 1
    while os.path.exists(dest):
        dest = os.path.join(dst_dir, f"{name}_{i}{ext}")
        i += 1
    shutil.move(src, dest)
    return dest

def remove_image_files(dirs, exts=None):
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".png", ".webp"}
    removed = 0
    for d in dirs:
        try:
            if not d:
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in exts:
                        fp = os.path.join(root, fn)
                        try:
                            os.remove(fp)
                            removed += 1
                        except Exception:
                            pass
        except Exception:
            pass
    if removed:
        log(f"Removed {removed} image files from {dirs}")

def maintain_roi_debug(dirpath, keep=5):
    try:
        if not dirpath:
            return
        files = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for old in files[keep:]:
            try:
                os.remove(old)
            except Exception:
                pass
    except Exception:
        pass

def process_file(path, cfg):
    with lock:
        if path in last_seen and time.time() - last_seen[path] < cfg["file_age_seconds"]:
            return
    if not wait_for_stable(path, cfg["stable_seconds"], cfg["max_wait"]):
        log(f"file not stable: {path}")
        return
    png = pdf_to_png(path, cfg["dpi"])
    if not png or not os.path.exists(png):
        log(f"conversion failed: {path}")
        safe_move(path, cfg["error_dir"])
        return
    coords = cfg.get("barcode_coords")
    coords_scaled = None
    if coords:
        coords_scaled = scale_coords(coords, cfg.get("coords_dpi", 600), cfg["dpi"])
    try:
        decoded = decode_pipeline(png, coords_scaled)
        if decoded:
            for d in decoded:
                only = "".join(ch for ch in d if ch.isdigit())
                if len(only) == 7:
                    target_name = f"{only}.pdf"
                    os.makedirs(cfg["output_dir"], exist_ok=True)
                    dest = os.path.join(cfg["output_dir"], target_name)
                    i = 1
                    base, ext = os.path.splitext(target_name)
                    while os.path.exists(dest):
                        dest = os.path.join(cfg["output_dir"], f"{base}_{i}{ext}")
                        i += 1
                    shutil.copyfile(path, dest)
                    log(f"Moved to output {dest}")
                    try:
                        os.remove(png)
                    except Exception:
                        pass
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    with lock:
                        last_seen[path] = time.time()
                    # cleanup image files
                    remove_image_files([cfg.get("error_dir"), cfg.get("output_dir"), os.path.dirname(png), cfg.get("watch_dir")])
                    # keep ROI debug small
                    maintain_roi_debug(os.path.join(cfg.get("error_dir",""), "roi_debug"), keep=5)
                    return
            log(f"Decoded but no matching numeric token length 7: {decoded}")
        else:
            log(f"No barcode found in {path}")
        if os.path.exists(png):
            safe_move(png, cfg["error_dir"])
        safe_move(path, cfg["error_dir"])
        remove_image_files([cfg.get("error_dir"), cfg.get("output_dir"), os.path.dirname(png), cfg.get("watch_dir")])
        maintain_roi_debug(os.path.join(cfg.get("error_dir",""), "roi_debug"), keep=5)
    except Exception as e:
        logging.exception("Processing error %s", e)
        if os.path.exists(png):
            safe_move(png, cfg["error_dir"])
        safe_move(path, cfg["error_dir"])
        remove_image_files([cfg.get("error_dir"), cfg.get("output_dir"), os.path.dirname(png), cfg.get("watch_dir")])
        maintain_roi_debug(os.path.join(cfg.get("error_dir",""), "roi_debug"), keep=5)

def monitor(cfg):
    pool = ProcessPoolExecutor(max_workers=cfg["workers"])
    futures = {}
    start_time = time.time()
    # initial clean
    remove_image_files([cfg.get("watch_dir"), cfg.get("error_dir"), cfg.get("output_dir")])
    try:
        while True:
            try:
                for fname in os.listdir(cfg["watch_dir"]):
                    if not fname.lower().endswith(".pdf"):
                        continue
                    full = os.path.join(cfg["watch_dir"], fname)
                    if full in futures:
                        f = futures[full]
                        if f.done():
                            futures.pop(full, None)
                        else:
                            continue
                    futures[full] = pool.submit(process_file, full, cfg)
            except FileNotFoundError:
                log(f"watch directory not found: {cfg.get('watch_dir')}")
            except Exception as e:
                log(f"Error in monitor loop listing files: {e}")
            time.sleep(1)
            if time.time() - start_time > RESTART_INTERVAL_SECONDS:
                log("Scheduled restart of pickingslip monitor")
                break
    finally:
        pool.shutdown(wait=True)

if __name__ == "__main__":
    while True:
        try:
            cfg = load_config("pickingslipconfig.json")
            log(f"Start monitor watching {cfg['watch_dir']}")
            monitor(cfg)
            log("Monitor cycle ended, restarting shortly")
            time.sleep(2)
        except Exception as e:
            log(f"Fatal error in pickingslip main loop: {e}")
            time.sleep(5)
