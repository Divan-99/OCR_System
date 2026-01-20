"""
Picking Slip Barcode OCR Processor

Purpose:
- Monitor a folder for incoming PDF picking slips.
- Convert the first page of each PDF into a PNG image.
- Optionally crop a defined barcode region.
- Apply multiple preprocessing strategies.
- Decode barcodes using pyzbar.
- Extract a 7-digit number from the barcode.
- Rename and move the PDF into the output folder.
- Move failed files into an error folder.
- Run continuously with automatic restart.

Key Features:
- Multiprocessing for CPU-heavy barcode decoding.
- File stability checking to avoid partial reads.
- DPI scaling support for ROI coordinates.
- Image rotation and enhancement fallback logic.
"""

import os
import json
import time
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
from pdf2image import convert_from_path
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from threading import Lock
from datetime import datetime, timezone
import contextlib
import sys
from colorama import init, Fore


# ------------------------------------------------------------
# Console setup
# ------------------------------------------------------------

# Enable colored console output on Windows
init(autoreset=True)

# Disable OpenCV internal logging noise
cv2.setLogLevel(0)


# ------------------------------------------------------------
# Runtime constants
# ------------------------------------------------------------

# Restart watcher every 24 hours for stability
RESTART_INTERVAL_SECONDS = 24 * 3600


# ------------------------------------------------------------
# Global concurrency control
# ------------------------------------------------------------

# Prevent duplicate processing of the same file
lock = Lock()
active_processing = set()


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

def now():
    """
    Returns current UTC timestamp string for logging.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg):
    """
    Print normal operational messages.
    """
    print(f"{Fore.WHITE}{now()} {msg}")


def log_success(msg):
    """
    Print successful barcode extraction messages.
    """
    print(f"{Fore.WHITE}{now()} {Fore.GREEN}{msg}")


def log_error(msg):
    """
    Print error messages.
    """
    print(f"{Fore.WHITE}{now()} {Fore.RED}{msg}")


def log_restart(msg):
    """
    Print restart messages.
    """
    print(f"{Fore.WHITE}{now()} {Fore.MAGENTA}{msg}")


# ------------------------------------------------------------
# Configuration loader
# ------------------------------------------------------------

def load_config(path):
    """
    Load JSON configuration and apply default values.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Default values if not present in config
    cfg.setdefault("dpi", 300)
    cfg.setdefault("coords_dpi", 600)
    cfg.setdefault("stable_seconds", 2)
    cfg.setdefault("max_wait", 30)
    cfg.setdefault("workers", max(1, (os.cpu_count() or 2) - 1))
    cfg.setdefault("file_age_seconds", 2)

    return cfg


# ------------------------------------------------------------
# File stability detection
# ------------------------------------------------------------

def wait_for_stable(path, stable_seconds, max_wait):
    """
    Wait until a file size remains unchanged for a period of time.
    Prevents reading files that are still being copied.
    """
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


# ------------------------------------------------------------
# PDF conversion
# ------------------------------------------------------------

def pdf_to_png(pdf_path, dpi):
    """
    Convert the first page of a PDF to a temporary PNG image.
    """
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=1,
            last_page=1
        )

        if not pages:
            return None

        tmp = tempfile.NamedTemporaryFile(
            prefix="ocr_",
            suffix=".png",
            delete=False
        )

        pages[0].save(tmp.name, format="PNG")
        tmp.close()

        return tmp.name

    except Exception:
        return None


# ------------------------------------------------------------
# Coordinate helpers
# ------------------------------------------------------------

def clamp(a, low, high):
    """
    Clamp a value between low and high bounds.
    """
    return max(low, min(a, high))


def scale_coords(coords, src_dpi, dst_dpi):
    """
    Scale ROI coordinates when image DPI differs from design DPI.
    """
    if not coords or len(coords) != 4:
        return None

    scale = dst_dpi / float(src_dpi)
    x0, y0, x1, y1 = coords

    return [
        int(x0 * scale),
        int(y0 * scale),
        int(x1 * scale),
        int(y1 * scale)
    ]


def crop_with_coords(img, coords):
    """
    Crop image using ROI coordinates with bounds safety.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = coords

    x0 = clamp(x0, 0, w - 1)
    x1 = clamp(x1, 1, w)
    y0 = clamp(y0, 0, h - 1)
    y1 = clamp(y1, 1, h)

    if x1 <= x0 or y1 <= y0:
        return None

    return img[y0:y1, x0:x1]


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

def preprocess_gray(img):
    """
    Convert image to grayscale and enhance contrast if needed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Low contrast correction
    if np.std(gray) < 30:
        gray = cv2.equalizeHist(gray)

    return gray


# ------------------------------------------------------------
# Barcode decoding safety wrapper
# ------------------------------------------------------------

@contextlib.contextmanager
def suppress_stderr():
    """
    Suppress noisy stderr output from native libraries.
    """
    try:
        devnull = open(os.devnull, "w")
        old_stderr = sys.stderr
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()


def safe_decode_image(img):
    """
    Safely attempt barcode decoding while suppressing native errors.
    """
    try:
        with suppress_stderr():
            return decode(img) or []
    except Exception:
        return []


# ------------------------------------------------------------
# Barcode decoding strategies
# ------------------------------------------------------------

def try_decode(img):
    """
    Attempt barcode decoding using multiple preprocessing variants.
    """
    # Ensure grayscale input
    if len(img.shape) == 2:
        target = img
    else:
        target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpen filter
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    proc = cv2.filter2D(target, -1, kernel)

    # Multiple threshold variants
    variants = [
        proc,
        cv2.threshold(
            proc, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        cv2.adaptiveThreshold(
            proc, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            15, 9
        )
    ]

    for v in variants:
        decoded_objs = safe_decode_image(v)
        if decoded_objs:
            return [
                d.data.decode("utf-8", errors="ignore").strip()
                for d in decoded_objs
            ]

    return []


def decode_pipeline(png_path, coords_scaled):
    """
    Full decoding pipeline including:
    - ROI cropping
    - Image scaling
    - Rotation attempts
    """
    img = cv2.imread(png_path)
    if img is None:
        return None

    candidates = []

    # Add cropped ROI candidate
    if coords_scaled:
        crop = crop_with_coords(img, coords_scaled)
        if crop is not None:
            candidates.append(crop)

    # Add full image fallback
    candidates.append(img)

    # Add scaled versions for small images
    for cand in list(candidates):
        h = cand.shape[0]
        if h < 250:
            scale = int(400 / max(1, h))
            nw = int(cand.shape[1] * scale)
            resized = cv2.resize(
                cand,
                (nw, 400),
                interpolation=cv2.INTER_LINEAR
            )
            candidates.append(resized)

    # Attempt decoding on all candidates and rotations
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


# ------------------------------------------------------------
# File movement helpers
# ------------------------------------------------------------

def safe_move(src, dst_dir):
    """
    Move a file into a directory while avoiding name collisions.
    """
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


# ------------------------------------------------------------
# File processing pipeline
# ------------------------------------------------------------

def process_file(path, cfg):
    """
    End-to-end pipeline for a single PDF file.
    """
    with lock:
        if path in active_processing:
            return
        active_processing.add(path)

    try:
        # Wait until file is fully written
        if not wait_for_stable(
            path,
            cfg["stable_seconds"],
            cfg["max_wait"]
        ):
            return

        # Convert PDF to image
        png = pdf_to_png(path, cfg["dpi"])
        if not png or not os.path.exists(png):
            log_error("PDF conversion failed")
            safe_move(path, cfg["error_dir"])
            return

        # Scale ROI coordinates
        coords = cfg.get("barcode_coords")
        coords_scaled = (
            scale_coords(coords, cfg.get("coords_dpi", 600), cfg["dpi"])
            if coords else None
        )

        # Decode barcode
        decoded = decode_pipeline(png, coords_scaled)
        log_info(f"Barcode OCR: {decoded}")

        if decoded:
            for d in decoded:
                digits = "".join(ch for ch in d if ch.isdigit())
                if len(digits) == 7:
                    target_name = f"{digits}.pdf"

                    os.makedirs(cfg["output_dir"], exist_ok=True)
                    dest = os.path.join(cfg["output_dir"], target_name)

                    # Avoid overwrite
                    base, ext = os.path.splitext(target_name)
                    i = 1
                    while os.path.exists(dest):
                        dest = os.path.join(
                            cfg["output_dir"],
                            f"{base}_{i}{ext}"
                        )
                        i += 1

                    shutil.copyfile(path, dest)
                    log_success(digits)

                    # Cleanup
                    try:
                        os.remove(png)
                    except Exception:
                        pass

                    try:
                        os.remove(path)
                    except Exception:
                        pass

                    return

        # Barcode not detected
        log_error("Barcode not found")

        try:
            os.remove(png)
        except Exception:
            pass

        safe_move(path, cfg["error_dir"])

    finally:
        with lock:
            active_processing.discard(path)


# ------------------------------------------------------------
# Folder monitoring loop
# ------------------------------------------------------------

def monitor(cfg):
    """
    Continuously scan watch directory and submit jobs to process pool.
    """
    pool = ProcessPoolExecutor(max_workers=cfg["workers"])
    futures = {}
    start_time = time.time()

    try:
        while True:
            try:
                for fname in os.listdir(cfg["watch_dir"]):
                    if not fname.lower().endswith(".pdf"):
                        continue

                    full = os.path.join(cfg["watch_dir"], fname)

                    # Skip active futures
                    if full in futures:
                        if futures[full].done():
                            futures.pop(full, None)
                        else:
                            continue

                    futures[full] = pool.submit(process_file, full, cfg)

            except Exception:
                pass

            time.sleep(1)

            # Scheduled restart
            if time.time() - start_time > RESTART_INTERVAL_SECONDS:
                log_restart("Daily restart")
                break

    finally:
        pool.shutdown(wait=True)


# ------------------------------------------------------------
# Application entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    log_info("Picking Slip OCR started")

    # Auto-restart loop for resilience
    while True:
        try:
            cfg = load_config("pickingslipconfig.json")
            monitor(cfg)
            time.sleep(3)
        except Exception as e:
            log_error(f"Fatal error: {e}")
            time.sleep(5)
