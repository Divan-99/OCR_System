"""
Shipment OCR Processor

Purpose:
- Monitor a folder for incoming shipment PDFs.
- Convert the first page of each PDF into a JPEG image.
- Crop a fixed region where the shipment order number appears.
- Run OCR using Tesseract.
- Extract a 5-digit order number using regex.
- Rename and move the PDF into the output folder.
- Move failed files into the error folder.
- Save ROI debug images for tuning and troubleshooting.
- Run continuously with scheduled restarts.

Design Notes:
- Identical structure to purchase order pipeline for consistency.
- ROI debug images allow rapid calibration of crop coordinates.
- Aggressive cleanup prevents disk bloat over time.
"""

import os
import time
import json
import re
import shutil
from datetime import datetime
from PIL import Image, ImageDraw
import pytesseract
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# ------------------------------------------------------------
# Tesseract configuration
# ------------------------------------------------------------

# Explicit path to Tesseract binary on Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\ISSA-OCRD\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)


# ------------------------------------------------------------
# Runtime constants
# ------------------------------------------------------------

# Restart watcher every 6 hours for stability
RESTART_INTERVAL_SECONDS = 6 * 3600


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

def now():
    """
    Return current UTC timestamp string.
    """
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    """
    Print timestamped log message.
    """
    print(f"{now()} - {msg}")


# ------------------------------------------------------------
# Configuration loader
# ------------------------------------------------------------

def load_config():
    """
    Load JSON configuration file.
    """
    with open("shipmentconfig.json", "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# Cleanup utilities
# ------------------------------------------------------------

def remove_jpgs_in_dirs(dirs):
    """
    Recursively remove image files from provided directories.
    Prevents disk usage growth.
    """
    for d in dirs:
        try:
            if not d:
                continue

            for root, _, files in os.walk(d):
                for fn in files:
                    if fn.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".tif", ".tiff")
                    ):
                        try:
                            os.remove(os.path.join(root, fn))
                        except Exception:
                            pass
        except Exception:
            pass


# ------------------------------------------------------------
# Image saving abstraction
# ------------------------------------------------------------

def _save_image_any(img, path):
    """
    Save image regardless of whether it is PIL or NumPy format.
    """
    try:
        if hasattr(img, "save"):
            img.save(path, "JPEG")
        else:
            try:
                import cv2
                cv2.imwrite(path, img)
            except Exception:
                from PIL import Image
                im = Image.fromarray(img)
                im.save(path, "JPEG")
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# ROI debug capture
# ------------------------------------------------------------

def save_roi_debug(error_dir, crop_img, full_img, keep=5):
    """
    Save cropped ROI image and full image with ROI overlay.
    Keeps only the latest N images.
    """
    try:
        roi_debug_dir = os.path.join(error_dir, "roi_debug")
        os.makedirs(roi_debug_dir, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        crop_name = f"roi_crop_{ts}.jpg"
        full_name = f"roi_full_{ts}.jpg"

        crop_path = os.path.join(roi_debug_dir, crop_name)
        full_path = os.path.join(roi_debug_dir, full_name)

        _save_image_any(crop_img, crop_path)
        _save_image_any(full_img, full_path)

        # Maintain latest snapshot
        latest_crop = os.path.join(roi_debug_dir, "roi_crop_latest.jpg")
        latest_full = os.path.join(roi_debug_dir, "roi_full_latest.jpg")

        try:
            if os.path.exists(latest_crop):
                os.remove(latest_crop)
            if os.path.exists(latest_full):
                os.remove(latest_full)
        except Exception:
            pass

        try:
            _save_image_any(crop_img, latest_crop)
            _save_image_any(full_img, latest_full)
        except Exception:
            pass

        # Remove older debug images
        def prune(prefix):
            files = []
            for fn in os.listdir(roi_debug_dir):
                if fn.startswith(prefix) and fn.lower().endswith(".jpg"):
                    files.append(os.path.join(roi_debug_dir, fn))

            if len(files) <= keep:
                return

            files.sort(key=lambda p: os.path.getmtime(p))
            to_remove = files[:len(files) - keep]

            for p in to_remove:
                try:
                    os.remove(p)
                except Exception:
                    pass

        prune("roi_crop_")
        prune("roi_full_")

    except Exception:
        pass


# ------------------------------------------------------------
# PDF conversion
# ------------------------------------------------------------

def convert_pdf_to_image(pdf_path, output_dir):
    """
    Convert first page of PDF into JPEG image.
    """
    from pdf2image import convert_from_path

    try:
        images = convert_from_path(
            pdf_path,
            dpi=300,
            fmt="jpeg"
        )

        if images:
            image_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}.jpg"
            )
            images[0].save(image_path, "JPEG")
            return image_path

    except Exception as e:
        log(f"Error converting PDF: {e}")

    return None


# ------------------------------------------------------------
# OCR extraction
# ------------------------------------------------------------

def extract_order_number_from_image(
    image: Image.Image,
    crop_box,
    error_dir
) -> str:
    """
    Crop ROI, run OCR, and extract 5-digit order number.
    """
    # Crop ROI
    cropped_image = image.crop(crop_box)

    # Draw ROI rectangle for debugging
    full_with_box = image.copy()
    draw = ImageDraw.Draw(full_with_box)
    draw.rectangle(crop_box, outline="green", width=3)

    # Save debug images
    save_roi_debug(error_dir, cropped_image, full_with_box, keep=5)

    cropped_path = os.path.join(
        error_dir,
        "roi_debug",
        "roi_crop_latest.jpg"
    )
    log(f"Saved cropped ROI to: {cropped_path}")

    # OCR
    text = pytesseract.image_to_string(cropped_image)
    log(f"Extracted text length {len(text)}")

    # Extract exactly 5 digits
    match = re.search(r"\b(\d{5})\b", text)
    if match:
        return match.group(1)

    return None


# ------------------------------------------------------------
# File processing pipeline
# ------------------------------------------------------------

def process_file(file_path, config):
    """
    End-to-end processing for a single shipment PDF.
    """
    output_dir = config["output_folder"]
    error_dir = config["error_folder"]
    crop_box = config["crop_box"]

    log(f"Processing file {file_path}")

    # Convert PDF to image
    image_path = convert_pdf_to_image(file_path, output_dir)
    if not image_path:
        log("PDF conversion failed, moving to error folder")
        try:
            shutil.move(
                file_path,
                os.path.join(error_dir, os.path.basename(file_path))
            )
        except Exception:
            pass
        return

    try:
        image = Image.open(image_path)

        # Extract order number
        order_number = extract_order_number_from_image(
            image,
            crop_box,
            error_dir
        )

        if order_number:
            # Build output filename
            new_filename = f"{order_number}.pdf"
            new_path = os.path.join(output_dir, new_filename)

            shutil.copyfile(file_path, new_path)
            log(f"Saved as: {new_path}")

            # Cleanup originals
            try:
                os.remove(file_path)
            except Exception:
                pass

            try:
                os.remove(image_path)
            except Exception:
                pass

        else:
            # OCR failed
            log("No valid order number found, moving to error")

            try:
                shutil.move(
                    file_path,
                    os.path.join(error_dir, os.path.basename(file_path))
                )
            except Exception:
                pass

            if os.path.exists(image_path):
                try:
                    shutil.move(
                        image_path,
                        os.path.join(error_dir, os.path.basename(image_path))
                    )
                except Exception:
                    pass

    except Exception as e:
        log(f"Error processing file: {e}")

        try:
            shutil.move(
                file_path,
                os.path.join(error_dir, os.path.basename(file_path))
            )
        except Exception:
            pass

        if os.path.exists(image_path):
            try:
                shutil.move(
                    image_path,
                    os.path.join(error_dir, os.path.basename(image_path))
                )
            except Exception:
                pass

    finally:
        # Cleanup leftover images
        remove_jpgs_in_dirs([
            config.get("watch_folder"),
            config.get("output_folder"),
            config.get("error_folder")
        ])


# ------------------------------------------------------------
# File stability check
# ------------------------------------------------------------

def is_file_stable(path, wait=1.0):
    """
    Ensure file size remains stable before processing.
    """
    try:
        if not os.path.exists(path):
            return False

        size1 = os.path.getsize(path)
        time.sleep(wait)
        size2 = os.path.getsize(path)

        return size1 == size2

    except Exception:
        return False


# ------------------------------------------------------------
# Watchdog event handler
# ------------------------------------------------------------

class MyHandler(FileSystemEventHandler):
    """
    Handle filesystem events for new shipment PDFs.
    """

    def __init__(self, config):
        self.config = config

    def on_created(self, event):
        """
        Trigger processing when a new PDF appears.
        """
        if event.is_directory or not event.src_path.lower().endswith(".pdf"):
            return

        if is_file_stable(event.src_path):
            process_file(event.src_path, self.config)
        else:
            time.sleep(1.5)
            if is_file_stable(event.src_path):
                process_file(event.src_path, self.config)


# ------------------------------------------------------------
# Startup scan
# ------------------------------------------------------------

def process_existing_files(config):
    """
    Process any PDFs already present when the app starts.
    """
    for file in os.listdir(config["watch_folder"]):
        if file.lower().endswith(".pdf"):
            full = os.path.join(config["watch_folder"], file)
            if is_file_stable(full):
                process_file(full, config)


# ------------------------------------------------------------
# Runtime control loop
# ------------------------------------------------------------

def run_one_cycle():
    """
    Run watcher until scheduled restart.
    """
    config = load_config()
    log("Shipment OCR started")

    # Initial cleanup
    remove_jpgs_in_dirs([
        config.get("watch_folder"),
        config.get("output_folder"),
        config.get("error_folder")
    ])

    # Process existing files
    process_existing_files(config)

    # Setup watchdog observer
    event_handler = MyHandler(config)
    observer = Observer()
    observer.schedule(
        event_handler,
        path=config["watch_folder"],
        recursive=False
    )
    observer.start()

    start_time = time.time()

    try:
        while True:
            time.sleep(2)

            # Scheduled restart
            if time.time() - start_time > RESTART_INTERVAL_SECONDS:
                log("Scheduled restart: stopping observer for restart")
                observer.stop()
                break

    except KeyboardInterrupt:
        log("Keyboard interrupt received, stopping")
        observer.stop()

    observer.join()
    log("Observer stopped, cycle end")


# ------------------------------------------------------------
# Application entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    while True:
        try:
            run_one_cycle()
            log("Restarting main loop now")
            time.sleep(2)
        except Exception as e:
            log(f"Fatal error in main loop: {e}")
            time.sleep(5)
