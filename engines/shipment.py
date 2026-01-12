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

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ISSA-OCRD\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

RESTART_INTERVAL_SECONDS = 6 * 3600

def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"{now()} - {msg}")

def load_config():
    with open("shipmentconfig.json", "r") as f:
        return json.load(f)

def remove_jpgs_in_dirs(dirs):
    for d in dirs:
        try:
            if not d:
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    if fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                        try:
                            os.remove(os.path.join(root, fn))
                        except Exception:
                            pass
        except Exception:
            pass

def _save_image_any(img, path):
    try:
        if hasattr(img, "save"):  # PIL Image
            img.save(path, "JPEG")
        else:
            # assume numpy array (cv2 BGR or RGB). Try cv2 if available.
            try:
                import cv2
                # if color, ensure BGR->RGB handled by caller
                cv2.imwrite(path, img)
            except Exception:
                # fallback: try PIL conversion
                from PIL import Image
                if img.ndim == 3:
                    im = Image.fromarray(img)
                else:
                    im = Image.fromarray(img)
                im.save(path, "JPEG")
        return True
    except Exception:
        return False

def save_roi_debug(error_dir, crop_img, full_img, keep=5):
    """
    Save crop_img and full_img into error_dir/roi_debug with timestamped names.
    Also write roi_crop_latest.jpg and roi_full_latest.jpg (overwritten).
    Prune older files, keeping at most 'keep' crop files and 'keep' full files.
    crop_img and full_img can be PIL.Image or numpy array.
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
        # latest links/overwrites
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
        # prune separately crop and full files
        def prune(prefix):
            files = []
            for fn in os.listdir(roi_debug_dir):
                if fn.startswith(prefix) and fn.lower().endswith(".jpg"):
                    files.append(os.path.join(roi_debug_dir, fn))
            if len(files) <= keep:
                return
            files.sort(key=lambda p: os.path.getmtime(p))
            to_remove = files[:len(files)-keep]
            for p in to_remove:
                try:
                    os.remove(p)
                except Exception:
                    pass
        prune("roi_crop_")
        prune("roi_full_")
    except Exception:
        pass

def convert_pdf_to_image(pdf_path, output_dir):
    from pdf2image import convert_from_path
    try:
        images = convert_from_path(pdf_path, dpi=300, fmt='jpeg')
        if images:
            image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.jpg")
            images[0].save(image_path, "JPEG")
            return image_path
    except Exception as e:
        log(f"Error converting PDF: {e}")
    return None

def extract_order_number_from_image(image: Image.Image, crop_box, error_dir) -> str:
    # Save ROI debug images and prune
    cropped_image = image.crop(crop_box)
    full_with_box = image.copy()
    draw = ImageDraw.Draw(full_with_box)
    draw.rectangle(crop_box, outline="green", width=3)
    save_roi_debug(error_dir, cropped_image, full_with_box, keep=5)

    cropped_path = os.path.join(error_dir, "roi_debug", "roi_crop_latest.jpg")
    log(f"Saved cropped ROI to: {cropped_path}")

    text = pytesseract.image_to_string(cropped_image)
    log(f"Extracted text length {len(text)}")

    match = re.search(r"\b(\d{5})\b", text)
    if match:
        return match.group(1)
    return None

def process_file(file_path, config):
    output_dir = config["output_folder"]
    error_dir = config["error_folder"]
    crop_box = config["crop_box"]

    log(f"Processing file {file_path}")
    image_path = convert_pdf_to_image(file_path, output_dir)
    if not image_path:
        log("PDF conversion failed, moving to error folder")
        try:
            shutil.move(file_path, os.path.join(error_dir, os.path.basename(file_path)))
        except Exception:
            pass
        return

    try:
        image = Image.open(image_path)
        order_number = extract_order_number_from_image(image, crop_box, error_dir)
        if order_number:
            new_filename = f"{order_number}.pdf"
            new_path = os.path.join(output_dir, new_filename)
            shutil.copyfile(file_path, new_path)
            log(f"Saved as: {new_path}")
            try:
                os.remove(file_path)
            except Exception:
                pass
            try:
                os.remove(image_path)
            except Exception:
                pass
        else:
            # save ROI debug (already saved in extract), move originals to error
            log("No valid order number found, moving to error")
            try:
                shutil.move(file_path, os.path.join(error_dir, os.path.basename(file_path)))
            except Exception:
                pass
            if os.path.exists(image_path):
                try:
                    shutil.move(image_path, os.path.join(error_dir, os.path.basename(image_path)))
                except Exception:
                    pass
    except Exception as e:
        log(f"Error processing file: {e}")
        try:
            shutil.move(file_path, os.path.join(error_dir, os.path.basename(file_path)))
        except Exception:
            pass
        if os.path.exists(image_path):
            try:
                shutil.move(image_path, os.path.join(error_dir, os.path.basename(image_path)))
            except Exception:
                pass
    finally:
        # cleanup images in key folders
        remove_jpgs_in_dirs([config.get("watch_folder"), config.get("output_folder"), config.get("error_folder")])

def is_file_stable(path, wait=1.0):
    try:
        if not os.path.exists(path):
            return False
        size1 = os.path.getsize(path)
        time.sleep(wait)
        size2 = os.path.getsize(path)
        return size1 == size2
    except Exception:
        return False

class MyHandler(FileSystemEventHandler):
    def __init__(self, config):
        self.config = config

    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".pdf"):
            return
        if is_file_stable(event.src_path):
            process_file(event.src_path, self.config)
        else:
            time.sleep(1.5)
            if is_file_stable(event.src_path):
                process_file(event.src_path, self.config)

def process_existing_files(config):
    for file in os.listdir(config["watch_folder"]):
        if file.lower().endswith(".pdf"):
            full = os.path.join(config["watch_folder"], file)
            if is_file_stable(full):
                process_file(full, config)

def run_one_cycle():
    config = load_config()
    log("Shipment OCR started")
    # initial cleanup of images
    remove_jpgs_in_dirs([config.get("watch_folder"), config.get("output_folder"), config.get("error_folder")])
    process_existing_files(config)

    event_handler = MyHandler(config)
    observer = Observer()
    observer.schedule(event_handler, path=config["watch_folder"], recursive=False)
    observer.start()
    start_time = time.time()
    try:
        while True:
            time.sleep(2)
            if time.time() - start_time > RESTART_INTERVAL_SECONDS:
                log("Scheduled restart: stopping observer for restart")
                observer.stop()
                break
    except KeyboardInterrupt:
        log("Keyboard interrupt received, stopping")
        observer.stop()
    observer.join()
    log("Observer stopped, cycle end")

if __name__ == "__main__":
    while True:
        try:
            run_one_cycle()
            log("Restarting main loop now")
            time.sleep(2)
        except Exception as e:
            log(f"Fatal error in main loop: {e}")
            time.sleep(5)
