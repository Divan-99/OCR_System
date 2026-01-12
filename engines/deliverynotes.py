import os
import sys
import json
import shutil
import easyocr
import cv2
import time
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pdf2image import convert_from_path
from datetime import datetime
import torch
from PIL import Image, ImageDraw

RESTART_INTERVAL_SECONDS = 6 * 3600
INACTIVITY_LIMIT = 600

def now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"{now()} - {msg}")

logger = logging.getLogger("deliverynotes")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
log_file = "deliverynotes.log"
rotating_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
rotating_handler.setFormatter(formatter)
logger.addHandler(rotating_handler)

class DeliveryNotesProcessor:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.last_processed = {}
        self.last_processed_lock = Lock()
        self.last_activity = time.time()
        self.reader = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.get("max_workers", 3))
        self.tmp_dir = os.path.join(self.config.get("temp_dir", os.path.join(self.config.get("output_dir", "."), "tmp_images")))
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.config.get("error_dir", "./error"), exist_ok=True)
        os.makedirs(self.config.get("output_dir", "./output"), exist_ok=True)
        self.initialize_ocr()

    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
            for k in ["watch_dir", "output_dir", "error_dir", "document_number_coords", "ocr_lang"]:
                if k not in cfg:
                    raise KeyError(f"Missing required config key: {k}")
            cfg["watch_dir"] = os.path.abspath(cfg["watch_dir"])
            cfg["output_dir"] = os.path.abspath(cfg["output_dir"])
            cfg["error_dir"] = os.path.abspath(cfg["error_dir"])
            cfg.setdefault("poppler_path", None)
            cfg.setdefault("temp_dir", os.path.join(cfg["output_dir"], "tmp_images"))
            cfg.setdefault("max_workers", 3)
            return cfg
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)

    def initialize_ocr(self):
        try:
            use_gpu = torch.cuda.is_available()
            log(f"Initializing OCR Reader, gpu available: {use_gpu}")
            self.reader = easyocr.Reader([self.config["ocr_lang"]], gpu=use_gpu)
            log("OCR Reader initialized")
        except Exception as e:
            logger.error(f"OCR init failed: {e}")
            try:
                self.reader = easyocr.Reader([self.config["ocr_lang"]], gpu=False)
                log("OCR Reader initialized on CPU as fallback")
            except Exception as e2:
                logger.error(f"Failed to initialize OCR Reader on CPU: {e2}")
                sys.exit(1)

    def remove_jpgs_in_dirs(self, dirs):
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

    def _save_image_any(self, img, path):
        try:
            if hasattr(img, "save"):
                img.save(path, "JPEG")
            else:
                import cv2
                cv2.imwrite(path, img)
            return True
        except Exception:
            return False

    def save_roi_debug(self, error_dir, crop_img, full_img, keep=5):
        try:
            roi_debug_dir = os.path.join(error_dir, "roi_debug")
            os.makedirs(roi_debug_dir, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            crop_name = f"roi_crop_{ts}.jpg"
            full_name = f"roi_full_{ts}.jpg"
            crop_path = os.path.join(roi_debug_dir, crop_name)
            full_path = os.path.join(roi_debug_dir, full_name)
            self._save_image_any(crop_img, crop_path)
            self._save_image_any(full_img, full_path)
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
                self._save_image_any(crop_img, latest_crop)
                self._save_image_any(full_img, latest_full)
            except Exception:
                pass
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

    def update_activity(self):
        self.last_activity = time.time()

    def pdf_to_image(self, pdf_path):
        try:
            poppler_path = self.config.get("poppler_path")
            images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path, first_page=1, last_page=1)
            if not images:
                return None
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            out_path = os.path.join(self.tmp_dir, f"{base}_page1.jpg")
            images[0].save(out_path, "JPEG")
            return out_path
        except Exception as e:
            logger.error(f"Error converting PDF to image: {e}")
            return None

    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image for preprocessing: {image_path}")
                return image_path
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_path = os.path.join(self.tmp_dir, f"preprocessed_{os.path.basename(image_path)}")
            cv2.imwrite(preprocessed_path, binary)
            return preprocessed_path
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            return image_path

    def extract_document_number(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Image not found for OCR: {image_path}")
                return None
            coords = self.config["document_number_coords"]
            x_min, y_min, x_max, y_max = coords
            h, w = img.shape[:2]
            x_min = max(0, min(w - 1, int(x_min)))
            x_max = max(0, min(w, int(x_max)))
            y_min = max(0, min(h - 1, int(y_min)))
            y_max = max(0, min(h, int(y_max)))
            if x_max <= x_min or y_max <= y_min:
                logger.error("Invalid ROI coordinates")
                return None
            roi = img[y_min:y_max, x_min:x_max]
            # prepare images for saving: full with box (PIL)
            full_with_box = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(full_with_box)
            draw.rectangle((x_min, y_min, x_max, y_max), outline="green", width=3)
            # crop (PIL)
            cropped_pil = full_with_box.crop((x_min, y_min, x_max, y_max))
            # save debug ROI and prune
            self.save_roi_debug(self.config["error_dir"], cropped_pil, full_with_box, keep=5)
            roi_debug_path = os.path.join(self.config["error_dir"], "roi_debug", "roi_latest.jpg")
            logger.info(f"ROI debug saved: {roi_debug_path}")
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            result = self.reader.readtext(gray_roi, detail=0)
            logger.info(f"OCR Result from ROI: {result}")
            for text in result:
                normalized_text = text.replace(" ", "")
                if normalized_text.isdigit() and len(normalized_text) == 7:
                    return normalized_text
            return None
        except Exception as e:
            logger.error(f"Error extracting document number: {e}")
            return None

    def delete_file(self, file_path):
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File {file_path} deleted")
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e}")

    def move_file_to_error_dir(self, file_path):
        try:
            if not file_path or not os.path.exists(file_path):
                return
            base_name = os.path.basename(file_path)
            error_path = os.path.join(self.config["error_dir"], base_name)
            counter = 1
            while os.path.exists(error_path):
                name, ext = os.path.splitext(base_name)
                error_path = os.path.join(self.config["error_dir"], f"{name}_{counter}{ext}")
                counter += 1
            shutil.move(file_path, error_path)
            logger.info(f"Moved {file_path} to error directory: {error_path}")
        except Exception as e:
            logger.error(f"Error moving file to error dir: {e}")

    def is_file_stable(self, path, wait=1.0):
        try:
            if not os.path.exists(path):
                return False
            size1 = os.path.getsize(path)
            time.sleep(wait)
            size2 = os.path.getsize(path)
            return size1 == size2
        except Exception:
            return False

    def process_file(self, file_path):
        self.update_activity()
        real_path = os.path.realpath(file_path)
        current_time = time.time()

        with self.last_processed_lock:
            keys_to_remove = [k for k, v in self.last_processed.items() if current_time - v > 3600]
            for k in keys_to_remove:
                del self.last_processed[k]
            if real_path in self.last_processed and current_time - self.last_processed[real_path] < 5:
                logger.info(f"Skipping {file_path} as it was recently processed")
                return

        if not os.path.exists(real_path):
            logger.error(f"Error: File {real_path} does not exist")
            return

        if not self.is_file_stable(real_path, wait=1.0):
            logger.info(f"File {real_path} is still being written, skipping for now")
            return

        logger.info(f"Processing file: {real_path}")

        image_path = self.pdf_to_image(real_path)
        if not image_path:
            logger.error("Failed to convert PDF to image. Moving to error directory")
            self.move_file_to_error_dir(real_path)
            return

        preprocessed_image_path = self.preprocess_image(image_path)
        document_number = self.extract_document_number(preprocessed_image_path)

        if document_number:
            new_filename = f"{document_number}.pdf"
            new_filepath = os.path.join(self.config["output_dir"], new_filename)
            base, ext = os.path.splitext(new_filepath)
            counter = 1
            while os.path.exists(new_filepath):
                new_filepath = f"{base}_{counter}{ext}"
                counter += 1
            try:
                shutil.copyfile(real_path, new_filepath)
                logger.info(f"File copied to {new_filepath}")
                self.delete_file(preprocessed_image_path)
                self.delete_file(image_path)
                self.delete_file(real_path)
                with self.last_processed_lock:
                    self.last_processed[real_path] = time.time()
            except Exception as e:
                logger.error(f"Error finalizing file processing: {e}")
                self.move_file_to_error_dir(real_path)
        else:
            logger.warning("Document number not found or invalid. Moving to error directory")
            # save ROI debug already handled in extract_document_number
            self.move_file_to_error_dir(preprocessed_image_path)
            self.move_file_to_error_dir(image_path)
            self.move_file_to_error_dir(real_path)
        # cleanup images
        self.remove_jpgs_in_dirs([self.tmp_dir, self.config.get("error_dir"), self.config.get("output_dir")])

    def process_existing_files(self):
        logger.info("Checking for existing files...")
        watch = self.config["watch_dir"]
        try:
            for filename in os.listdir(watch):
                filepath = os.path.join(watch, filename)
                if os.path.isfile(filepath) and filepath.lower().endswith(".pdf"):
                    if self.is_file_stable(filepath):
                        self.executor.submit(self.process_file, filepath)
                    else:
                        logger.info(f"Skipping unstable file at startup: {filepath}")
        except Exception as e:
            logger.error(f"Error while scanning existing files: {e}")

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.processor.config.get("max_workers", 3))

    def _schedule_if_stable(self, path):
        try:
            if os.path.isfile(path) and path.lower().endswith(".pdf"):
                if self.processor.is_file_stable(path):
                    self.executor.submit(self.processor.process_file, path)
                else:
                    self.executor.submit(self._retry_process, path, 2.0)
        except Exception as e:
            logger.error(f"Error scheduling file {path}: {e}")

    def _retry_process(self, path, wait):
        time.sleep(wait)
        if self.processor.is_file_stable(path):
            self.processor.process_file(path)
        else:
            logger.info(f"File still unstable after retry: {path}")

    def on_created(self, event):
        if event.is_directory:
            return
        self._schedule_if_stable(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        dest_path = getattr(event, "dest_path", None)
        if dest_path:
            self._schedule_if_stable(dest_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        self._schedule_if_stable(event.src_path)

def run_one_cycle():
    cfg_file = "deliverynotesconfig.json"
    processor = DeliveryNotesProcessor(cfg_file)
    watch_dir = processor.config["watch_dir"]
    if not os.path.isdir(watch_dir):
        logger.error(f"Watch directory does not exist: {watch_dir}")
        sys.exit(1)

    # initial cleanup
    processor.remove_jpgs_in_dirs([processor.tmp_dir, processor.config.get("error_dir"), processor.config.get("output_dir")])

    processor.process_existing_files()

    event_handler = FileEventHandler(processor)
    observer = Observer()
    try:
        observer.schedule(event_handler, path=watch_dir, recursive=False)
    except Exception as e:
        logger.error(f"Failed to schedule observer: {e}")
        sys.exit(1)

    try:
        observer.start()
        logger.info("Observer started")
    except Exception as e:
        logger.error(f"Failed to start observer: {e}")
        sys.exit(1)

    start_time = time.time()
    try:
        while True:
            time.sleep(5)
            if time.time() - processor.last_activity > INACTIVITY_LIMIT:
                logger.info("No activity for inactivity limit. Exiting for restart")
                observer.stop()
                break
            if time.time() - start_time > RESTART_INTERVAL_SECONDS:
                logger.info("Scheduled restart: stopping observer for restart")
                observer.stop()
                break
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logger.info("Observer stopped, cycle end")

if __name__ == "__main__":
    while True:
        try:
            run_one_cycle()
            log("Delivery notes main loop restarting")
            time.sleep(2)
        except Exception as e:
            log(f"Fatal error in main loop: {e}")
            time.sleep(5)
