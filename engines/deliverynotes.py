"""
Delivery Notes OCR Processor

Purpose:
- Monitor a folder for incoming PDF delivery notes.
- Convert the first page of each PDF into an image.
- Preprocess the image for OCR accuracy.
- Crop a defined region of interest (ROI).
- Extract a 7-digit document number using EasyOCR.
- Rename and move the PDF into the output folder using the extracted number.
- Move failed files into an error folder.
- Run continuously with automatic recovery and daily restarts.

Key Features:
- Threaded processing for parallel files.
- File stability checking to avoid reading incomplete files.
- Automatic retry logic for startup scans.
- Safe deletion and error handling.
"""

import os
import sys
import json
import shutil
import easyocr
import cv2
import time
import concurrent.futures
from threading import Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pdf2image import convert_from_path
from datetime import datetime, timezone
import torch
from colorama import init, Fore


# ------------------------------------------------------------
# Console setup
# ------------------------------------------------------------

# Enables colored terminal output on Windows
init(autoreset=True)

# Disable OpenCV console spam
cv2.setLogLevel(0)


# ------------------------------------------------------------
# Global runtime constants
# ------------------------------------------------------------

# Restart the full watcher every 24 hours to prevent memory leaks
RESTART_INTERVAL_SECONDS = 24 * 3600

# Time to wait between file size checks
FILE_STABLE_WAIT = 3.0

# Retry logic when deleting files
DELETE_RETRY_COUNT = 3
DELETE_RETRY_DELAY = 1.0

# Startup scan retry behavior
STARTUP_SCAN_RETRIES = 5
STARTUP_SCAN_DELAY = 2.0


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

def now():
    """
    Returns the current UTC timestamp string.
    Used for consistent logging.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg):
    """
    Prints normal operational messages.
    """
    print(f"{Fore.WHITE}{now()} {msg}")


def log_success(msg):
    """
    Prints successful OCR or processing messages.
    """
    print(f"{Fore.WHITE}{now()} {Fore.GREEN}{msg}")


def log_error(msg):
    """
    Prints error messages.
    """
    print(f"{Fore.WHITE}{now()} {Fore.RED}{msg}")


def log_restart(msg):
    """
    Prints restart messages.
    """
    print(f"{Fore.WHITE}{now()} {Fore.MAGENTA}{msg}")


# ------------------------------------------------------------
# Main OCR processing engine
# ------------------------------------------------------------

class DeliveryNotesProcessor:
    """
    Core processing engine.

    Responsibilities:
    - Load and validate configuration.
    - Initialize OCR engine.
    - Watch and process PDFs.
    - Convert PDF to image.
    - Preprocess image for OCR.
    - Extract document number from ROI.
    - Rename and move files safely.
    """

    def __init__(self, config_file):
        """
        Initialize the processor and prepare runtime resources.
        """
        # Load JSON configuration
        self.config = self.load_config(config_file)

        # OCR reader instance
        self.reader = None

        # Thread pool for parallel file processing
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 3)
        )

        # Temporary folder for image files
        self.tmp_dir = os.path.join(
            self.config.get(
                "temp_dir",
                os.path.join(self.config.get("output_dir", "."), "tmp_images")
            )
        )

        # Ensure required directories exist
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.config.get("error_dir", "./error"), exist_ok=True)
        os.makedirs(self.config.get("output_dir", "./output"), exist_ok=True)

        # Track currently processing files to avoid duplicates
        self.active_processing = set()
        self.active_lock = Lock()

        # Initialize OCR engine
        self.initialize_ocr()

    # --------------------------------------------------------

    def load_config(self, config_file):
        """
        Load and validate the configuration file.

        Required fields:
        - watch_dir
        - output_dir
        - error_dir
        - document_number_coords
        - ocr_lang
        """
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)

            # Validate required keys
            for k in ["watch_dir", "output_dir", "error_dir", "document_number_coords", "ocr_lang"]:
                if k not in cfg:
                    raise KeyError(k)

            # Normalize paths
            cfg["watch_dir"] = os.path.abspath(cfg["watch_dir"])
            cfg["output_dir"] = os.path.abspath(cfg["output_dir"])
            cfg["error_dir"] = os.path.abspath(cfg["error_dir"])

            # Optional defaults
            cfg.setdefault("poppler_path", None)
            cfg.setdefault("temp_dir", os.path.join(cfg["output_dir"], "tmp_images"))
            cfg.setdefault("max_workers", 3)

            return cfg

        except Exception as e:
            log_error(f"Config load failed: {e}")
            sys.exit(1)

    # --------------------------------------------------------

    def initialize_ocr(self):
        """
        Initialize EasyOCR.
        Attempts GPU usage first, then falls back to CPU.
        """
        try:
            use_gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader([self.config["ocr_lang"]], gpu=use_gpu)
        except Exception:
            try:
                self.reader = easyocr.Reader([self.config["ocr_lang"]], gpu=False)
            except Exception as e:
                log_error(f"OCR init failed: {e}")
                sys.exit(1)

    # --------------------------------------------------------

    def delete_file_retry(self, path):
        """
        Safely delete a file with retry logic.
        Prevents failures from file locks or delayed releases.
        """
        if not path:
            return

        for _ in range(DELETE_RETRY_COUNT):
            try:
                if os.path.exists(path):
                    os.remove(path)
                return
            except Exception:
                time.sleep(DELETE_RETRY_DELAY)

    # --------------------------------------------------------

    def move_pdf_to_error(self, pdf_path):
        """
        Move a failed PDF into the error directory.
        Auto-increments filename if duplicates exist.
        """
        try:
            if not pdf_path or not os.path.exists(pdf_path):
                return

            base = os.path.basename(pdf_path)
            dst = os.path.join(self.config["error_dir"], base)
            counter = 1

            while os.path.exists(dst):
                name, ext = os.path.splitext(base)
                dst = os.path.join(self.config["error_dir"], f"{name}_{counter}{ext}")
                counter += 1

            shutil.move(pdf_path, dst)
            log_error(f"Moved to error: {os.path.basename(dst)}")

        except Exception as e:
            log_error(f"Failed moving to error folder: {e}")

    # --------------------------------------------------------

    def pdf_to_image(self, pdf_path):
        """
        Convert the first page of a PDF into a JPEG image.
        """
        try:
            images = convert_from_path(
                pdf_path,
                dpi=300,
                poppler_path=self.config.get("poppler_path"),
                first_page=1,
                last_page=1,
            )

            if not images:
                return None

            base = os.path.splitext(os.path.basename(pdf_path))[0]
            out_path = os.path.join(self.tmp_dir, f"{base}_page1.jpg")
            images[0].save(out_path, "JPEG")

            return out_path

        except Exception:
            return None

    # --------------------------------------------------------

    def preprocess_image(self, image_path):
        """
        Convert image to grayscale and apply binary thresholding.
        Improves OCR accuracy.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            preprocessed_path = os.path.join(
                self.tmp_dir, f"preprocessed_{os.path.basename(image_path)}"
            )

            cv2.imwrite(preprocessed_path, binary)
            return preprocessed_path

        except Exception:
            return image_path

    # --------------------------------------------------------

    def extract_document_number(self, image_path):
        """
        Crop the configured ROI and extract a 7-digit number using OCR.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            # ROI coordinates from config
            x_min, y_min, x_max, y_max = self.config["document_number_coords"]
            h, w = img.shape[:2]

            # Clamp coordinates safely
            x_min = max(0, min(w - 1, int(x_min)))
            x_max = max(0, min(w, int(x_max)))
            y_min = max(0, min(h - 1, int(y_min)))
            y_max = max(0, min(h, int(y_max)))

            if x_max <= x_min or y_max <= y_min:
                return None

            # Crop ROI
            roi = img[y_min:y_max, x_min:x_max]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Run OCR
            result = self.reader.readtext(gray_roi, detail=0)
            log_info(f"ROI OCR: {result}")

            # Validate extracted text
            for text in result:
                normalized = text.replace(" ", "")
                if normalized.isdigit() and len(normalized) == 7:
                    return normalized

            return None

        except Exception:
            return None

    # --------------------------------------------------------

    def is_file_stable(self, path, wait=FILE_STABLE_WAIT):
        """
        Check if a file size remains constant over time.
        Prevents reading partially copied files.
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

    # --------------------------------------------------------

    def process_file(self, file_path):
        """
        Main pipeline for processing a single PDF file.
        """
        real_path = os.path.realpath(file_path)

        # Prevent duplicate processing
        with self.active_lock:
            if real_path in self.active_processing:
                return
            self.active_processing.add(real_path)

        try:
            if not os.path.exists(real_path):
                return

            # Wait for file stability
            stable = False
            for _ in range(5):
                if self.is_file_stable(real_path):
                    stable = True
                    break
                time.sleep(1.0)

            if not stable:
                return

            # Convert PDF to image
            image_path = self.pdf_to_image(real_path)
            if not image_path:
                log_error("PDF conversion failed")
                self.move_pdf_to_error(real_path)
                return

            # Preprocess image
            preprocessed_image_path = self.preprocess_image(image_path)

            # Extract document number
            document_number = self.extract_document_number(preprocessed_image_path)

            if document_number:
                # Build output filename
                new_filename = f"{document_number}.pdf"
                new_filepath = os.path.join(self.config["output_dir"], new_filename)

                # Prevent overwrite
                base, ext = os.path.splitext(new_filepath)
                counter = 1
                while os.path.exists(new_filepath):
                    new_filepath = f"{base}_{counter}{ext}"
                    counter += 1

                try:
                    # Copy file to output
                    shutil.copyfile(real_path, new_filepath)
                    time.sleep(0.5)

                    # Cleanup temp files and original
                    self.delete_file_retry(preprocessed_image_path)
                    self.delete_file_retry(image_path)
                    self.delete_file_retry(real_path)

                    log_success(document_number)

                except Exception as e:
                    log_error(f"File move failed: {e}")
                    self.move_pdf_to_error(real_path)

            else:
                # OCR failed
                log_error("OCR failed")
                self.delete_file_retry(preprocessed_image_path)
                self.delete_file_retry(image_path)
                self.move_pdf_to_error(real_path)

        finally:
            # Release active processing lock
            with self.active_lock:
                self.active_processing.discard(real_path)

    # --------------------------------------------------------

    def process_existing_files_with_retry(self):
        """
        Scan watch folder on startup and retry unstable files.
        """
        watch = self.config["watch_dir"]

        for _ in range(STARTUP_SCAN_RETRIES):
            try:
                files = [
                    os.path.join(watch, f)
                    for f in os.listdir(watch)
                    if f.lower().endswith(".pdf")
                ]

                if not files:
                    return

                pending = []

                for path in files:
                    if self.is_file_stable(path):
                        self.executor.submit(self.process_file, path)
                    else:
                        pending.append(path)

                if not pending:
                    return

                time.sleep(STARTUP_SCAN_DELAY)

            except Exception:
                pass


# ------------------------------------------------------------
# Filesystem event handler
# ------------------------------------------------------------

class FileEventHandler(FileSystemEventHandler):
    """
    Receives filesystem events and schedules file processing.
    """

    def __init__(self, processor):
        self.processor = processor

    def _schedule(self, path):
        """
        Submit a PDF for processing.
        """
        try:
            if os.path.isfile(path) and path.lower().endswith(".pdf"):
                self.processor.executor.submit(
                    self.processor.process_file, path
                )
        except Exception:
            pass

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            dest = getattr(event, "dest_path", None)
            if dest:
                self._schedule(dest)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)


# ------------------------------------------------------------
# Runtime control loop
# ------------------------------------------------------------

def run_one_cycle():
    """
    Run the watcher until a scheduled restart or shutdown occurs.
    """
    cfg_file = "deliverynotesconfig.json"
    processor = DeliveryNotesProcessor(cfg_file)
    watch_dir = processor.config["watch_dir"]

    if not os.path.isdir(watch_dir):
        log_error("Watch directory missing")
        sys.exit(1)

    # Allow filesystem to settle
    time.sleep(3)

    # Process any files already in the folder
    processor.process_existing_files_with_retry()

    # Setup watchdog observer
    handler = FileEventHandler(processor)
    observer = Observer()
    observer.schedule(handler, path=watch_dir, recursive=False)
    observer.start()

    start_time = time.time()
    last_rescan = time.time()

    try:
        while True:
            time.sleep(5)

            # Periodic rescan safety net
            if time.time() - last_rescan > 120:
                processor.process_existing_files_with_retry()
                last_rescan = time.time()

            # Scheduled restart
            if time.time() - start_time > RESTART_INTERVAL_SECONDS:
                log_restart("Daily restart")
                observer.stop()
                break

    except KeyboardInterrupt:
        observer.stop()

    observer.join()


# ------------------------------------------------------------
# Application entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    log_info("Delivery Notes OCR started")

    # Auto-restart loop for resilience
    while True:
        try:
            run_one_cycle()
            time.sleep(3)
        except Exception as e:
            log_error(f"Fatal error: {e}")
            time.sleep(5)
