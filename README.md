Title
OCR System

Description
A configurable Python OCR platform for extracting document numbers and barcodes from PDF documents. The system supports multiple document types using modular engines and JSON-based configuration. No code edits are required to change behavior.

Supported engines

Delivery Note
Landscape documents. Extracts numeric identifiers from a fixed region using OCR.

Purchase Order
Portrait documents. Extracts numeric identifiers from a fixed region using OCR.

Shipment
Multi-page documents. Extracts the identifier from the relevant page only.

Picking Slip
Barcode-based extraction using Code128 and similar formats.

Core features

Folder monitoring using watchdog.

PDF to image conversion using pdf2image.

OCR using EasyOCR and Tesseract.

Barcode decoding using pyzbar.

ROI based extraction.

Automatic file renaming.

Error handling and debug image capture.

Fully configurable via JSON.

Project structure

core
Shared utilities and helpers.

engines
Document specific processors.

configs
Example configuration files.

examples
Sample input files.

docs
Architecture and diagrams.

How it works

You configure paths and OCR settings in a JSON file.

The engine watches the input folder.

When a PDF appears, it waits until the file is stable.

The PDF is converted into an image.

The ROI region is extracted.

OCR or barcode decoding runs.

The extracted number becomes the new filename.

The renamed PDF moves to the output folder.

Failed files move to the error folder with debug images.

Requirements

Python 3.9 or newer

poppler installed for pdf2image

Tesseract installed if using OCR engines

Python packages

watchdog

pdf2image

easyocr

pytesseract

opencv-python

pyzbar

pillow

torch

numpy

Installation

Clone the repository.

Create a virtual environment.

Install dependencies.

Edit a config file in configs.

Run the engine you need.

Example

python engines/delivery_note.py

License
MIT License
