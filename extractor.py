import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pdfplumber
from pdfplumber.utils import extract_text
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

from models import ExtractionMethod, Page
from omr import correct_perspective, detect_omr_sheet

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 30
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def correct_reversed_text(text: str) -> str:
    if not text:
        return ""
    common_words = {
        "the", "and", "of", "to", "in", "is", "that", "it", "on", "for", 
        "with", "as", "are", "by", "this", "from", "an", "at", "be", "was", 
        "or", "which"
    }
    reversed_common = {w[::-1] for w in common_words}

    lines = text.split("\n")
    corrected_lines = []
    for line in lines:
        words = line.lower().split()
        rev_count = sum(1 for w in words if w.strip(".,;:?!()\"'-") in reversed_common)
        normal_count = sum(1 for w in words if w.strip(".,;:?!()\"'-") in common_words)

        if rev_count > normal_count and rev_count >= 1:
            corrected_lines.append(line[::-1])
        else:
            corrected_lines.append(line)

    return "\n".join(corrected_lines)


def extract_page_custom(page) -> str:
    upright_chars = [c for c in page.chars if c.get("upright") != False]
    rotated_chars = [c for c in page.chars if c.get("upright") == False]
    
    parts = []
    if upright_chars:
        text_upright = extract_text(upright_chars, x_tolerance=3, y_tolerance=3)
        if text_upright.strip():
            parts.append(text_upright.strip())
            
    if rotated_chars:
        c0 = rotated_chars[0]
        matrix = c0.get("matrix", (1, 0, 0, 1, 0, 0))
        H = page.height
        W = page.width
        
        rotated_chars_transformed = []
        for c in rotated_chars:
            c_new = c.copy()
            if len(matrix) >= 4 and matrix[1] < 0:
                # 90 degrees clockwise visual rotation
                c_new['x0'] = c['top']
                c_new['x1'] = c['bottom']
                c_new['top'] = W - c['x1']
                c_new['bottom'] = W - c['x0']
            else:
                # 90 degrees counter-clockwise visual rotation (default)
                c_new['x0'] = H - c['bottom']
                c_new['x1'] = H - c['top']
                c_new['top'] = c['x0']
                c_new['bottom'] = c['x1']
            rotated_chars_transformed.append(c_new)
            
        text_rotated = extract_text(rotated_chars_transformed, x_tolerance=8, y_tolerance=3)
        if text_rotated.strip():
            parts.append(text_rotated.strip())
            
    return "\n\n".join(parts)


def detect_file_type(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in SUPPORTED_IMAGES:
        return "image"
    raise ValueError(f"Unsupported file type: {ext}")


def extract(filepath: str, ocr_lang: str = "eng", dpi: int = 300, is_omr: bool = False) -> list[Page]:
    filepath = str(Path(filepath).resolve())
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    file_type = detect_file_type(filepath)
    if file_type == "image":
        return _extract_image(filepath, ocr_lang, is_omr)
    return _extract_pdf(filepath, ocr_lang, dpi, is_omr)


def _extract_pdf(filepath: str, ocr_lang: str, dpi: int, is_omr: bool) -> list[Page]:
    pages: list[Page] = []

    with pdfplumber.open(filepath) as pdf:
        logger.info("PDF has %d pages", len(pdf.pages))

        for i, page in enumerate(pdf.pages, start=1):
            text = ""
            tables = []

            # If not running in OMR mode, try to extract digital text first
            if not is_omr:
                text = extract_page_custom(page).strip()
                tables = _extract_tables(page)

            if not is_omr and len(text) >= MIN_TEXT_CHARS:
                pages.append(Page(
                    number=i,
                    text=text,
                    method=ExtractionMethod.TEXT,
                    word_count=len(text.split()),
                    tables=tables,
                ))
            else:
                logger.info("Page %d: falling back to image extraction (OMR=%s)", i, is_omr)
                
                # Render only the current scanned page as an image
                images = convert_from_path(filepath, dpi=dpi, first_page=i, last_page=i)
                if not images:
                    raise RuntimeError(f"Failed to render page {i} for image processing")

                # Convert PIL image to BGR numpy array
                cv_img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
                
                # Align and correct perspective (for phone-scanned documents)
                cv_img = correct_perspective(cv_img)

                if is_omr:
                    omr_text = detect_omr_sheet(cv_img)
                    pages.append(Page(
                        number=i,
                        text=omr_text,
                        method=ExtractionMethod.OMR,
                        word_count=len(omr_text.split()),
                        tables=[],
                    ))
                else:
                    # Convert back to PIL for Tesseract OCR
                    aligned_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    ocr_text, confidence = _ocr_image(aligned_pil, ocr_lang)
                    pages.append(Page(
                        number=i,
                        text=ocr_text,
                        method=ExtractionMethod.OCR,
                        confidence=confidence,
                        word_count=len(ocr_text.split()),
                        tables=tables,
                    ))

    return pages


def _extract_image(filepath: str, ocr_lang: str, is_omr: bool) -> list[Page]:
    cv_img = cv2.imread(filepath)
    if cv_img is None:
        raise ValueError(f"Failed to read image: {filepath}")
        
    # Align and correct perspective (for phone-scanned documents)
    cv_img = correct_perspective(cv_img)
    
    if is_omr:
        omr_text = detect_omr_sheet(cv_img)
        return [Page(
            number=1,
            text=omr_text,
            method=ExtractionMethod.OMR,
            word_count=len(omr_text.split()),
        )]
    else:
        aligned_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        text, confidence = _ocr_image(aligned_pil, ocr_lang)
        return [Page(
            number=1,
            text=text,
            method=ExtractionMethod.OCR,
            confidence=confidence,
            word_count=len(text.split()),
        )]


def _ocr_image(image: Image.Image, lang: str) -> tuple[str, float]:
    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        image, lang=lang, config=config,
        output_type=pytesseract.Output.DICT,
    )
    
    scores = [int(c) for c in data["conf"] if int(c) > -1]
    avg_confidence = sum(scores) / len(scores) if scores else 0.0

    words = data["text"]
    line_nums = data["line_num"]
    block_nums = data["block_num"]

    lines = []
    current_line = []
    current_line_num = -1
    current_block_num = -1

    for i, word in enumerate(words):
        word_str = str(word).strip()
        if not word_str:
            continue

        if block_nums[i] != current_block_num or line_nums[i] != current_line_num:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
            if block_nums[i] != current_block_num and lines:
                lines.append("")
            current_line_num = line_nums[i]
            current_block_num = block_nums[i]

        current_line.append(word_str)

    if current_line:
        lines.append(" ".join(current_line))

    text = "\n".join(lines).strip()
    return text, round(avg_confidence, 2)


def _extract_tables(page) -> list[list[list[str]]]:
    try:
        raw = page.extract_tables()
        if not raw:
            return []
        return [
            [
                [str(cell).strip() if cell else "" for cell in row]
                for row in table
            ]
            for table in raw
        ]
    except Exception as e:
        logger.warning("Table extraction failed: %s", e)
        return []
