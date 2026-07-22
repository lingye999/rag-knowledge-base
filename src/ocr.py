"""EasyOCR 图片识别，用于扫描件 PDF"""
import numpy as np
import fitz
from .cleaner import clean_table_noise

_EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(['ch_sim', 'en'])
    return _EASYOCR_READER


def read_pdf_ocr_pages(path: str,
                       page_numbers: set[int] | None = None) -> list[str]:
    """EasyOCR 图片识别"""
    try:
        doc = fitz.open(path)
        reader = _get_easyocr_reader()
        pages = [""] * len(doc)
        for page_number, page in enumerate(doc, start=1):
            if page_numbers is not None and page_number not in page_numbers:
                continue
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            result = reader.readtext(img)
            line_texts = [text for _, text, conf in result if conf > 0.3]
            pages[page_number - 1] = "".join(line_texts)

        doc.close()
        return [clean_table_noise(page) for page in pages]
    except Exception:
        return []


def read_pdf_ocr(path: str) -> str:
    """Keep the legacy text-only OCR interface."""
    pages = read_pdf_ocr_pages(path)
    result = "\n".join(pages)
    return result if result.strip() else ""
