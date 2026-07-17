"""EasyOCR 图片识别，用于扫描件 PDF"""
import numpy as np
import fitz
from .cleaner import clean_table_noise


def read_pdf_ocr(path: str) -> str:
    """EasyOCR 图片识别"""
    try:
        import easyocr

        doc = fitz.open(path)
        reader = easyocr.Reader(['ch_sim', 'en'])
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            result = reader.readtext(img)
            line_texts = [text for _, text, conf in result if conf > 0.3]
            pages.append("".join(line_texts))

        doc.close()
        result = "\n".join(pages)
        result = clean_table_noise(result)
        return result if result.strip() else ""
    except Exception:
        return ""
