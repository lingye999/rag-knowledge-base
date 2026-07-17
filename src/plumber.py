"""pdfplumber PDF 文本提取"""
import pdfplumber
from .cleaner import clean_table_noise, _is_garbage_page


def read_pdf_plumber(path: str) -> str:
    """pdfplumber 文本提取"""
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            empty_pages = 0
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text()
                page_text = text.strip() if text else ""
                is_garbage = _is_garbage_page(page_text)

                if not is_garbage:
                    if page_text:
                        pages.append(page_text)
                else:
                    empty_pages += 1

                if not is_garbage:
                    tables = page.extract_tables()
                    for table in tables:
                        rows = []
                        for row in table:
                            cells = [c.strip() if c else "" for c in row]
                            line = " | ".join(cells)
                            noise_ratio = sum(1 for c in line if c in "√/|0123456789 ") / len(line) if line else 1
                            if noise_ratio > 0.6:
                                continue
                            rows.append(line)
                        if rows:
                            pages.append("\n".join(rows))

            result = "\n".join(pages)
            result = clean_table_noise(result)
            empty_ratio = empty_pages / total_pages if total_pages > 0 else 1
            if result.strip() and empty_ratio < 0.5:
                return result
    except Exception:
        pass
    return ""
