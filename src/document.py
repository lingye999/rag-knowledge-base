"""文档读取：txt / docx / pdf 多格式支持"""
import os
import docx
from .plumber import read_pdf_plumber
from .ocr import read_pdf_ocr
from .marker_reader import read_pdf_marker


def read_file(path: str, force_ocr: bool = False) -> str:
    """根据文件后缀自动选择读取方式，支持 txt/docx/pdf"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.endswith(".txt"):
        for encoding in ["utf-8", "gbk", "gb2312", "gb18030"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    elif path.endswith(".docx"):
        return _read_docx(path)
    elif path.endswith(".pdf"):
        return _read_pdf(path, force_ocr=force_ocr)
    else:
        raise ValueError(f"不支持的文件格式: {path}，仅支持 .txt / .docx / .pdf")


def _read_docx(path: str) -> str:
    """读取word文本，包括段落和表格"""
    try:
        doc = docx.Document(path)
    except Exception as e:
        raise RuntimeError(f"无法读取 DOCX 文件: {path}") from e

    paragraph = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraph.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraph.append(cell.text)

    result = "\n".join(paragraph)
    if not result.strip():
        print(f"[警告] DOCX 文件中未提取到文字: {path}")
    return result


def _read_pdf(path: str, force_ocr: bool = False) -> str:
    """读取 PDF 文本：Marker → pdfplumber → OCR 三级降级"""
    # ── 第 0 级：Marker（Day 4 新增，主力解析器） ──
    if not force_ocr:
        try:
            result = read_pdf_marker(path)
            if result and result.strip():
                return result
        except Exception:
            pass  # 降级

    # ── 第 1 级：OCR 优先模式 ──
    if force_ocr:
        result = read_pdf_ocr(path)
        if result and result.strip():
            return result
        result = read_pdf_plumber(path)
        if result and result.strip():
            return result
        raise RuntimeError(f"无法提取 PDF 文字（已尝试 OCR + 文本提取）: {path}")

    # ── 第 2 级：pdfplumber 降级 ──
    result = read_pdf_plumber(path)
    if result and result.strip():
        return result

    # ── 第 3 级：OCR 兜底 ──
    result = read_pdf_ocr(path)
    if result and result.strip():
        return result

    raise RuntimeError(f"无法提取 PDF 文字（已尝试 Marker → pdfplumber → OCR）: {path}")
