import re
import os
import jieba
import docx
import fitz


def read_file(path: str) -> str:
    """根据文件后缀自动选择读取方式，支持 txt/docx/pdf"""
    # 检查文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.endswith(".txt"):
        # 自动检测编码：先试 UTF-8，失败则试常见中文编码
        for encoding in ["utf-8", "gbk", "gb2312", "gb18030"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # 所有编码都失败，最后用 utf-8 抛原始异常
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    elif path.endswith(".docx"):
        return read_docx(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    else:
        raise ValueError(f"不支持的文件格式: {path}，仅支持 .txt / .docx / .pdf")


def read_docx(path: str) -> str:
    """读取word文本，包括段落和表格"""
    try:
        doc = docx.Document(path)
    except Exception as e:
        raise RuntimeError(f"无法读取 DOCX 文件（可能已损坏、加密或格式为旧版 .doc）: {path}") from e

    paragraph = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraph.append(para.text)

    # 同时提取表格中的文字
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraph.append(cell.text)

    result = "\n".join(paragraph)
    if not result.strip():
        print(f"[警告] DOCX 文件中未提取到文字（可能全是图片）: {path}")
    return result


def read_pdf(path: str) -> str:
    """读取pdf文本"""
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"无法打开 PDF 文件（可能已加密、损坏或不是有效 PDF）: {path}") from e

    # 检查 PDF 是否需要密码（加密检测）
    if doc.is_encrypted:
        doc.close()
        raise RuntimeError(f"PDF 文件已加密，无法读取（需要密码）: {path}")

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        print(f"[警告] PDF 文件无页面: {path}")
        return ""

    # 大文件警告
    if page_count > 500:
        print(f"[警告] PDF 页数较多 ({page_count} 页)，可能需要较长时间...")

    pages = []
    empty_pages = 0
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
        else:
            empty_pages += 1

    doc.close()

    result = "\n".join(pages)

    # 如果所有页面都是空的（扫描件/纯图片 PDF）
    if not result.strip():
        print(f"[警告] PDF 全部 {page_count} 页均未提取到文字（可能是扫描件/纯图片 PDF，需要 OCR）: {path}")
    elif empty_pages > 0:
        print(f"[提示] PDF 共 {page_count} 页，其中 {empty_pages} 页无文字")

    return result


def chunk_by_sentence(text: str) -> list[str]:
    """按照【。！？.!?】分割句子，支持中英文"""

    sentences = re.split(r"[。！？.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_paragraph(text: str) -> list[str]:
    """按照段落去分割这个文本"""

    paragraph = text.split("\n")
    return [s.strip() for s in paragraph if s.strip()]


def chunk_by_size(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:

    chunks = []  # 存储这个截断的语句
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # 减去重叠的部分
    return chunks


def chunk_by_jieba(text: str, max_words: int = 50) -> list[str]:
    """按中文结果分词分块"""
    words = jieba.lcut(text)  # 分词
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = "".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def chunk_text(text: str, method: str = "auto") -> list[str]:
    """统一分块调度，支持 auto 自动选择"""
    # 空文本直接返回空列表
    if not text or not text.strip():
        print("[警告] 文本为空，无内容可分块")
        return []

    if method == "auto":
        # 1. 数段落
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        # 2. 算中文占比
        total_chars = len(text.strip())
        if total_chars > 0:
            chinese_chars = len(re.findall(r'[一-鿿]', text))
            chinese_ratio = chinese_chars / total_chars
        else:
            chinese_ratio = 0

        # 3. 自动选择
        if chinese_ratio > 0.3:
            method = "jieba"
        elif len(paragraphs) >= 3:
            method = "paragraph"
        else:
            method = "sentence"

        print(f"[Auto] 中文占比 {chinese_ratio:.0%}，段落 {len(paragraphs)} 个，使用 {method} 分块")

    # 4. 调度
    if method == "sentence":
        return chunk_by_sentence(text)
    elif method == "paragraph":
        return chunk_by_paragraph(text)
    elif method == "jieba":
        return chunk_by_jieba(text)
    elif method == "size":
        return chunk_by_size(text)
    else:
        raise ValueError(f"不支持的分块方法: {method}，可选: sentence / paragraph / jieba / size / auto")


def filter_stopwords(words: list[str]) -> list[str]:
    """去掉常见的停用词（的、了、是、在...）"""
    stopwords = {"的", "了", "是", "在", "和", "就", "都", "而", "及", "与",
                 "着", "或", "一个", "没有", "我们", "你们", "他们", "它",
                 "她", "他", "有", "不", "被", "把", "这", "那", "也"}
    return [w for w in words if w not in stopwords and w.strip()]
