"""文本清洗：表格噪声过滤 + 乱码页检测 + 停用词 + OCR 空白清理"""

import re


def clean_ocr_text(text: str) -> str:
    """清洗 OCR/Marker 输出中的残留空白和 HTML 标签"""
    # 去掉 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 多个空格合并为一个
    text = re.sub(r' {3,}', ' ', text)
    # 多个换行合并为一个
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 行首行尾去空白
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def clean_table_noise(text: str) -> str:
    """清洗 pdfplumber 表格提取产生的脏数据"""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if all(c in "√/| " for c in s):
            continue
        s = s.replace(" | ", " ").replace("/ ", " ")
        s = s.replace(" |", "").replace("| ", "")
        s = s.strip("| ")
        if len(s) <= 3 and not re.search(r'[\u4e00-\u9fff]', s):
            continue
        no_alpha = sum(1 for c in s if c in "0123456789/\\-CDVYEABPJ ")
        if len(s) > 3 and no_alpha / len(s) > 0.7:
            continue
        if s:
            cleaned.append(s)
    return "\n".join(cleaned)


def _is_garbage_page(text: str) -> bool:
    """判断一页提取的文本是否是垃圾（技术图纸标注），是则跳过"""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    if len(stripped) < 20:
        return False

    chinese = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
    ch_ratio = chinese / len(stripped)
    if ch_ratio < 0.15:
        return True

    words = stripped.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2.5:
            return True

    return False


def filter_stopwords(words: list[str]) -> list[str]:
    """去掉常见的停用词（的、了、是、在...）"""
    stopwords = {"的", "了", "是", "在", "和", "就", "都", "而", "及", "与",
                 "着", "或", "一个", "没有", "我们", "你们", "他们", "它",
                 "她", "他", "有", "不", "被", "把", "这", "那", "也"}
    return [w for w in words if w not in stopwords and w.strip()]
