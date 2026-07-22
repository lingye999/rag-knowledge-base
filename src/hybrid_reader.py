"""轻量混合 PDF 读取：逐块判断乱码，选择性用 OCR 替换

原理（参考 RAGFlow DeepDoc）：
  1. pdfplumber 提取文本行（带坐标位置 + 字体信息）
  2. 对每行检测乱码：
     a. 显式乱码：□、�、▯ 等替换字符
     b. 隐式乱码：子集字体 CID 映射错位 → 产生「合法但错误」的汉字
     c. 字体检测：子集字体（fontname 含 '+'）/ CID 字体直接判定
  3. 正常行 → 直接用 pdfplumber 的文本（快且准）
  4. 乱码行 → 从页面图片裁剪该区域 → EasyOCR
"""
import pdfplumber
import fitz
import numpy as np

_EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    return _EASYOCR_READER


# ── 常用汉字集（用于检测 CID 映射错位）──
# GB2312 一级汉字（最常用的 3755 个汉字的大致范围）
_COMMON_CJK = set(
    "的一是不了人我在有他这来之小中上大个也到为时说学国"
    "和会可以要没地着看过天去生子得自么对出家年下后好"
    "开那能然心起都于还成事作当所如法用道它行间想见长"
    "经他同工知公把日新方正入其从两关她些面什样因什等"
    "内员军者向手美高点己业体里学农名其定外多水高间合"
    "去生把等从很会家可主通开让然人又部种月力重但女太"
    "前由回实分文已给次明任无文或只全再利它又目第此石"
    "少与立必每北科入建强关平各代期较十基数完今南或社"
    "至却口决及品电度变门革九性说机东路图知决比应总长"
    "叫报务很运机条接根海安西任北收规集装历质引段制你"
    "示头带程车层证习组县石需且候半青议完际金列交导温"
    "装山统论保等风级达布改八热声除际精算复究称效走斯"
    "标查世该深许光研林低万术千写备级维计识除未府束火"
    "器格增斗状步毛反具准片治极造科火众铁均施族向段类"
    "规族准府包维置土候型厂程站片")  # 只列了部分示意，实际用范围判断


def _is_rare_cjk(char: str) -> bool:
    """判断一个 CJK 字符是否属于罕见字

    常见汉字集中在 U+4E00–U+9FFF，但同样在这个区域内的字也有常用/罕用之别。
    这里用 Unicode 区块 + 简单启发式判断：
      - 扩展区汉字（U+3400–U+4DBF, U+20000–U+2A6DF）→ 罕见
      - 基本区但不在常见 5000 字范围内 → 可能异常
    """
    code = ord(char)
    # 扩展 A 区（罕见）
    if 0x3400 <= code <= 0x4DBF:
        return True
    # 扩展 B 区及以上（极罕见）
    if 0x20000 <= code <= 0x2FFFF:
        return True
    # 基本区的中日韩兼容表意文字（U+FA0E–U+FA2F 等罕用区）
    if 0xF900 <= code <= 0xFAFF:
        return True
    # 基本区中的生僻字判断：U+4E00–U+9FFF 但超过常用范围
    # 常用汉字集中在 U+4E00–U+9FCC 的前半段
    # 粗略：大于 U+9A00 的汉字在日常文本中很少见
    if 0x9FC0 <= code <= 0x9FFF:
        return True
    return False


# ── 乱码检测 ──

def _is_garbled(text: str) -> bool:
    """判断文本块是否乱码

    检测维度（四级逐层筛查）：
      L1 - 显式乱码：□、�、▯ 等替换字符
      L2 - 连续重复检测：OCR / CID 错位的典型特征（汽汽车车、宇宇航航）
      L3 - 罕用汉字检测：CID 映射错位产生的非正常汉字
      L4 - 统计异常：控制字符、标点占比过高
    """
    if not text or not text.strip():
        return True

    # L1: 替换字符检测
    garbled_chars = sum(1 for c in text if c in '□�▯')
    if garbled_chars / max(len(text), 1) > 0.02:
        return True

    # L2: 连续重复字符检测（OCR / CID 错误的典型特征）
    # 如「汽汽车车」「宇宇航航」「的的的的」
    repeat_chars = 0
    i = 0
    while i < len(text) - 1:
        if text[i] == text[i + 1]:
            repeat_chars += 2
            i += 2
        else:
            i += 1
    if repeat_chars > 0:
        ratio = repeat_chars / max(len(text), 1)
        if ratio >= 0.25:
            return True  # 超过 1/4 的字符是重复的 → 垃圾

    # L3: 罕用汉字检测（CID 映射错位的另一表现）
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    rare_count = sum(1 for c in text if _is_rare_cjk(c))
    if cjk_count > 5 and rare_count / max(cjk_count, 1) > 0.3:
        return True

    # L4a: 不可见控制字符
    unreadable = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    if unreadable / max(len(text), 1) > 0.05:
        return True

    # L4b: 标点占比过高
    punct = sum(1 for c in text if c in '.,;:!?()[]{}<>/\\|`~@#$%^&*-=+')
    if len(text) > 5 and punct / len(text) > 0.4:
        return True

    return False


# ── 字体检测 ──

def _is_subset_font(fontname: str) -> bool:
    """判断字体名是否表示子集字体（subset font）

    子集字体命名特征：
      - 'ABCDEF+SimSun'（含 '+'）
      - 'AAAAAA+TimesNewRoman'
    这些字体的 CID 映射经常损坏。
    """
    return '+' in fontname


def _page_has_subset_fonts(page) -> bool:
    """快速扫描整页：是否有任何子集字体

    比逐块检测更高效——如果整页都用子集字体，直接全部 OCR 即可。
    """
    try:
        for char in page.chars[:200]:  # 前 200 个字符足够判断
            if _is_subset_font(char.get('fontname', '')):
                return True
    except Exception:
        pass
    return False


def _block_has_suspicious_font(page, bbox) -> bool:
    """检测指定区域的字体是否可疑（子集字体 或 CID 编码）
    
    即使字体名不包含 '+', CID 字体（Type 0）也可能有映射问题。
    """
    try:
        chars = page.within_bbox(bbox).chars
        for char in chars[:50]:
            fn = char.get('fontname', '')
            if _is_subset_font(fn):
                return True
            # 有些 PDF 的 CID 字体名不含 '+', 但字体编码是 Identity-H
            encoding = char.get('fontencoding', '')
            if 'Identity' in encoding:
                return True
    except Exception:
        pass
    return False


# ── 文本行分组 ──

def _group_into_lines(words: list[dict]) -> list[dict]:
    """将 pdfplumber 的 words 列表按行分组

    返回: [{'text': '...', 'bbox': (x0, top, x1, bottom)}, ...]
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

    lines = []
    cur_line = [sorted_words[0]]
    cur_bottom = sorted_words[0]['bottom']

    for w in sorted_words[1:]:
        if w['top'] < cur_bottom + 2:
            cur_line.append(w)
            cur_bottom = max(cur_bottom, w['bottom'])
        else:
            lines.append(_line_to_block(cur_line))
            cur_line = [w]
            cur_bottom = w['bottom']

    if cur_line:
        lines.append(_line_to_block(cur_line))

    return lines


def _line_to_block(line_words: list[dict]) -> dict:
    """将一行的 words 合并为一个 block"""
    x0 = min(w['x0'] for w in line_words)
    x1 = max(w['x1'] for w in line_words)
    top = min(w['top'] for w in line_words)
    bottom = max(w['bottom'] for w in line_words)
    text = "".join(w['text'] for w in line_words)
    return {'text': text, 'bbox': (x0, top, x1, bottom)}


# ── 主入口 ──

def read_pdf_hybrid_pages(path: str, dpi: int = 200,
                          page_numbers: set[int] | None = None,
                          allow_ocr: bool = True) -> list[str]:
    """轻量混合读取 PDF

    流程（逐页）：
      0. 快速检测整页字体类型
      1. pdfplumber 提取本页所有单词及坐标
      2. 分组为文本行
      3. 对每行检测乱码：
         - Level 1: 显式替换字符（□�▯）
         - Level 2: 罕用汉字比例（CID 映射错位）
         - Level 3: 统计异常 + 字体检测
      4. 正常 → 直接取用；可疑 → 裁剪图片做 OCR
      5. 整页无文字 → 全页 OCR 兜底

    Args:
        path: PDF 文件路径
        dpi: 渲染分辨率，越高 OCR 越准但越慢
    """
    reader = None

    doc_fitz = fitz.open(path)

    with pdfplumber.open(path) as pdf:
        total_pages = min(len(doc_fitz), len(pdf.pages))
        all_pages = [""] * total_pages
        for page_idx in range(total_pages):
            if page_numbers is not None and page_idx + 1 not in page_numbers:
                continue
            page_plumber = pdf.pages[page_idx]
            page_fitz = doc_fitz[page_idx]

            # ── 渲染页面图片 ──
            pix = page_fitz.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            scale = dpi / 72

            # ── 提取单词 ──
            words = page_plumber.extract_words()

            # 情况 A：整页无文字 → 全页 OCR
            if not words:
                if not allow_ocr:
                    continue
                if reader is None:
                    reader = _get_easyocr_reader()
                ocr_result = reader.readtext(img)
                line_texts = [t for _, t, c in ocr_result if c > 0.3]
                all_pages[page_idx] = "".join(line_texts)
                continue

            # 情况 B：有文字 → 逐行判断
            lines = _group_into_lines(words)
            page_lines = []

            for line in lines:
                text = line['text']
                bbox = line['bbox']

                # ── 决定是否 OCR ──
                # 原则：只对确实有乱码问题（_is_garbled）的行做 OCR
                # 字体检测（_block_has_suspicious_font）仅作为辅助信息，
                # 不作为触发 OCR 的独立条件——
                # 测试表明：正常的行送到 OCR 反而会引入新错误（变→娈）
                need_ocr = _is_garbled(text)

                if need_ocr:
                    if not allow_ocr:
                        page_lines.append(text)
                        continue
                    # 坐标转换：points → pixels
                    x0 = int(max(0, bbox[0] * scale - 3))
                    y0 = int(max(0, bbox[1] * scale - 3))
                    x1 = int(min(pix.width, bbox[2] * scale + 3))
                    y1 = int(min(pix.height, bbox[3] * scale + 3))

                    if x1 > x0 and y1 > y0:
                        if reader is None:
                            reader = _get_easyocr_reader()
                        line_img = img[y0:y1, x0:x1]
                        ocr_result = reader.readtext(line_img)
                        ocr_text = "".join(
                            t for _, t, c in ocr_result if c > 0.3
                        )
                        page_lines.append(ocr_text if ocr_text else text)
                    else:
                        page_lines.append(text)
                else:
                    page_lines.append(text)

            all_pages[page_idx] = "".join(page_lines)

            # 进度提示（长文档）
            if (page_idx + 1) % 10 == 0:
                print(f"  [Hybrid] 第 {page_idx + 1}/{len(pdf.pages)} 页处理完成")

    doc_fitz.close()
    total_chars = sum(len(p) for p in all_pages)
    print(f"[Hybrid] {path} 解析完成 ({len(pdf.pages)} 页, {total_chars} 字符)")
    return all_pages


def read_pdf_hybrid(path: str, dpi: int = 200) -> str:
    """Keep the legacy text-only hybrid-reader interface."""
    return "\n".join(read_pdf_hybrid_pages(path, dpi=dpi))
