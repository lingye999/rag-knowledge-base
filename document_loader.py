import re #这里的这个re是python本省自带的某种规则表达式
import jieba
import docx
import fitz

def read_file(path:str)->str:
    """根据文件后缀自动选择读取方式，支持 txt/docx/pdf"""
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
             return f.read()
    elif path.endswith(".docx"):
         return read_docx(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    else:
        raise ValueError(f"不支持的文件格式: {path}，仅支持 .txt / .docx / .pdf")


def read_docx(path:str)->str:
    """读取word文本"""
    doc=docx.Document(path)#读取文档
    paragraph=[]
    for para in doc.paragraphs:
        if para.text.strip():
            paragraph.append(para.text)
    return "\n".join(paragraph)

def read_pdf(path:str)->str:
    """读取pdf文本"""
    with fitz.open(path) as pdf:
       pages=[]
       for page in pdf:
           text=page.get_text()
           if text.strip():
              pages.append(text)
    return "\n".join(pages)


def chunk_by_sentence(text:str)->list[str]:
    """按照【。！？】分割句子"""

    sentences=re.split(r"[。！？]",text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_by_paragraph(text:str)->list[str]:
    """按照段落去分割这个文本"""


    paragraph=text.split("\n")
    return [s.strip()for s in paragraph if s.strip()]

def chunk_by_size(text: str,chunk_size:int=200,overlap:int =50)->list[str]:

    chunks=[]#存储这个截断的语句
    start=0

    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start+=chunk_size-overlap#减去重叠的部分
    return chunks

def chunk_by_jieba(text:str,max_words:int=50)->list[str]:
    """按中文结果分词分块"""
    words=jieba.lcut(text)  #分词
    chunks=[]
    for i in range(0,len(words),max_words):
        chunk="".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks


def chunk_text(text: str, method: str = "auto") -> list[str]:
    """统一分块调度，支持 auto 自动选择"""
    if method == "auto":
        # 1. 数段落
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        # 2. 算中文占比
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

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
