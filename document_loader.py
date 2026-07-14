import re #这里的这个re是python本省自带的某种规则表达式
import jieba
def read_file(path:str)->str:
    "读取txt的文本"
    with open(path, "r", encoding="utf-8") as f:
        return f.read();

def chunk_by_sentence(text:str)->list[str]:
    "按照【。！？】分割句子"

    sentences=re.split(r"[。！？]",text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_by_paragraph(text:str)->list[str]:
    "按照段落去分割这个文本"


    paragraph=text.split("\n\n")
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

def filter_stopwords(words: list[str]) -> list[str]:
    """去掉常见的停用词（的、了、是、在...）"""
    stopwords = {"的", "了", "是", "在", "和", "就", "都", "而", "及", "与",
                 "着", "或", "一个", "没有", "我们", "你们", "他们", "它",
                 "她", "他", "有", "不", "被", "把", "这", "那", "也"}
    return [w for w in words if w not in stopwords and w.strip()]
