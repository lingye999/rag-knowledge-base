"""文档入库：读取 → 清洗 → 分块 → 向量化 → 入库"""

import os
from .document import read_file
from .chunker import chunk_text
from .cleaner import clean_ocr_text


class IngestionService:
    """统一入库服务

    用法:
        svc = IngestionService(emb, db, hybrid)
        svc.add("data/说明书.pdf")                     # 默认模式
        svc.add("data/扫描件.pdf", force_ocr=True)     # OCR 模式
    """

    def __init__(self, emb, db, hybrid):
        self.emb = emb
        self.db = db
        self.hybrid = hybrid

    def add(self, file_path: str, chunk_method: str = "auto",
            force_ocr: bool = False):
        """添加入库一个文档

        返回: (chunk_count, doc_name)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        text = read_file(file_path, force_ocr=force_ocr)
        text = clean_ocr_text(text)  # 清洗 OCR 残留空白和 HTML 标签
        chunks = chunk_text(text, chunk_method)
        if not chunks:
            print(f"[警告] 文件 {file_path} 未提取到有效文本块，已跳过")
            return 0, ""

        vectors = self.emb.encode_batch(chunks)
        if not vectors:
            print(f"[警告] 文件 {file_path} 向量化失败，已跳过")
            return 0, ""

        doc_name = os.path.basename(file_path)
        self.db.add_batch(chunks, vectors, doc_name=doc_name)
        self.hybrid.add_texts(chunks)

        mode = "OCR模式" if force_ocr else "默认"
        print(f"文件 {file_path} 加载完成({mode}): {len(chunks)} 个文本块")
        return len(chunks), doc_name
