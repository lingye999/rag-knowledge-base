"""文档入库：读取 → 清洗 → 分块 → 质量评分 → 向量化 → 入库 + BM25 索引"""

import os
from .document import read_file
from .chunker import chunk_text
from .cleaner import clean_ocr_text
from .quality_scorer import compute_quality_scores


class IngestionService:
    """统一入库服务

    用法:
        svc = IngestionService(emb, db, retriever)
        svc.add("data/说明书.pdf")
        svc.add("data/扫描件.pdf", force_ocr=True)
    """

    def __init__(self, emb, db, retriever):
        self.emb = emb
        self.db = db
        self.retriever = retriever

    def add(self, file_path: str, chunk_method: str = "auto",
            force_ocr: bool = False, use_marker: bool = False):
        """添加入库一个文档

        Args:
            use_marker: 复杂 PDF 使用 Marker 解析（需 GPU + 本地模型）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        text = read_file(file_path, force_ocr=force_ocr, use_marker=use_marker)
        text = clean_ocr_text(text)
        chunks = chunk_text(text, chunk_method)
        if not chunks:
            print(f"[警告] 文件 {file_path} 未提取到有效文本块，已跳过")
            return 0, ""

        vectors = self.emb.encode_batch(chunks)
        if not vectors:
            print(f"[警告] 文件 {file_path} 向量化失败，已跳过")
            return 0, ""

        doc_name = os.path.basename(file_path)
        qualities = compute_quality_scores(chunks)  # ← 算质量分
        self.db.add_batch(chunks, vectors, doc_name=doc_name, qualities=qualities)
        self.retriever.add_texts(chunks)  # ← 同时构建 BM25 索引

        mode = "Marker" if use_marker else ("OCR模式" if force_ocr else "默认(混合)")
        print(f"文件 {file_path} 加载完成({mode}): {len(chunks)} 个文本块")
        return len(chunks), doc_name
