"""文档入库：读取 → 清洗 → 分块 → 质量评分 → 向量化 → 入库 + BM25 索引"""

import os
from dataclasses import replace
from config import config as _cfg
from ..parsing.cleaner import clean_ocr_text
from ..parsing.document import read_file_structured
from ..parsing.quality_gate import assess_parse_quality, needs_ocr_fallback
from ..retrieval.chunker import chunk_blocks
from ..retrieval.quality_scorer import compute_quality_scores


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
        self.repository = getattr(retriever, "repository", None)

    def bind_store(self, db, repository=None):
        """Bind a replacement index after an index lifecycle operation."""
        self.db = db
        self.repository = repository or getattr(self.retriever, "repository", None)

    def add(self, file_path: str, chunk_method: str = "auto",
            force_ocr: bool = False, use_marker: bool = False,
            allow_ocr: bool = True):
        """添加入库一个文档

        Args:
            use_marker: 复杂 PDF 使用 Marker 解析（需 GPU + 本地模型）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        parsed = read_file_structured(
            file_path,
            force_ocr=force_ocr,
            use_marker=use_marker,
            allow_ocr=allow_ocr,
        )
        quality_settings = _cfg.get("ingestion", {}).get("parse_quality", {})
        initial_quality = assess_parse_quality(parsed.blocks)
        if (
            file_path.lower().endswith(".pdf")
            and not force_ocr
            and allow_ocr
            and needs_ocr_fallback(initial_quality, quality_settings)
        ):
            print(
                "[解析质量] 混合解析质量不足，尝试全页 OCR："
                f"字符={initial_quality.total_chars}，"
                f"可读率={initial_quality.readable_ratio:.0%}，"
                f"乱码率={initial_quality.garbled_ratio:.0%}"
            )
            ocr_parsed = read_file_structured(
                file_path,
                force_ocr=True,
                use_marker=False,
                allow_ocr=True,
            )
            ocr_quality = assess_parse_quality(ocr_parsed.blocks)
            if ocr_quality.score > initial_quality.score:
                parsed = ocr_parsed
                print("[解析质量] 已采用 OCR 结果")
            else:
                print("[解析质量] 保留原混合解析结果")
        doc_name = os.path.basename(file_path)
        cleaned_blocks = [
            replace(block, text=clean_ocr_text(block.text))
            for block in parsed.blocks
            if block.text.strip()
        ]
        page_context = _cfg["chunking"].get("page_context", {})
        chunks = chunk_blocks(
            cleaned_blocks,
            chunk_method,
            doc_name,
            include_page_context=bool(page_context.get("enabled", False)),
            page_context_max_chars=int(page_context.get("max_chars", 900)),
        )
        if not chunks:
            print(f"[警告] 文件 {file_path} 未提取到有效文本块，已跳过")
            return 0, ""

        texts = [chunk.text for chunk in chunks]
        vectors = self.emb.encode_batch(texts)
        if not vectors:
            print(f"[警告] 文件 {file_path} 向量化失败，已跳过")
            return 0, ""

        qualities = compute_quality_scores(texts)  # ← 算质量分
        chunks = [
            replace(chunk, quality=quality)
            for chunk, quality in zip(chunks, qualities)
        ]
        if self.repository is not None:
            self.repository.add_batch(chunks, vectors)
        else:
            self.db.add_batch(
                texts,
                vectors,
                doc_name=doc_name,
                qualities=qualities,
                pages=[chunk.page for chunk in chunks],
                sources=[chunk.source for chunk in chunks],
            )
        self.retriever.add_texts(texts)  # ← 同时构建 BM25 索引

        mode = "Marker" if use_marker else ("OCR模式" if force_ocr else "默认(混合)")
        print(f"文件 {file_path} 加载完成({mode}): {len(chunks)} 个文本块")
        return len(chunks), doc_name
