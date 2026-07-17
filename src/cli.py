"""CLI 命令解析与主循环"""
import time
import os
import jieba
import numpy as np
import logging
from .embedding import EmbeddingService
from .vector_store.faiss_store import FaissVectorStore
from .vector_store.ivf_store import IvfVectorStore
from .vector_store.hnsw_store import HnswVectorStore
from .vector_store.hybrid import HybridRetriever
from .llm_service import LLMService
from .retriever import Retriever
from .ingestion import IngestionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

INDEX_TYPES = {
    "flat": FaissVectorStore,
    "ivf": IvfVectorStore,
    "hnsw": HnswVectorStore,
}


def run():
    emb = EmbeddingService()
    current_type = "flat"
    if current_type == "ivf":
        db = IvfVectorStore(emb.dimension)
    elif current_type == "hnsw":
        db = HnswVectorStore(emb.dimension)
    else:
        db = FaissVectorStore(emb.dimension)
    log.info(f"向量搜索系统已启动（当前索引: {current_type}，/help 查看帮助）")
    hybrid = HybridRetriever(db)

    # LLM 初始化（容错：没 API Key 也能启动）
    try:
        llm = LLMService(
            api_key=os.environ.get("DEEPSEEK_API_KEY", "") or "sk-c2419e869b7f4123a1fd0c69fcabc9c0",
            model="deepseek-v4-flash"
        )
        log.info("LLM 已就绪（DeepSeek V4 Flash）")
    except ValueError as e:
        llm = None
        log.warning(f"LLM 未配置: {e}")

    # 入库 + 检索引擎
    ingestion = IngestionService(emb, db, hybrid)
    retriever = Retriever(db)

    def _rewrite(query: str) -> str:
        """自动改写查询：LLM 可用时改写，不可用时返回原始"""
        if llm is None:
            return query
        try:
            rewritten = llm.rewrite(query)
            if rewritten and not rewritten.startswith("[改写失败"):
                log.info(f"改写: {query} → {rewritten}")
                return rewritten
        except Exception:
            pass
        return query

    while True:
        cmd = input().strip()

        if cmd == "/exit":
            log.info("结束")
            break

        elif cmd == "/help":
            print("可用命令：")
            print("  /search <query>      - 搜索（显示耗时）")
            print("  /search_jieba <query> - 先分词再搜索")
            print("  /hybrid_search <query> - 混合检索（Dense+BM25）")
            print("  /ask <query>         - LLM 回答（检索+AI生成）")
            print("  /rewrite <query>     - LLM 改写查询为关键词")
            print("  /add <file>          - 从文件添加")
            print("  /add <file> ocr      - 强制 OCR 模式添加")
            print("  /count               - 查看总数")
            print("  /switch <type>       - 切换索引: flat / ivf / hnsw")
            print("  /delete <doc_name>   - 标记删除文档（/save 后永久生效）")
            print("  /save <path>         - 保存")
            print("  /load <path>         - 加载")
            print("  /exit                - 退出")

        elif cmd == "/count":
            log.info(f"数据库的条数是{db.count}")

        elif cmd.startswith("/search "):
            rest = cmd[len("/search "):].strip()
            parts = rest.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                query = parts[0]
                top_k = int(parts[1])
            else:
                query = rest
                top_k = 5

            query = _rewrite(query)
            doc_filter = None
            if llm is not None:
                sq, filters = llm.self_query(query)
                if filters.get("doc"):
                    doc_filter = filters["doc"]
                    log.info(f"限定文档: {doc_filter}")
                query = sq

            t0 = time.time()
            vec = emb.encode(query)
            results = retriever.search(vec, top_k=top_k, doc_filter=doc_filter)
            t1 = time.time()
            print(f"搜索耗时: {(t1-t0)*1000:.1f}ms")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] [{r['doc']}] {r['text']}")

        elif cmd.startswith("/search_jieba "):
            rest = cmd[len("/search_jieba "):].strip()
            parts = rest.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                query = parts[0]
                top_k = int(parts[1])
            else:
                query = rest
                top_k = 5

            query = _rewrite(query)
            words = jieba.lcut(query)
            segmented = " ".join(words)
            print(f"分词结果: {segmented}")
            doc_filter = None
            if llm is not None:
                sq, filters = llm.self_query(query)
                if filters.get("doc"):
                    doc_filter = filters["doc"]
                query = sq

            t0 = time.time()
            vec = emb.encode(segmented)
            results = retriever.search(vec, top_k=top_k, doc_filter=doc_filter)
            t1 = time.time()
            print(f"搜索耗时: {(t1-t0)*1000:.1f}ms")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] [{r['doc']}] {r['text']}")

        elif cmd.startswith("/hybrid_search "):
            rest = cmd[len("/hybrid_search "):].strip()
            parts = rest.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                query = parts[0]
                top_k = int(parts[1])
            else:
                query = rest
                top_k = 5

            query = _rewrite(query)
            doc_filter = None
            if llm is not None:
                sq, filters = llm.self_query(query)
                if filters.get("doc"):
                    doc_filter = filters["doc"]
                query = sq

            t0 = time.time()
            tokens = jieba.lcut(query)
            vec = emb.encode(query)
            raw = hybrid.search(vec, tokens, top_k=top_k * 5)
            if doc_filter:
                raw = [r for r in raw if db.meta[r["index"]].get("doc") == doc_filter]
            if not raw:
                print("无结果")
                continue
            raw = sorted(raw, key=lambda x: x["score"], reverse=True)[:top_k]
            t1 = time.time()
            print(f"混合检索耗时: {(t1-t0)*1000:.1f}ms")
            for i, r in enumerate(raw):
                doc = db.meta[r["index"]].get("doc", "未知")
                print(f"{i+1}. [hybrid {r['score']:.4f}] [{doc}] {r['text']}")

        elif cmd.startswith("/ask "):
            if llm is None:
                print("LLM 未配置。请检查 API Key 是否正确")
                continue
            query = cmd[len("/ask "):].strip()
            search_query = _rewrite(query)

            doc_filter = None
            if llm is not None:
                sq, filters = llm.self_query(search_query)
                if filters.get("doc"):
                    doc_filter = filters["doc"]
                search_query = sq

            t0 = time.time()
            vec = emb.encode(search_query)
            results = retriever.search(vec, top_k=8, doc_filter=doc_filter)
            chunks = [r["text"] for r in results]
            t1 = time.time()
            log.info(f"检索到 {len(chunks)} 条，耗时 {(t1-t0)*1000:.1f}ms")

            if chunks:
                log.info("LLM 生成回答中...")
                answer = llm.ask(query, chunks)
                print(f"\n{answer}")
            else:
                print("未检索到相关内容。")

        elif cmd.startswith("/rewrite "):
            if llm is None:
                print("LLM 未配置。请检查 API Key 是否正确")
                continue
            query = cmd[len("/rewrite "):].strip()
            log.info("LLM 改写中...")
            rewritten = llm.rewrite(query)
            print(f"改写前: {query}")
            print(f"改写后: {rewritten}")

        elif cmd.startswith("/switch "):
            idx_type = cmd[len("/switch "):].strip()

            if idx_type not in INDEX_TYPES:
                log.warning(f"不支持的索引类型: {idx_type}，可选: flat / ivf / hnsw")
            else:
                n = db.count
                old_texts = db.texts[:]
                old_vecs = np.zeros((n, db.dimension), dtype=np.float32)
                db.index.reconstruct_n(0, n, old_vecs)

                if idx_type == "ivf":
                    db = IvfVectorStore(emb.dimension)
                elif idx_type == "hnsw":
                    db = HnswVectorStore(emb.dimension)
                else:
                    db = FaissVectorStore(emb.dimension)

                if old_texts:
                    db.add_batch(old_texts, old_vecs.tolist())
                    log.info(f"已切换到 {idx_type} 索引（已保留 {len(old_texts)} 条数据）")
                else:
                    log.info(f"已切换到 {idx_type} 索引（当前为空）")

                # 更新检索引擎指向新索引
                retriever.db = db
                hybrid.db = db
                ingestion.db = db
                ingestion.hybrid = hybrid

                current_type = idx_type

        elif cmd.startswith("/delete "):
            doc_name = cmd[len("/delete "):].strip()
            db.delete_doc(doc_name)
            log.info(f"文档 {doc_name} 已标记删除（/save 后永久生效）")

        elif cmd.startswith("/add "):
            try:
                rest = cmd[len("/add "):].strip()
                parts = rest.split(" ")
                path = parts[0]
                if not path:
                    print("用法: /add <文件路径> [ocr]")
                    continue
                method = parts[1] if len(parts) > 1 and parts[1] != "ocr" else "auto"
                force_ocr = "ocr" in parts
                n, fname = ingestion.add(path, chunk_method=method, force_ocr=force_ocr)
                if n:
                    log.info(f"文件 {fname} 入库完成: {n} 个文本块")
            except Exception as e:
                log.error(f"添加失败: {e}")

        elif cmd.startswith("/save "):
            try:
                path = cmd[len("/save "):].strip()
                db._compact()
                db.save(path)
                log.info(f"已保存到 {path}")
            except Exception as e:
                log.error(f"保存失败: {e}")

        elif cmd.startswith("/load "):
            try:
                path = cmd[len("/load "):].strip()
                db.load(path)
                log.info(f"已从 {path} 加载")
            except Exception as e:
                log.error(f"加载失败: {e}")

        else:
            print("未知命令，输入 /help 查看帮助")
