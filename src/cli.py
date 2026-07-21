"""CLI 命令解析与主循环"""
import time
import os
import jieba
import numpy as np
import logging
import torch
from .embedding import EmbeddingService
from .vector_store.faiss_store import FaissVectorStore
from .vector_store.ivf_store import IvfVectorStore
from .vector_store.hnsw_store import HnswVectorStore
from .llm_service import LLMService
from .retriever import Retriever
from .ingestion import IngestionService
from .reranker import Reranker

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

    # 检索引擎 + 重排序
    try:
        reranker = Reranker("BAAI/bge-reranker-base",
                           device="cuda" if torch.cuda.is_available() else "cpu")
        log.info("重排序器已就绪（Cross-Encoder）")
    except Exception as e:
        reranker = None
        log.warning(f"重排序器加载失败，将跳过精排: {e}")
    retriever = Retriever(db, reranker=reranker)

    # 入库服务
    ingestion = IngestionService(emb, db, retriever)

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
            if db._save_path_prefix is not None:
                log.info("自动保存中...")
                try:
                    db.save(db._save_path_prefix)
                    log.info("已保存")
                except Exception as e:
                    log.error(f"自动保存失败: {e}")
            log.info("结束")
            break

        elif cmd == "/help":
            print("可用命令：")
            print("  /search <query>         - 搜索（Dense + BM25 混合）")
            print("  /search_jieba <query>   - 先分词再搜索")
            print("  /ask <query>            - LLM 回答（检索+AI生成）")
            print("  /rewrite <query>        - LLM 改写查询为关键词")
            print("  /add <file>             - 从文件添加")
            print("  /add <file> ocr         - 强制 OCR 模式添加")
            print("  /count                  - 查看总数")
            print("  /switch <type>          - 切换索引: flat / ivf / hnsw")
            print("  /delete <doc_name>      - 标记删除文档（/save 后永久生效）")
            print("  /list                  - 查看已有文档列表")
            print("  /clear                 - 清空所有数据（需确认）")
            print("  /save <path>            - 保存")
            print("  /load <path>            - 加载")
            print("  /exit                   - 退出")

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
            results = retriever.search(query, vec, top_k=top_k, doc_filter=doc_filter)
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
            results = retriever.search(query, vec, top_k=top_k, doc_filter=doc_filter)
            t1 = time.time()
            print(f"搜索耗时: {(t1-t0)*1000:.1f}ms")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] [{r['doc']}] {r['text']}")

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
            results = retriever.search(search_query, vec, top_k=8, doc_filter=doc_filter)
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
                old_registry = getattr(db, 'doc_registry', {}).copy()
                old_meta = getattr(db, 'meta', []).copy()
                old_deleted = getattr(db, 'deleted', set()).copy()
                old_vecs = None
                if n > 0:
                    old_vecs = np.zeros((n, db.dimension), dtype=np.float32)
                    try:
                        db.index.reconstruct_n(0, n, old_vecs)
                    except RuntimeError:
                        for i in range(n):
                            old_vecs[i] = db.index.reconstruct(i)

                if idx_type == "ivf":
                    db = IvfVectorStore(emb.dimension)
                elif idx_type == "hnsw":
                    db = HnswVectorStore(emb.dimension)
                else:
                    db = FaissVectorStore(emb.dimension)

                if old_texts:
                    db.add_batch(old_texts, old_vecs.tolist())
                    db.doc_registry = old_registry
                    db.meta = old_meta
                    db.deleted = old_deleted  # 恢复删除标记
                    # 重建 BM25 索引
                    retriever._rebuild_bm25(old_texts)
                    log.info(f"已切换到 {idx_type} 索引"
                             f"（已保留 {len(old_texts)} 条数据"
                             f"，{len(old_deleted)} 条标记删除）")
                else:
                    log.info(f"已切换到 {idx_type} 索引（当前为空）")

                retriever.db = db
                ingestion.db = db

                current_type = idx_type

        elif cmd == "/clear":
            print("确认清空所有数据？(yes/no): ", end="", flush=True)
            confirm = input().strip()
            if confirm == "yes":
                # 清空内存数据
                db.texts.clear()
                db.meta.clear()
                db.doc_registry.clear()
                db.deleted.clear()
                # 重建空 FAISS 索引
                import numpy as np
                db.index = db._build_index()
                # 清空 BM25
                retriever._tokenized.clear()
                retriever._bm25 = None
                # 清空 SQLite
                if db._conn is not None:
                    db._conn.execute("DELETE FROM chunks")
                    db._conn.commit()
                log.info("所有数据已清空")
            else:
                print("已取消")

        elif cmd.startswith("/delete "):
            doc_name = cmd[len("/delete "):].strip()
            db.delete_doc(doc_name)
            log.info(f"文档 {doc_name} 已标记删除，数据已自动保存")

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

        elif cmd == "/list":
            docs = list(db.doc_registry.keys())
            if not docs:
                print("（暂无文档）")
            else:
                print(f"共 {len(docs)} 个文档：")
                for d in docs:
                    n = len(db.doc_registry[d])
                    flag = " 🗑️" if any(i in db.deleted for i in db.doc_registry[d]) else ""
                    print(f"  {d}（{n} 个 chunk）{flag}")

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
                # 加载后重建 BM25 索引
                if db.texts:
                    retriever._rebuild_bm25(db.texts)
                log.info(f"已从 {path} 加载（{db.count} 条数据，BM25 已重建）")
            except Exception as e:
                log.error(f"加载失败: {e}")

        else:
            print("未知命令，输入 /help 查看帮助")
