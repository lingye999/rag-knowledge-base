"""CLI 命令解析与主循环"""
import time
import os
import jieba
import torch
from config import config
from .logger import get_logger, setup_logging
from .embedding import EmbeddingService
from .vector_store.faiss_store import FaissVectorStore
from .vector_store.ivf_store import IvfVectorStore
from .vector_store.hnsw_store import HnswVectorStore
from .llm_service import LLMService
from .retriever import Retriever
from .ingestion import IngestionService
from .reranker import Reranker
from .index_service import IndexService

log = get_logger("cli")
setup_logging(
    level=config["logging"]["level"],
    log_file=config["logging"]["file"],
    fmt=config["logging"]["format"],
)

INDEX_TYPES = {
    "flat": FaissVectorStore,
    "ivf": IvfVectorStore,
    "hnsw": HnswVectorStore,
}


def _create_index(index_type: str, dimension: int):
    if index_type == "ivf":
        return IvfVectorStore(dimension, nlist=config["index"]["ivf"]["nlist"])
    if index_type == "hnsw":
        return HnswVectorStore(dimension, M=config["index"]["hnsw"]["M"])
    return FaissVectorStore(dimension)


def run():
    cfg_embedding = config["embedding"]
    emb = EmbeddingService(
        model_name=cfg_embedding["model"],
        device=cfg_embedding["device"],
    )
    current_type = config["index"]["type"]

    idx_cls = INDEX_TYPES.get(current_type)
    if idx_cls is None:
        log.warning(f"不支持的索引类型: {current_type}，回退到 flat")
        db = FaissVectorStore(emb.dimension)
    else:
        db = _create_index(current_type, emb.dimension)
    log.info("系统启动", index_type=current_type, dim=emb.dimension)

    # LLM
    cfg_llm = config["llm"]
    try:
        llm = LLMService(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            model=cfg_llm["model"],
            base_url=cfg_llm["base_url"],
        )
        log.info("LLM已就绪", model=cfg_llm["model"])
    except ValueError as e:
        llm = None
        log.warning("LLM未配置", error=str(e))

    # 检索引擎
    cfg_reranker = config["reranker"]
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reranker = Reranker(cfg_reranker["model"], device=device)
        log.info("重排序器已就绪", model=cfg_reranker["model"])
    except Exception as e:
        reranker = None
        log.warning("重排序器未加载", error=str(e))
    retriever = Retriever(db, reranker=reranker)
    index_service = IndexService(db, index_factory=_create_index)

    # 入库
    ingestion = IngestionService(emb, db, retriever)
    index_service.bind_consumers(retriever=retriever, ingestion=ingestion)

    cfg_search = config["retrieval"]

    def _rewrite(query: str) -> str:
        if llm is None:
            return query
        try:
            rewritten = llm.rewrite(query)
            if rewritten and not rewritten.startswith("[改写失败"):
                return rewritten
        except Exception:
            pass
        return query

    while True:
        cmd = input().strip()

        if cmd == "/exit":
            if index_service.has_save_path:
                log.info("自动保存中")
                try:
                    index_service.save()
                except Exception as e:
                    log.error("自动保存失败", error=str(e))
            log.info("系统退出")
            break

        elif cmd == "/help":
            print("可用命令：")
            print("  /search <query>         - 搜索（Dense + BM25 混合）")
            print("  /search_jieba <query>   - 先分词再搜索")
            print("  /ask <query>            - LLM 回答（检索+AI生成）")
            print("  /rewrite <query>        - LLM 改写查询为关键词")
            print("  /add <file>             - 从文件添加（默认混合模式）")
            print("  /add <file> ocr         - 强制 OCR 模式添加")
            print("  /add <file> marker      - Marker 深度解析（需 GPU+本地模型）")
            print("  /count                  - 查看总数")
            print("  /switch <type>          - 切换索引: flat / ivf / hnsw")
            print("  /delete <doc_name>      - 标记删除文档")
            print("  /list                  - 查看已有文档列表")
            print("  /clear                 - 清空所有数据（需确认）")
            print("  /save <path>            - 保存")
            print("  /load <path>            - 加载")
            print("  /exit                   - 退出")

        elif cmd == "/count":
            log.info("查询总数", count=db.count)
            print(f"数据库共 {db.count} 条")

        elif cmd.startswith("/search "):
            rest = cmd[len("/search "):].strip()
            parts = rest.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                query, top_k = parts[0], int(parts[1])
            else:
                query, top_k = rest, cfg_search["top_k"]

            query = _rewrite(query)
            doc_filter = None
            if llm is not None:
                sq, filters = llm.self_query(query)
                if filters.get("doc"):
                    doc_filter = filters["doc"]
                query = sq

            t0 = time.time()
            vec = emb.encode(query)
            results = retriever.search(query, vec, top_k=top_k, doc_filter=doc_filter)
            t1 = time.time()
            elapsed = (t1 - t0) * 1000
            log.info("搜索完成", query=rest[:40], elapsed_ms=round(elapsed, 1),
                     results=len(results))
            print(f"搜索耗时: {elapsed:.0f}ms")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] [{r['doc']}] {r['text'][:80]}")

        elif cmd.startswith("/search_jieba "):
            rest = cmd[len("/search_jieba "):].strip()
            parts = rest.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                query, top_k = parts[0], int(parts[1])
            else:
                query, top_k = rest, cfg_search["top_k"]

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
            elapsed = (t1 - t0) * 1000
            log.info("分词搜索完成", query=rest[:40], elapsed_ms=round(elapsed, 1))
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] [{r['doc']}] {r['text'][:80]}")

        elif cmd.startswith("/ask "):
            if llm is None:
                print("LLM 未配置")
                continue
            query = cmd[len("/ask "):].strip()
            sq = _rewrite(query)

            doc_filter = None
            if llm is not None:
                sq2, filters = llm.self_query(sq)
                if filters.get("doc"):
                    doc_filter = filters["doc"]
                sq = sq2

            t0 = time.time()
            vec = emb.encode(sq)
            results = retriever.search(sq, vec, top_k=8, doc_filter=doc_filter)
            chunks = [r["text"] for r in results]
            t1 = time.time()
            log.info("检索完成", query=query[:40], elapsed_ms=round((t1-t0)*1000, 1),
                     chunks=len(chunks))

            if chunks:
                log.info("LLM生成中")
                answer = llm.ask(query, chunks)
                print(f"\n{answer}")
            else:
                print("未检索到相关内容")

        elif cmd.startswith("/rewrite "):
            if llm is None:
                print("LLM 未配置")
                continue
            query = cmd[len("/rewrite "):].strip()
            rewritten = llm.rewrite(query)
            print(f"改写前: {query}")
            print(f"改写后: {rewritten}")

        elif cmd.startswith("/switch "):
            idx_type = cmd[len("/switch "):].strip()
            if idx_type not in INDEX_TYPES:
                log.warning("不支持的索引类型", type=idx_type)
            else:
                n = index_service.count
                index_service.switch_index(idx_type)
                index_service.bind_consumers(
                    retriever=retriever, ingestion=ingestion
                )
                db = index_service.vector_store
                current_type = idx_type
                log.info("索引已切换", to=idx_type, count=n)

        elif cmd.startswith("/delete "):
            doc_name = cmd[len("/delete "):].strip()
            index_service.delete_document(doc_name)
            log.info("文档已标记删除", doc=doc_name)

        elif cmd.startswith("/add "):
            try:
                rest = cmd[len("/add "):].strip()
                parts = rest.split(" ")
                path = parts[0]
                if not path:
                    print("用法: /add <文件路径> [ocr|marker]")
                    continue
                method = "auto"
                for p in parts[1:]:
                    if p in ("ocr", "marker"):
                        continue
                    method = p
                force_ocr = "ocr" in parts
                use_marker = "marker" in parts
                n, fname = ingestion.add(
                    path, chunk_method=method,
                    force_ocr=force_ocr, use_marker=use_marker
                )
                if n:
                    log.info("文件入库完成", file=fname, chunks=n)
            except Exception as e:
                log.error("添加失败", error=str(e))

        elif cmd == "/clear":
            print("确认清空所有数据？(yes/no): ", end="", flush=True)
            confirm = input().strip()
            if confirm == "yes":
                index_service.clear()
                retriever._tokenized.clear()
                retriever._bm25 = None
                log.info("所有数据已清空")
            else:
                print("已取消")

        elif cmd == "/list":
            documents = index_service.list_documents()
            docs = [item["doc"] for item in documents if not item["deleted"]]
            if not docs:
                print("（暂无文档）")
            else:
                print(f"共 {len(docs)} 个文档：")
                for item in documents:
                    flag = " [deleted]" if item["deleted"] else ""
                    print(f"  {item['doc']} ({item['count']} chunks){flag}")

        elif cmd.startswith("/save "):
            try:
                path = cmd[len("/save "):].strip()
                index_service.compact()
                index_service.save(path)
                log.info("已保存", path=path, count=db.count)
            except Exception as e:
                log.error("保存失败", error=str(e))

        elif cmd.startswith("/load "):
            try:
                path = cmd[len("/load "):].strip()
                index_service.load(path)
                index_service.bind_consumers(
                    retriever=retriever, ingestion=ingestion
                )
                db = index_service.vector_store
                log.info("已加载", path=path, count=db.count)
            except Exception as e:
                log.error("加载失败", error=str(e))

        else:
            print("未知命令，输入 /help 查看帮助")
