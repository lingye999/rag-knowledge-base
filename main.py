import time
import jieba
import numpy as np
import logging
from embedding import EmbeddingService
from Vector_Store.vector_store_faiss import FaissVectorStore
from Vector_Store.vector_store_ivf import IvfVectorStore
from Vector_Store.vector_store_hnsw import HnswVectorStore

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

def main():
    emb = EmbeddingService()
    current_type = "flat"
    if current_type == "ivf":
        db = IvfVectorStore(emb.dimension)
    elif current_type == "hnsw":
        db = HnswVectorStore(emb.dimension)
    else:
        db = FaissVectorStore(emb.dimension)
    log.info(f"向量搜索系统已启动（当前索引: {current_type}，/help 查看帮助）")

    while True:
        cmd=input().strip()

        if cmd=="/exit":
            log.info("结束")
            break

        elif cmd=="/help":
            print("可用命令：")
            print("  /search <query>  - 搜索（显示耗时）")
            print("  /search_jieba <query>  - 先分词再搜索")
            print("  /add <file>      - 从文件添加")
            print("  /count           - 查看总数")
            print("  /switch <type>   - 切换索引: flat / ivf / hnsw")
            print("  /save <path>     - 保存")
            print("  /load <path>     - 加载")
            print("  /exit            - 退出")

        elif cmd == "/count":
            log.info(f"数据库的条数是{db.count}")


        elif cmd.startswith("/search "):
            rest=cmd[len("/search "):].strip()
            parts = rest.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                query = parts[0]
                top_k = int(parts[1])
            else:
                query = rest
                top_k = 5

            vec=emb.encode(query)
            t0 = time.time()
            results=db.search(vec,top_k=top_k)
            t1 = time.time()
            print(f"搜索耗时: {(t1-t0)*1000:.1f}ms")
            for i,r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] {r['text']}")

        elif cmd.startswith("/search_jieba "):
            rest = cmd[len("/search_jieba "):].strip()
            parts = rest.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                query = parts[0]
                top_k = int(parts[1])
            else:
                query = rest
                top_k = 5

            words = jieba.lcut(query)
            segmented = " ".join(words)
            print(f"分词结果: {segmented}")
            vec = emb.encode(segmented)
            t0 = time.time()
            results = db.search(vec, top_k=top_k)
            t1 = time.time()
            print(f"搜索耗时: {(t1-t0)*1000:.1f}ms")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] {r['text']}")

        elif cmd.startswith("/switch "):
            idx_type = cmd[len("/switch "):].strip()

            if idx_type not in INDEX_TYPES:
                log.warning(f"不支持的索引类型: {idx_type}，可选: flat / ivf / hnsw")
            else:
                # ① 从旧 db 读数据
                n = db.count
                old_texts = db.texts[:]
                old_vecs = np.zeros((n, db.dimension), dtype=np.float32)
                db.index.reconstruct_n(0, n, old_vecs)

                # ② 创建新 db
                if idx_type == "ivf":
                    db = IvfVectorStore(emb.dimension)
                elif idx_type == "hnsw":
                    db = HnswVectorStore(emb.dimension)
                else:
                    db = FaissVectorStore(emb.dimension)

                # ③ 数据加回去
                if old_texts:
                    db.add_batch(old_texts, old_vecs.tolist())
                    log.info(f"已切换到 {idx_type} 索引（已保留 {len(old_texts)} 条数据）")
                else:
                    log.info(f"已切换到 {idx_type} 索引（当前为空）")

                current_type = idx_type


        elif cmd.startswith("/add "):
            try:
                rest = cmd[len("/add "):].strip()
                parts = rest.split(" ")
                path = parts[0]
                method = parts[1] if len(parts) > 1 else "auto"
                db.add_from_file(path, emb, method)
            except Exception as e:
                log.error(f"添加失败: {e}")


        elif cmd.startswith("/save "):
            try:
                path = cmd[len("/save "):].strip()
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

if __name__ == "__main__":
    main()

