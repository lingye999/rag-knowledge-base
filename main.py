import time
import jieba
from embedding import EmbeddingService
from vector_store_faiss import FaissVectorStore
from vector_store_ivf import IvfVectorStore
from vector_store_hnsw import HnswVectorStore
from document_loader import read_file, chunk_by_sentence

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
    print(f"向量搜索系统已启动（当前索引: {current_type}，/help 查看帮助）")

    while True:
        cmd=input().strip()

        if cmd=="/exit":
            print("结束")
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
            # TODO: 打印数据库条数
            print(f"数据库的条数是{db.count}")


        elif cmd.startswith("/search "):
            query=cmd[len("/search "):]
            vec=emb.encode(query)
            t0 = time.time()
            results=db.search(vec,top_k=5)
            t1 = time.time()
            print(f"搜索耗时: {(t1-t0)*1000:.1f}ms")
            for i,r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] {r['text']}")

        elif cmd.startswith("/search_jieba "):
            query = cmd[len("/search_jieba "):]
            words = jieba.lcut(query)
            segmented = " ".join(words)
            print(f"分词结果: {segmented}")
            vec = emb.encode(segmented)
            t0 = time.time()
            results = db.search(vec, top_k=5)
            t1 = time.time()
            print(f"搜索耗时: {(t1-t0)*1000:.1f}ms")
            for i, r in enumerate(results):
                print(f"{i+1}. [{r['score']:.4f}] {r['text']}")

        elif cmd.startswith("/switch "):
            idx_type = cmd[len("/switch "):].strip()
            if idx_type not in INDEX_TYPES:
                print(f"不支持的索引类型: {idx_type}，可选: flat / ivf / hnsw")
            else:
                if idx_type == "ivf":
                    db = IvfVectorStore(emb.dimension)
                elif idx_type == "hnsw":
                    db = HnswVectorStore(emb.dimension)
                else:
                    db = FaissVectorStore(emb.dimension)
                current_type = idx_type
                print(f"已切换到 {idx_type} 索引（数据为空，请重新添加）")

        elif cmd.startswith("/add "):
            rest = cmd[len("/add "):].strip()
            parts = rest.split(" ")  # 按空格切分
            path = parts[0]  # 第一个是文件路径
            method = parts[1] if len(parts) > 1 else "sentence"  # 第二个是分块方式，默认 sentence
            db.add_from_file(path, emb, method)


        elif cmd.startswith("/save "):
            # TODO: 提取路径 → save
            path=cmd[len("/save "):]
            db.save(path)

        elif cmd.startswith("/load "):
            # TODO: 提取路径 → load
            path=cmd[len("/load "):]
            db.load(path)

        else:
            print("未知命令，输入 /help 查看帮助")

if __name__ == "__main__":
    main()

