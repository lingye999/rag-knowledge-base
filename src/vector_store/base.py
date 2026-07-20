from abc import ABC, abstractmethod
import json
import os
import sqlite3
import faiss
import numpy as np


class BaseVectorStore(ABC):
    """统一向量存储基类

    子类只需实现 _build_index() 和可选的 _before_add()。

    持久化：
        - 每次 add_batch / delete_doc 自动写入 SQLite（文本 + 向量 + 元数据）
        - load() 时可以完全从 SQLite 重建（不需要 .faiss 文件也能恢复）
        - .faiss 文件只作为加载加速缓存，丢了也不影响数据恢复
    """

    DB_SUFFIX = ".db"

    def __init__(self, dimension: int,
                 index_type: str = "flat",
                 index_params: dict = None):
        self.dimension = dimension
        self.texts: list[str] = []
        self.doc_registry: dict[str, list[int]] = {}
        self.meta: list[dict] = []
        self.deleted: set[int] = set()
        self._index_type = index_type
        self._index_params = index_params or {}
        self.index = self._build_index()

        # SQLite 持久化
        self._db_path: str | None = None
        self._conn: sqlite3.Connection | None = None
        self._save_path_prefix: str | None = None  # 最近一次 save/load 的路径

    # ── 子类工厂 ──

    @abstractmethod
    def _build_index(self):
        """返回 FAISS Index 实例（每个子类不同）"""
        ...

    def _before_add(self, vectors: np.ndarray):
        """Hook：添加前的钩子（IVF 用它做 train）"""
        pass

    # ── 核心操作 ──

    def add_batch(self,
                  texts: list[str],
                  vectors: list[list[float]],
                  doc_name: str | None = None,
                  qualities: list[float] | None = None):
        """批量添加文本和向量

        自动写入 SQLite + 自动保存 FAISS（如果有 save 路径）
        """
        start = len(self.texts)
        vecs = self._to_normalized(vectors)

        self._before_add(vecs)
        self.index.add(vecs)
        self.texts.extend(texts)

        if doc_name is not None:
            indices = list(range(start, start + len(texts)))
            if doc_name in self.doc_registry:
                self.doc_registry[doc_name].extend(indices)
            else:
                self.doc_registry[doc_name] = indices
            for i, text in enumerate(texts):
                entry = {"doc": doc_name}
                if qualities and i < len(qualities):
                    entry["quality"] = qualities[i]
                self.meta.append(entry)

        # 增量写入 SQLite + 自动保存 FAISS
        self._sync_add_batch(start, texts, vectors, doc_name, qualities)
        self._auto_save()

    def _sync_add_batch(self, start: int, texts: list[str],
                        vectors: list[list[float]],
                        doc_name: str | None,
                        qualities: list[float] | None):
        """INSERT 文本 + 向量 + 质量分到 SQLite"""
        if self._conn is None:
            return
        doc = doc_name or ""
        for i, text in enumerate(texts):
            pos = start + i
            q = qualities[i] if qualities and i < len(qualities) else 0.5
            vec_bytes = np.array(vectors[i], dtype=np.float32).tobytes()
            self._conn.execute(
                "INSERT INTO chunks (position, text, doc_name, quality, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                (pos, text, doc, q, vec_bytes)
            )
        self._conn.commit()

    def search(self, query_vec: list[float],
               top_k: int = 5) -> list[dict]:
        """向量搜索"""
        return self._run_search(self.index, query_vec, top_k)

    @property
    def count(self) -> int:
        return self.index.ntotal

    def get_chunks_by_doc(self, doc_name: str) -> list[str]:
        indices = self.doc_registry.get(doc_name, [])
        return [self.texts[i] for i in indices]

    def delete_doc(self, doc_name: str):
        """标记删除文档（更新内存 + SQLite + 自动保存）"""
        indices = self.doc_registry.pop(doc_name, [])
        self.deleted.update(indices)

        if self._conn is not None:
            for pos in indices:
                self._conn.execute(
                    "UPDATE chunks SET deleted = 1 WHERE position = ?",
                    (pos,)
                )
            self._conn.commit()

        self._auto_save()

    # ── SQLite 持久化 ──

    def _connect_db(self, path: str) -> sqlite3.Connection:
        """打开（或创建）SQLite 数据库并建表"""
        db_path = f"{path}{self.DB_SUFFIX}"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                position INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                doc_name TEXT NOT NULL DEFAULT '',
                quality REAL NOT NULL DEFAULT 0.5,
                vector BLOB,
                deleted INTEGER NOT NULL DEFAULT 0
            )
        """)
        # 向前兼容：旧数据库可能缺少某些列
        for col in ["quality", "vector"]:
            try:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} ?",
                             ["REAL" if col == "quality" else "BLOB"])
            except sqlite3.OperationalError:
                pass
        conn.commit()
        self._db_path = db_path
        self._conn = conn
        return conn

    def _save_to_db(self, path: str):
        """将全部内存数据 + 向量批量写入 SQLite"""
        conn = self._connect_db(path)
        conn.execute("DELETE FROM chunks")
        for pos, (text, meta_entry) in enumerate(zip(self.texts, self.meta)):
            doc_name = meta_entry.get("doc", "") if meta_entry else ""
            quality = meta_entry.get("quality", 0.5) if meta_entry else 0.5
            deleted = 1 if pos in self.deleted else 0
            # 从 FAISS index 提取向量
            vec_bytes = None
            try:
                vec = self.index.reconstruct(pos)
                vec_bytes = vec.tobytes()
            except RuntimeError:
                pass
            conn.execute(
                "INSERT INTO chunks (position, text, doc_name, quality, vector, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (pos, text, doc_name, quality, vec_bytes, deleted)
            )
        conn.commit()

    def _rebuild_index_from_db(self, rows: list[tuple]) -> faiss.Index:
        """从 SQLite 行数据重建 FAISS 索引"""
        vecs = []
        for _, _, _, _, vec_bytes, _ in rows:
            if vec_bytes:
                vec = np.frombuffer(vec_bytes, dtype=np.float32).reshape(1, -1)
                vecs.append(vec[0])
        if not vecs:
            return self._build_index()
        all_vecs = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(all_vecs)
        new_index = self._build_index()
        # 如果建的是 IVF，需要 train
        if hasattr(new_index, 'train') and not new_index.is_trained:
            new_index.train(all_vecs)
        new_index.add(all_vecs)
        return new_index

    def _load_from_db(self, path: str):
        """从 SQLite 读取全部数据到内存，并从向量重建 FAISS 索引"""
        conn = self._connect_db(path)
        cursor = conn.execute(
            "SELECT position, text, doc_name, quality, vector, deleted "
            "FROM chunks ORDER BY position"
        )
        rows = cursor.fetchall()

        n = len(rows)
        self.texts = [""] * n
        self.meta = [{} for _ in range(n)]
        self.deleted = set()
        self.doc_registry = {}

        for pos, text, doc_name, quality, _, deleted in rows:
            self.texts[pos] = text
            self.meta[pos] = {"doc": doc_name, "quality": quality}
            if deleted:
                self.deleted.add(pos)
            if doc_name:
                self.doc_registry.setdefault(doc_name, []).append(pos)

        # 从 SQLite 里的向量重建 FAISS 索引
        self.index = self._rebuild_index_from_db(rows)

    # ── 文件持久化 ──

    def save(self, path: str):
        """保存 FAISS 索引 + SQLite"""
        self._save_path_prefix = path
        if not os.path.exists(os.path.dirname(path) or "."):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            faiss.write_index(self.index, f"{path}.faiss")
        except Exception:
            pass  # 某些索引类型可能不支持写文件，忽略
        self._save_to_db(path)

    def load(self, path: str):
        """加载数据（优先 SQLite，兼容旧 JSON + FAISS）"""
        self._save_path_prefix = path
        db_path = f"{path}{self.DB_SUFFIX}"
        json_path = f"{path}_texts.json"

        if os.path.exists(db_path):
            self._load_from_db(path)
        elif os.path.exists(json_path):
            # 旧格式兼容
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.texts = data
            else:
                self.texts = data["texts"]
                self.doc_registry = data.get("doc_registry", {})
                self.meta = data.get("meta", [])
            self.deleted = set()
            # 读取 .faiss 文件重建索引
            faiss_path = f"{path}.faiss"
            if os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
                self.dimension = self.index.d
            else:
                raise FileNotFoundError(f"找不到 {faiss_path}")
        else:
            raise FileNotFoundError(
                f"找不到数据库文件 {db_path} 或 {json_path}"
            )

    def _auto_save(self):
        """自动保存 FAISS + SQLite 到最后一次 save/load 的路径"""
        if self._save_path_prefix is not None:
            self.save(self._save_path_prefix)

    # ── 删除压缩（事务安全）──

    def _compact(self):
        """物理删除已标记数据，重建索引 + SQLite + 自动保存"""
        if not self.deleted:
            return

        alive = [i for i in range(len(self.texts)) if i not in self.deleted]

        # 重建向量
        try:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        except RuntimeError:
            n = self.index.ntotal
            all_vectors = np.zeros((n, self.dimension), dtype=np.float32)
            for i in range(n):
                all_vectors[i] = self.index.reconstruct(i)

        all_vectors = all_vectors[alive]
        vecs = self._to_normalized(all_vectors)

        # 在局部变量中构建新索引
        idx_type = self._index_type
        if idx_type == "hnsw":
            M = self._index_params.get("M", 32)
            new_index = faiss.IndexHNSWFlat(self.dimension, M,
                                            faiss.METRIC_INNER_PRODUCT)
            new_index.hnsw.efConstruction = 128
            new_index.hnsw.efSearch = 64
            new_index.add(vecs)
        elif idx_type == "ivf":
            nlist = self._index_params.get("nlist", 100)
            quantizer = faiss.IndexFlatIP(self.dimension)
            new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist,
                                           faiss.METRIC_INNER_PRODUCT)
            new_index.nprobe = 10
            new_index.train(vecs)
            new_index.add(vecs)
        else:
            new_index = faiss.IndexFlatIP(self.dimension)
            new_index.add(vecs)

        # 在局部变量中准备 text / meta / registry
        new_texts = [self.texts[i] for i in alive]
        new_meta = [self.meta[i] for i in alive]

        old_to_new = {old: new for new, old in enumerate(alive)}
        new_registry: dict[str, list[int]] = {}
        for doc_name, old_indices in self.doc_registry.items():
            new_indices = [old_to_new[i] for i in old_indices if i in old_to_new]
            if new_indices:
                new_registry[doc_name] = new_indices

        # ── 原子替换 ──
        self.index = new_index
        self.texts = new_texts
        self.meta = new_meta
        self.doc_registry = new_registry
        self.deleted.clear()

        # 自动保存
        self._auto_save()

    # ── 工具方法 ──

    def _to_normalized(self, vectors: list[list[float]]) -> np.ndarray:
        vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    def _run_search(self, index, query_vec: list[float],
                    top_k: int = 5) -> list[dict]:
        q = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, indices = index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(score),
                    "index": int(idx),
                })
        return results
