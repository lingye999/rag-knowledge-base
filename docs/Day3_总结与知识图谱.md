# Day3 总结与知识图谱

> 日期：2026-07-17
> 项目：RAG 知识库系统
> 分支：day3
> Commit: cda77cb

---

## 一、今日完成功能清单

### Phase A：存储基础设施升级

| 模块 | 功能 | 涉及文件 |
|------|------|---------|
| 三层数据结构 | `doc_registry`（文档→位置）、`meta`（位置→来源）、`deleted`（标记删除） | faiss/ivf/hnsw_store.py |
| 双向索引 | `add_batch` 支持 `doc_name` 参数，自动维护 doc_registry 和 meta | 同上 |
| 文档操作 | `get_chunks_by_doc`（按文档取 chunk）、`delete_doc`（标记删除）、`_compact`（物理重建） | base.py |
| 索引保持 | `_compact` 根据 `_index_type` 保持 HNSW/IVF/Flat 类型不退化 | base.py + 三个 store |
| 持久化 | save/load 现在保存和恢复 doc_registry + meta，兼容旧格式（纯列表） | 三个 store |
| CLI 联动 | `/add` 传文件名、新增 `/delete` 命令、`/save` 自动 compact | cli.py |
| 清理 | 删除僵尸代码 `add_from_file`（4 处定义 + 冗余 import） | 所有 store + base.py |

### Phase B：检索引擎

| 步骤 | 功能 | 涉及文件 |
|------|------|---------|
| B1 Self-Querying | LLM 从自然语言提取 `(semantic_query, filters)`，支持文档过滤 | llm_service.py |
| B2-B5 统一引擎 | Route A（chunk 搜索）+ Route B（文档聚合）+ 三维度加权 + 阈值过滤 | retriever.py（新建） |
| 归一化修正 | 从经验 min/max 改为理论锚点归一化，消除批次差异 | retriever.py |
| 入库封装 | IngestionService 封装 read→chunk→encode→add→hybrid 全流程 | ingestion.py（新建） |

### 架构优化

| 优化 | 说明 | 影响 |
|------|------|------|
| HybridRetriever 共享 db | 不再自己 new FaissVectorStore，改为接收外部引用 | 消除数据重复存储，内存减半 |
| IngestionService | cli.py 的 /add 从 15 行手动组装缩减到 4 行 | 将来加 Web API 可直接复用 |
| retriever 依赖精简 | 去掉 self.hybrid (未使用) 和 self.emb (编码外移) | 职责更清晰 |

---

## 二、当前系统架构

```
main.py → cli.py
              │
              ├── IngestionService  ──→ document.py → plumber.py / ocr.py
              │       │                    ↓
              │       │              chunker.py + cleaner.py
              │       │                    ↓
              │       └── embedding.py → db.add_batch(doc_name) + hybrid.add_texts
              │
              ├── Retriever ──→ db.search() → 文档聚合 → 加权 → 阈值
              │       │
              │       └── self_query() → doc_filter
              │
              └── LLMService (rewrite / ask / self_query)
```

### 依赖方向

```
cli.py → retriever.py → db
cli.py → ingestion.py → db + hybrid
cli.py → llm_service.py (独立)

底层模块（cleaner/chunker/document/embedding）零反向依赖
```

### 数据流：搜索全链路

```
用户输入 "说明书里E-VAC的参数"
    │
    ├── _rewrite()        →  "E-VAC参数 说明书"
    ├── self_query()      →  ("E-VAC参数", {"doc": "说明书.pdf"})
    ├── emb.encode()      →  向量
    │
    ├── Route A: db.search(向量) × 15条     → chunk 候选
    ├── Route B: 按文档聚合 chunk 分数      → doc_scores
    ├── 三维度加权: α0.7×chunk + β0.1×doc + γ0.2×0.8
    ├── Threshold 过滤:  < 0.3 丢弃
    └── 返回 top_k
```

---

## 三、核心知识点

### 1. 双向索引（doc_registry ↔ meta）

```
                    doc_registry["A.pdf"] = [0,1,2]
                          │
                          ▼
位置:   [0]      [1]      [2]      [3]      [4]
       ┌────    ────    ────    ────    ────┐
FAISS  │ vec0    vec1    vec2    vec3    vec4 │
texts  │ t0      t1      t2      t3      t4  │
meta   │ {A}     {A}     {A}     {B}     {B} │
       └────    ────    ────    ────    ────┘
```

- `meta[i]` → 从位置查来源（搜索结果溯源）
- `doc_registry[d]` → 从文档查所有位置（删除、文档级召回）
- 两者共享同一套位置编号，互为反向

### 2. 标记删除 vs 物理删除

```
标记删除（delete_doc）:
  deleted.add(0,1,2)         ← FAISS 和 texts 不动，只加标记

物理删除（_compact）:
  1. 找出 alive（不在 deleted 里的位置）
  2. 从旧 FAISS reconstruct 向量 → 只保留 alive → 建新索引
  3. 同步清理 texts / meta
  4. 重映射 doc_registry（因为位置编号变了）
  5. 清空 deleted
```

### 3. 归一化的两种方式

| 方式 | 做法 | 问题 |
|------|------|------|
| 经验归一化（已废弃） | `(score - min) / (max - min)` | 同文档内微小差异被放大；垃圾结果被标满分 |
| 理论锚点（当前） | `score` 直接使用（余弦相似度天然 [0, 1]）；`doc_rrf / 1.0` 截断 | 无批次差异，threshold 真正有意义 |

### 4. 数据所有权

- **修复前：** HybridRetriever 自己 new FaissVectorStore，和 db 各存一份数据
- **修复后：** HybridRetriever 接收外部 db 引用，只维护 BM25 分词索引
- **原则：** 数据只存一份，检索策略是数据的"用户"而非"拥有者"

### 5. Self-Querying 设计

```
输入: "说明书里E-VAC的参数"
  → LLM 提取: {"semantic_query": "E-VAC参数", "filters": {"doc": "说明书.pdf"}}
  → 搜索时先全局搜，再按 meta["doc"] 过滤
  → 没有 filter 时传 None，不影响正常搜索
```

### 6. 三维度加权公式

```
finalScore = 0.7 × α(chunk相关性) + 0.1 × β(文档排名) + 0.2 × γ(文档质量)

α = 余弦相似度原始值 [0, 1]
β = doc_rrf / 1.0，截断 [0, 1]
γ = 0.8（暂用默认值）

finalScore = rawScore × (1 - threshold) + threshold   ← 映射到 [threshold, 1.0]
```

### 7. IngestionService（关注点分离）

```
入库 = 读文件 + 分块 + 向量化 + 双路存索引

这些逻辑原来散落在 cli.py 里（15 行手动组装），
现在收进 IngestionService.add()（cli.py 只调一行）。
好处：Web API、定时任务可复用。
```

---

## 四、当前架构问题（已知待改进）

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | save/load 在三个子类中重复 | 🟡 | 待提至 base.py |
| 2 | retriever 直接访问 db.meta[r["index"]] | 🟡 | 应通过 db.get_doc_name(i) |
| 3 | Route B 只是聚合，不是独立文档搜索 | 🟢 | 可用 chunk 向量取平均做文档向量 |
| 4 | hybrid.add_texts 只支持一次性添加 | 🟢 | 需改为增量 BM25 |
| 5 | `/hybrid_search` 未接入 retriever 三维度加权 | 🟢 | 待统一 |

---

## 五、下一步

| 任务 | 说明 |
|------|------|
| 端到端测试 | 在 PyCharm 中运行 tests/test_e2e.py |
| Day 4：Marker 解析器 | 替换 pdfplumber，根治文本提取质量 |
| 代码复用优化 | save/load 提至 base.py、meta 访问封装 |
| 扩展方向 | FastAPI Web 服务 / Docker 容器化 / 抄 LlamaIndex |
