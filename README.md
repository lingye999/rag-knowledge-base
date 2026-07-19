# RAG Knowledge Base

基于 FAISS 的智能 RAG（检索增强生成）知识库系统，支持多格式文档解析、多路混合检索、Cross-Encoder 精排、LLM 智能问答。

## 功能特性

- **多格式文档解析**：支持 `.txt` / `.docx` / `.pdf`，PDF 三级降级（Marker → pdfplumber → EasyOCR）
- **多策略文本分块**：auto / sentence / paragraph / jieba / size，智能检测扫描件自动切换
- **多路混合检索**：Dense（语义）+ BM25（关键词）双路召回 + RRF 融合
- **Cross-Encoder 精排**：初排 → 三维加权 → Cross-Encoder 精排，层层提纯
- **LLM 智能服务**：查询改写、自查询过滤、RAG 问答生成（DeepSeek V4 Flash）
- **多种向量索引**：Flat / IVF / HNSW，支持运行时切换并保留数据
- **GPU 加速**：CUDA 加速精排和 OCR（RTX 4060+）
- **持久化存储**：FAISS 索引 + JSON 序列化，文档级元数据管理
- **CLI 交互界面**：命令行搜索、入库、切换索引、问答

## 项目结构

```
rag-knowledge-base/
├── main.py                     # CLI 入口
├── src/
│   ├── cli.py                  # 命令解析与主循环
│   ├── embedding.py            # 文本向量化 (BAAI/bge-small-zh-v1.5)
│   ├── document.py             # 多格式文档读取（三级降级）
│   ├── chunker.py              # 文本分块（5 种策略）
│   ├── cleaner.py              # 文本清洗 + OCR 空白清理
│   ├── plumber.py              # pdfplumber 解析器
│   ├── ocr.py                  # EasyOCR 兜底解析
│   ├── marker_reader.py        # Marker AI 解析器（主力）
│   ├── ingestion.py            # 入库流水线
│   ├── retriever.py            # 检索引擎（召回 + 加权 + 精排）
│   ├── reranker.py             # Cross-Encoder 精排器
│   ├── llm_service.py          # LLM 服务（改写/自查询/问答）
│   └── vector_store/
│       ├── base.py             # 抽象基类
│       ├── faiss_store.py      # Flat 精确索引
│       ├── ivf_store.py        # IVF 倒排索引
│       ├── hnsw_store.py       # HNSW 图索引
│       └── hybrid.py           # Dense + BM25 混合检索
├── tests/
│   └── test_e2e.py             # 端到端测试
└── data/                       # 测试数据目录
```

## 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU + CUDA 12.4（可选，CPU 也能跑）

### 安装

```bash
# 基础依赖
pip install fastembed faiss-cpu numpy jieba python-docx PyMuPDF pdfplumber easyocr rank-bm25 openai

# 精排（Day 3）
pip install sentence-transformers

# GPU 加速
pip install torch --index-url https://download.pytorch.org/whl/cu124

# PDF AI 解析（Day 4）
pip install marker-pdf
```

### 启动

```bash
python main.py
```

## CLI 命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `/add <路径> [方法] [ocr]` | 导入文件 | `/add data/sample.txt auto` |
| `/search <查询> [top_k]` | 语义搜索 + 精排 | `/search 机器学习 5` |
| `/hybrid_search <查询> [top_k]` | Dense+BM25 混合检索 | `/hybrid_search 深度学习 3` |
| `/search_jieba <查询> [top_k]` | 分词后搜索 | `/search_jieba 编程 5` |
| `/ask <问题>` | LLM 问答 | `/ask Python是什么` |
| `/rewrite <查询>` | LLM 查询改写 | `/rewrite 有没有AI的资料` |
| `/switch <类型>` | 切换索引 | `/switch hnsw` |
| `/delete <文档名>` | 删除文档 | `/delete sample.txt` |
| `/count` | 查看总数 | `/count` |
| `/save <路径>` | 保存 | `/save data/my_db` |
| `/load <路径>` | 加载 | `/load data/my_db` |
| `/help` | 帮助 | `/help` |
| `/exit` | 退出 | `/exit` |

## 检索流程

```
查询 → LLM改写 → 分词/编码
                     │
             多路召回（Dense + BM25）
                     │
               RRF 融合排序
                     │
               三维度加权（α=0.7 β=0.1 γ=0.2）
                     │
            Cross-Encoder 精排（GPU）
                     │
               返回 top_k 结果
```

## 后续计划

- [x] Dense + BM25 混合检索 + RRF 融合
- [x] LLM 智能检索（查询改写 + Self-Querying）
- [x] Cross-Encoder 精排
- [x] Marker AI PDF 解析
- [ ] SQLite 持久化 + 质量评分
- [ ] FastAPI Web 接口
- [ ] 知识库治理与监控
