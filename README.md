# RAG Knowledge Base

基于 FAISS 的轻量级知识库向量检索系统，支持多种文件格式导入、多策略文本分块、多索引类型切换，为 RAG（检索增强生成）场景提供知识检索基础能力。

## 功能特性

- **多格式文档解析**：支持 `.txt` / `.docx` / `.pdf` 文件导入
- **多策略文本分块**：自动选择（auto）、按句子、按段落、按大小、jieba 分词
- **多种向量索引**：Flat（精确）、IVF（倒排加速）、HNSW（图索引），支持运行时切换
- **持久化存储**：索引和文本可保存/加载（FAISS 二进制格式 + JSON）
- **CLI 交互界面**：命令行直接搜索、添加、切换索引

## 项目结构

```
rag-knowledge-base/
├── main.py                    # CLI 交互入口
├── document_loader.py         # 文档读取 + 文本分块
├── embedding.py               # 文本向量化服务（BGE 模型）
├── generate_test_files.py     # 生成测试文件（.txt / .docx / .pdf）
├── Vector_Store/
│   ├── base_vector_store.py   # 抽象基类（统一接口规范）
│   ├── vector_store_faiss.py  # Flat 索引（精确检索）
│   ├── vector_store_ivf.py    # IVF 索引（倒排加速）
│   ├── vector_store_hnsw.py   # HNSW 索引（图索引）
│   └── simple_vector_store.py # 纯 Python 暴力检索（无 FAISS 依赖）
└── data/                      # 测试数据目录
```

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install fastembed faiss-cpu numpy jieba python-docx PyMuPDF
```

### 生成测试数据

```bash
python generate_test_files.py
```

这会在 `data/` 目录下生成三个不同主题的测试文件。

### 启动

```bash
python main.py
```

## CLI 命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `/add <文件路径> [分块方法]` | 导入文件 | `/add data/AI_概述.txt auto` |
| `/search <查询词> [top_k]` | 语义搜索 | `/search 人工智能 5` |
| `/search_jieba <查询词> [top_k]` | 先分词再搜索 | `/search_jieba 机器学习 3` |
| `/count` | 查看索引条数 | `/count` |
| `/switch <索引类型>` | 切换索引 | `/switch hnsw` |
| `/save <路径>` | 保存索引 | `/save ./backup` |
| `/load <路径>` | 加载索引 | `/load ./backup` |
| `/help` | 帮助 | `/help` |
| `/exit` | 退出 | `/exit` |

### 分块方法

| 方法 | 说明 |
|------|------|
| `auto` | 自动选择（根据中文占比和段落数智能判断） |
| `sentence` | 按句号/感叹号/问号分割（支持中英文） |
| `paragraph` | 按换行分割 |
| `jieba` | jieba 分词后按词数分块 |
| `size` | 固定窗口滑动分块 |

### 索引类型

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| `flat` | 精确检索，内存占用大 | 小数据量（< 10万） |
| `ivf` | 倒排索引，检索快 | 中等数据量 |
| `hnsw` | 图索引，检索最快 | 大数据量 |

## 使用示例

```
$ python main.py
[Embedding] 模型 BAAI/bge-small-zh-v1.5 设备 cpu 加载成功，维度: 512
向量搜索系统已启动（当前索引: flat，/help 查看帮助）

/add data/AI_概述.txt auto
[Auto] 中文占比 99%，段落 11 个，使用 jieba 分块
文件 data/AI_概述.txt 加载完成: 33 个文本块

/search 机器学习 3
搜索耗时: 1.2ms
1. [0.7265] 人工智能概述...
2. [0.6183] 机器学习是 AI 的核心技术之一...
3. [0.5912] 深度学习是机器学习的一个子领域...

/switch hnsw
已切换到 hnsw 索引（已保留 33 条数据）
```

## 后续路线

- [ ] 混合检索（BM25 + Dense 双路融合 + RRF 排序）
- [ ] LLM 智能检索（查询改写 + Self-Querying + 自动打标）
- [ ] 入库流水线（文本清洗 + 质量评分 + SQLite 持久化）
- [ ] FastAPI Web 接口
- [ ] 知识库治理（热度衰减 + 健康巡检）
