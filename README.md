# RAG Knowledge Base

基于 FAISS 的智能 RAG（检索增强生成）知识库系统，从零实现的深度学习项目。

## 功能特性

- **多格式文档解析**：`.txt` / `.docx` / `.pdf`，PDF 三级选择（轻量混合 → EasyOCR → Marker）
- **轻量混合 PDF 读取**：逐块乱码检测（显式乱码/连续重复/罕用汉字/标点异常）→ 选择性 OCR 替换，参考 RAGFlow DeepDoc 思路
- **多策略文本分块**：auto / sentence / paragraph / jieba / size，智能检测扫描件自动切换，参数对齐 LangChain / LlamaIndex 标准
- **多路混合检索**：Dense（语义）+ BM25（关键词）双路召回 + RRF 融合，支持 recall_expand 调节候选数
- **三维度加权评分**：α×chunk 相关性 + β×文档排名 + γ×质量分，支持动态质量评分
- **Cross-Encoder 精排**：log 长度归一化消除短文本偏置
- **LLM 智能服务**：查询改写、自查询过滤、RAG 问答生成（DeepSeek）
- **多种向量索引**：Flat / IVF / HNSW，支持运行时切换并保留数据
- **SQLite 自动持久化**：向量 BLOB 存储 + FAISS 索引自动重建，无需额外 `.faiss` 文件
- **动态质量评分 v2**：统计指标 + OCR 重复检测 + jieba 碎片分类
- **YAML 配置中心**：环境变量可覆写，修改参数无需改代码
- **结构化日志**：支持 JSON 格式，文件 + 控制台双输出
- **评估框架**：解析检查 / 检索基线 / QA 黄金集，覆盖文档抽取、召回证据和回答约束

## 项目结构

```
rag-knowledge-base/
├── config/
│   ├── __init__.py            # YAML 配置加载器（支持环境变量覆写）
│   └── default.yaml           # 默认配置（embedding/chunking/retrieval/...）
├── eval/
│   ├── extraction_checks.jsonl # PDF 解析质量检查
│   ├── retrieval_queries.jsonl # 检索证据集
│   ├── qa_golden.jsonl         # QA 黄金答案约束
│   ├── document_manifest.json  # 受测文档哈希清单
│   ├── evaluation.py           # 评测判定规则
│   ├── run_extraction_checks.py # 解析检查入口
│   ├── run_baseline.py         # 检索基线入口
│   └── validate_dataset.py     # 测试集校验入口
├── main.py                    # CLI 入口
├── src/
│   ├── cli.py                 # 命令解析与主循环
│   ├── embedding.py           # 文本向量化 (BAAI/bge-small-zh-v1.5)
│   ├── document.py            # 多格式文档读取
│   ├── hybrid_reader.py       # 轻量混合 PDF 读取（逐块乱码检测 + OCR）
│   ├── chunker.py             # 文本分块（5 种策略，参数可配置）
│   ├── cleaner.py             # 文本清洗 + OCR 空白清理
│   ├── plumber.py             # pdfplumber 文本提取
│   ├── ocr.py                 # EasyOCR 图片识别
│   ├── marker_reader.py       # Marker AI 深度解析
│   ├── quality_scorer.py      # 动态质量评分 v2（统计 + OCR + jieba）
│   ├── ingestion.py           # 入库流水线
│   ├── retriever.py           # 检索引擎（Dense + BM25 → RRF → 3D → 精排）
│   ├── reranker.py            # Cross-Encoder 精排器（log 长度归一化）
│   ├── llm_service.py         # LLM 服务（改写/自查询/问答）
│   ├── logger.py              # 结构化日志（JSON / 纯文本）
│   └── vector_store/
│       ├── base.py            # 抽象基类（模板方法 + 工厂）
│       ├── faiss_store.py     # Flat 精确索引
│       ├── ivf_store.py       # IVF 倒排索引
│       └── hnsw_store.py      # HNSW 图索引
├── tests/
│   ├── test_core_regressions.py   # 核心回归测试
│   └── test_evaluation_contracts.py # 评测规则契约测试
└── data/                      # 测试数据目录
```

## 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU + CUDA（可选，CPU 也能跑）

### 安装

```bash
# 运行依赖
pip install -r requirements.txt

# 开发 / 测试依赖
pip install -r requirements-dev.txt

# 可选重依赖（PaddleOCR / Marker）
pip install -r requirements-optional.txt
```

### 启动

```bash
python main.py
```

## 评测

本项目的评测分三层：

```bash
# 校验评测集结构、文档哈希和页码锚点覆盖
python eval/validate_dataset.py

# 快速验证带页码锚点的 PDF 解析样本
python eval/run_extraction_checks.py --quick

# 检索基线，可按 ID 查看命中的证据 chunk
python eval/run_baseline.py --ids ret_evac_003 --show-matches --show-top 5 --only-query-docs --no-reranker --no-ocr-fallback
```

检索评测支持两种证据策略：

- `same_chunk`：默认策略，要求同一个 chunk 同时包含证据组中的所有关键词，适合数值事实、定义、引用类问题。
- `multi_chunk_same_doc`：允许同一正确文档内多个 chunk 合并覆盖证据，适合列表型问题、表格页和图纸页。

评测 PDF 不提交到仓库，需要将 `eval/document_manifest.json` 中列出的文件放入 `data/`，并保持文件哈希一致。

## CLI 命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `/add <路径> [方法] [ocr\|marker]` | 导入文件 | `/add data/doc.pdf` |
| `/search <查询> [top_k]` | 语义搜索 + Dense/BM25 混合 | `/search 机器学习 5` |
| `/search_jieba <查询> [top_k]` | 分词后搜索 | `/search_jieba 编程 5` |
| `/ask <问题>` | LLM 问答 | `/ask Python是什么` |
| `/rewrite <查询>` | LLM 查询改写 | `/rewrite 有没有AI的资料` |
| `/switch <类型>` | 切换索引（flat/ivf/hnsw） | `/switch hnsw` |
| `/delete <文档名>` | 标记删除文档 | `/delete sample.txt` |
| `/list` | 查看文档列表 | `/list` |
| `/clear` | 清空所有数据 | `/clear` |
| `/count` | 查看总数 | `/count` |
| `/save <路径>` | 保存会话 | `/save data/my_db` |
| `/load <路径>` | 加载会话 | `/load data/my_db` |
| `/help` | 帮助 | `/help` |
| `/exit` | 退出（自动保存） | `/exit` |

## 检索流程

```
查询 → LLM 改写 → 编码
                     │
              Dense 召回 + BM25 召回
                     │
                RRF 融合排序（k=60）
                     │
               Route B 文档聚合（几何衰减 λ=0.2）
                     │
              三维度加权（α=0.7 β=0.1 γ=0.2）
                     │
            Cross-Encoder 精排（log 长度归一化）
                     │
               阈值过滤（默认 0.3）
                     │
               返回 top_k 结果
```

## PDF 解析策略

| 模式 | 命令 | 适用场景 | 质量 | 速度 |
|------|------|----------|------|------|
| 轻量混合 | `/add doc.pdf` | 普通 PDF（含子集字体） | ✅ | ⚡ 快 |
| 纯 OCR | `/add doc.pdf ocr` | 扫描件、图片 PDF | ✅ | 🐢 慢 |
| Marker | `/add doc.pdf marker` | 复杂排版、学术论文 | ✅✅ | 🐢 需 GPU |

轻量混合模式逐块检测四级乱码（替换字符/连续重复/罕用汉字/控制字符+标点），只有在检测到乱码的行才使用 OCR 替换，避免 OCR 破坏正常文本。

## 配置

所有参数集中在 `config/default.yaml`，支持环境变量覆写：

```yaml
# 示例：chunk 参数
chunking:
  sentence:
    max_sentences: 8
    overlap: 1

# 示例：检索权重
retrieval:
  alpha: 0.7       # chunk 相关性
  beta: 0.1        # 文档排名
  gamma: 0.2       # 质量分
  rrf_k: 60
  recall_expand: 10
```

环境变量覆写：`export RETRIEVAL_ALPHA=0.8`

## 学习参考

本项目在实现过程中参考了以下开源项目的设计思路：

- **RAGFlow DeepDoc**：文档视觉分析（OCR + 布局 + TSR）、乱码检测、XGBoost 文本合并
- **LangChain / LlamaIndex**：chunk 参数标准（400-512 tokens，12.5-20% 重叠）
- **Pinecone 推荐**：chunk_size ≈ 512 tokens

## 后续计划

- [x] Dense + BM25 混合检索 + RRF 融合
- [x] Cross-Encoder 精排 + log 长度归一化
- [x] SQLite 自动持久化 + 向量 BLOB 存储
- [x] 动态质量评分 v2（统计 + OCR + jieba）
- [x] YAML 配置中心 + 环境变量覆写
- [x] 结构化日志（JSON + 文件输出）
- [x] 评估框架（解析检查 / 检索基线 / QA 黄金集）
- [x] 轻量混合 PDF 读取（逐块乱码检测 + OCR 替换）
- [ ] FastAPI Web 接口
- [ ] 知识库治理与监控
- [ ] 布局感知分块（标题/表格/正文分离）
- [ ] 增量索引与异步入库
