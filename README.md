# Chat_with_Datawhale_langchain · 个人知识库助手（中转 API 改造版）

基于 LangChain 的本地知识库 RAG 问答应用。原项目来自 [logan-zou/Chat_with_Datawhale_langchain](https://github.com/logan-zou/Chat_with_Datawhale_langchain)（Datawhale [llm-universe](https://github.com/datawhalechina/llm-universe) 教程实战项目），本版本在保留原架构分层的基础上做了现代化改造。

## 功能特性

- **统一 OpenAI 兼容通道**：所有大模型（gpt / glm / deepseek 等）通过一套协议调用，支持官方 API 或任意中转平台（one-api / new-api 等）
- **动态模型列表**：启动时自动从中转 `/models` 端点拉取可用模型，界面下拉框直接选择，另有一键刷新按钮，无需在代码中维护模型名单
- **两种 Embedding 通道**：
  - `m3e`：本地推理（HuggingFace moka-ai/m3e-base），免费、无网络依赖、不受 API 限流影响
  - `api`：调用自部署的 Embedding 服务（OpenAI 兼容 `/embeddings` 协议，如 TEI / Xinference / Ollama 部署的 BGE-M3）
- **三种问答方式**：知识库检索问答（带 / 不带对话历史）+ 纯大模型直聊
- **Gradio Web 界面 + FastAPI 接口**双服务
- **路径锚定**：向量库等数据路径固定在项目根目录，从任何目录启动行为一致
- 向量化完成后界面显示结果反馈（源文件数 / 文本块数 / 存储位置）

## 项目结构

```
├── llm/                        大模型调用层
│   └── call_llm.py               统一入口：对话、/models 拉取、Key 解析
├── embedding/                  向量化层
│   ├── call_embedding.py         工厂：按名称返回 Embedding 实例
│   └── api_embedding.py          自部署 Embedding 客户端（OpenAI 兼容）
├── database/                   数据层
│   ├── create_db.py              语料加载 → 文本切分 → 向量化 → Chroma 持久化
│   └── readme_db/                内置语料：Datawhale 90+ 开源仓库的 README
├── qa_chain/                   应用层
│   ├── QA_chain_self.py          检索问答链（不带历史）
│   ├── Chat_QA_chain_self.py     检索问答链（带历史）
│   ├── model_to_llm.py           模型名 → LangChain LLM 实例
│   └── get_vectordb.py           向量库智能加载（已有则加载，没有则自动建）
├── serve/                      服务层
│   ├── run_gradio.py             Web 界面（主入口）
│   └── api.py                    FastAPI 接口
├── vector_db/                  向量库数据（运行时生成，已 gitignore）
├── figures/                    界面图片资源
├── .env                        密钥与服务配置（不入库！）
└── requirements.txt            Python 依赖
```

## 工作原理

标准 RAG 流程：用户提问 → 问题向量化 → 在向量库中检索最相似的 top-k 文本块 → 检索结果与问题拼接为 Prompt → 大模型生成回答。

![RAG 流程](./figures/rag.png)

**建库（索引）阶段**：

```
知识文件（md/pdf/txt）
  → 加载器（UnstructuredMarkdownLoader / PyMuPDFLoader / UnstructuredFileLoader）
  → 文本切分 RecursiveCharacterTextSplitter（chunk_size=500，chunk_overlap=150）
  → Embedding 向量化（m3e 本地推理 或 远程服务）
  → Chroma 向量库持久化到 <项目根>/vector_db/chroma/
```

## 环境准备

| 依赖 | 说明 |
|---|---|
| Python 3.9 | 推荐 conda 独立环境 |
| Microsoft C++ Build Tools | **仅 Windows 需要**：编译 chromadb 的依赖 hnswlib（勾选「使用 C++ 的桌面开发」） |
| torch + sentence-transformers | 仅使用 m3e 本地向量化时需要（见下） |

```powershell
# 1. 创建环境（Windows 示例）
conda create -n rag_agent python=3.9.0
conda activate rag_agent

# 2. 安装依赖（仓库根目录）
pip install -r requirements.txt

# 3.（可选）m3e 本地向量化所需
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
# 国内下载 HuggingFace 模型较慢时：$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 配置 .env

复制仓库中的 `.env` 模板并填写（该文件已被 gitignore，不会提交）：

```ini
# ===== 大模型（OpenAI 兼容通道，支持中转）=====
OPENAI_API_KEY = "sk-你的Key"
OPENAI_API_BASE = "https://你的中转地址/v1"    # 不填默认 OpenAI 官方

# ===== 自部署 Embedding 服务（界面选 api 时生效）=====
EMBEDDING_API_BASE = ""                        # 如 http://192.168.1.10:9997/v1
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_API_KEY = ""                         # 服务端有鉴权才填
```

自部署 Embedding 服务参考（任选其一，均为 OpenAI 兼容协议）：

| 部署方式 | 服务端命令 | EMBEDDING_API_BASE |
|---|---|---|
| Ollama | `ollama pull bge-m3` | `http://服务器:11434/v1` |
| Xinference | `xinference launch --model-name bge-m3 --model-type embedding` | `http://服务器:9997/v1` |
| TEI | `docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference --model-id BAAI/bge-m3` | `http://服务器:8080/v1` |

## 快速开始（Gradio 界面）

```powershell
cd serve
python run_gradio.py
# 浏览器打开 http://127.0.0.1:7860
```

界面操作顺序：

1. 右侧「模型选择」：large language model 从下拉框选一个（列表来自中转 `/models` 实时拉取）；Embedding model 选 `m3e` 或 `api`
2. 上传知识库目录（可直接选择本仓库的 `database/readme_db`，或任意包含 md/pdf/txt 的文件夹）
3. 点击【知识库文件向量化】，等待输入框出现"向量化完成：N 个源文件 → N 个文本块"提示
4. 左侧输入问题，**点击按钮提问**：
   - 【Chat db without history】知识库检索问答（推荐，每问消耗 1 次大模型调用）
   - 【Chat db with history】知识库检索 + 多轮记忆（每问消耗 2 次大模型调用）
   - 【Chat with llm】纯大模型直聊，**不查知识库**
5. 【Clear console】清空界面显示与服务端对话记忆，开始新会话
6. 【🔄 从 API 刷新模型列表】中转端新增模型后无需重启即可刷新下拉框

> ⚠️ **注意：输入框按回车 = 纯大模型直聊（不查知识库）**。要基于你的文档提问，必须点击带 "db" 字样的按钮。

## FastAPI 接口方式

```powershell
cd serve
python api.py        # 监听 http://127.0.0.1:8000
```

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/ -Method Post -ContentType "application/json" -Body (@{
    prompt = "介绍一下 joyrl 项目"
    model  = "glm-5.3-flash"
    embedding = "m3e"
} | ConvertTo-Json)
```

## 常见问题

| 问题 | 说明 |
|---|---|
| 提问答不上文档内容 | 确认点的是【Chat db】按钮而不是回车；确认向量化时输入框有成功提示 |
| 更换 Embedding 后检索错乱 | 不同模型的向量维度不同（m3e 768 维 / BGE-M3 1024 维），**必须删除 `vector_db/` 重新向量化** |
| 中转限流报错 | 多数中转有每分钟请求数限制；带历史的问答每次提问消耗 2 次调用 |
| 启动时模型列表为内置名单 | `/models` 拉取失败自动回退代码内置名单，检查 `.env` 中地址与 Key |
| Windows 装 hnswlib 报错缺编译器 | 安装 Microsoft C++ Build Tools（勾选「使用 C++ 的桌面开发」）后重试 |

## 与原版的主要差异

1. 大模型调用统一为 OpenAI 兼容通道（官方 / 中转），移除文心、讯飞星火、智谱旧版 SDK 封装
2. 模型列表由 `/models` 端点动态拉取，界面支持一键刷新
3. Embedding 通道改为 m3e（本地）+ 自部署服务（OpenAI 兼容），移除 openai 官方与智谱旧版通道
4. 修复依赖冲突与错误版本号（`requirements.txt` 可直接安装）
5. 数据路径锚定项目根目录，与启动目录无关；向量化增加结果反馈；向量化后自动刷新问答链缓存
6. 移除一次性语料生成脚本（README 语料已包含在 `database/readme_db/`）

## 致谢

- 原项目：[logan-zou/Chat_with_Datawhale_langchain](https://github.com/logan-zou/Chat_with_Datawhale_langchain)
- 教程：[datawhalechina/llm-universe](https://github.com/datawhalechina/llm-universe)《动手学大模型应用开发》
- 语料：[Datawhale](https://github.com/datawhalechina) 各开源项目 README
