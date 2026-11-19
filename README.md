# SimpleRAG — 模型与 RAG 工作台

面向单用户的 FastAPI Web 原型：OpenAI 兼容模型连通检查与聊天、BGE 向量调用、MinerU/MarkItDown 文档解析与批量转换、Milvus 向量检索、MinIO 原件存储、SQLite Token 用量账本。

> 当前为测试阶段版本：`v0.1.0`。

## 功能概览

| 页面 | 说明 |
| --- | --- |
| 总览 | 选择并保存默认 chat / embedding 模型；模型与 Milvus / MinIO / MinerU 连通状态；用量统计 |
| 知识库 | 上传文档 → 保存原件 → 切分 → embedding → 写 Milvus，六阶段进度可见；可重建索引、删除文档 |
| 文档转换 | MinerU 批量转换，产出统一 JSON（`schema: rag.document`）+ Markdown；转换后可「继续切分入库」或「下载 ZIP」 |
| 问答 | 可选启用 RAG 检索增强，回答标注来源 |
| Token 用量 | 按模型统计调用次数与 Token，最近调用明细 |
| 存储设置 | 在线修改 MinIO / Milvus 的 IP 与端口，一键跳转各自管理界面 |

## 存储设置与自动建库

「存储设置」页可修改 Milvus（gRPC，默认 19530）与 MinIO（S3 API，默认 9000）的地址和端口，保存后立即生效并写入本地 SQLite（`app_settings` 表），**不会改写 `.env`**；点「恢复 .env 默认」可清除覆盖值。页面同时提供跳转按钮：

- Milvus 管理界面：`http://<host>:9091`
- MinIO 管理界面：`http://<host>:9001`

容错行为：

- `MINIO_BUCKET` 指向的 bucket 不存在时自动创建；
- `MILVUS_DATABASE` 指向的数据库不存在时自动创建（Milvus 数据库名仅允许字母、数字、下划线，非法字符会被替换为 `_` 并记录告警）；
- collection 不存在时按实际 embedding 维度创建 schema、AUTOINDEX 索引并加载；
- 地址不可达时「测试连接」在数秒内返回错误，不会长时间挂起。

## 配置

```bash
cp .env.example .env      # 然后填写 AI_API_KEY 与各服务地址
```

`.env` 含密钥，**不要提交到任何仓库或外部服务**；仓库只保留 `.env.example`。

## 启动

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)：

```bash
uv venv
uv pip install -r requirements.txt
.venv/bin/python app.py
```

打开 `http://127.0.0.1:8000`（地址与端口由 `.env` 的 `HOST` / `PORT` 决定），API 文档在 `/docs`。

## 开发说明

模块在代码中统一以 `agent_test.rag.*` 导入。本地跑脚本或测试时，用一个 `PYTHONPATH` 目录放置指向本仓库的软链即可，不要改写导入语句：

```bash
mkdir -p /tmp/rag_shim/agent_test
touch /tmp/rag_shim/agent_test/__init__.py
ln -s "$PWD" /tmp/rag_shim/agent_test/rag
PYTHONPATH=/tmp/rag_shim .venv/bin/python -c "import agent_test.rag.app"
```

注意不要加 `-I`（隔离模式会忽略 `PYTHONPATH`）。

## 边界

- Token 以 API 返回的 usage 为准；没有 usage 时标记估算/缺失。
- PDF/图片走 MinerU，未配置或不可达时回退 MarkItDown（PDF 提取质量较弱）。
- RAG 会拒绝用与已有向量不匹配的 embedding 模型查询；换模型需重建索引。
- 转换任务在进程内线程池执行，重启时未完成的任务会被标记为失败/可重试。
- 请求日志只记录路径、状态码与耗时；不记录密钥、提示词正文或文档内容。
