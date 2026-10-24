# Research RAG Agent

基于科研论文知识库的 RAG 与多节点 Agent 工作流。

---

## 功能

- **PDF 上传与索引**：自动解析 PDF、分块、向量化、存入 Qdrant
- **混合检索**：Dense 向量 + BM25 关键词 + RRF 融合 + 轻量重排
- **引用证据**：每条回答附带来源页码和原文片段
- **受控 CSV 指标分析**：MAE / RMSE / R² 自动计算
- **LangGraph 路由**：自动判断问题类型（检索 / 数据分析 / 汇总报告 / 普通对话）
- **人工审批（HITL）**：敏感操作前暂停等待用户确认，可恢复
- **节点 Streaming**：实时展示每个节点执行进度

---

## 架构

```
用户浏览器
    ↓
Gradio 前端 (7860)
    ↓ HTTP
FastAPI 后端 (8000)
    ↓
LangGraph 工作流
    ├─ Router 节点（问题分类）
    ├─ Retrieval 节点（混合检索）
    ├─ Data 节点（CSV 分析）
    ├─ Approval 节点（人工审批）
    └─ Report 节点（生成回答）
    ↓
Qdrant 向量库 (6333)        Qwen LLM API（外部）
```

---

## 环境要求

- Python 3.13+
- Docker Desktop（可选，用于容器化部署）
- Qwen API Key（用于 Embedding 和 LLM）

> 免费额度、模型名称和 API Endpoint 由用户自己的云服务控制台决定。

---

## 本地启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 3. 启动后端
uvicorn app.main:app --reload

# 4. 启动前端（另开一个终端）
python ui/gradio_app.py
```

---

## Docker Compose 启动

```bash
docker compose up --build
```

启动后：
- API 文档：http://127.0.0.1:8000/docs
- Gradio 前端：http://127.0.0.1:7860
- Qdrant 控制台：http://127.0.0.1:6333/dashboard

---

## 测试

```bash
# 单元测试（不需要 API Key，快）
python -m pytest -m "not integration" -q

# 集成测试（需要真实 API Key，慢）
python -m pytest -m integration -s
```

---

## 检索评测结果

| 日期 | 检索方式 | Chunk 参数 (size/overlap) | Hit@5 | MRR | 测试集大小 |
|---|---|---|---|---|---|
| 2026-10-21 | Dense 向量检索 | 900 / 150 | 0.50 | 0.30 | 8 题 |
| 2026-10-21 | Hybrid (Dense + BM25 + RRF + Rerank) | 900 / 150 | 0.375 | 0.25 | 8 题 |

### 已知局限

- 单篇论文下 Hybrid 增益不明显，BM25 可能把 Dense 好结果挤下去
- 评测集只有 8 道题，样本量小，结果波动大
- 重排使用轻量关键词重叠 baseline，Cross-Encoder 未默认启用
- 结论类、综述类问题的检索效果有待优化

---

## 安全边界

⚠️ **本项目是作品集 Demo，不是生产级系统。**

尚未提供：
- 生产级用户登录与身份认证
- 细粒度授权与权限控制
- API 限流与防滥用
- 完整操作审计日志

**注意事项：**
- API Key 不要提交到 Git（.env 已在 .gitignore 中）
- 用户隔离是简单的 user_id 过滤，不是真正的身份认证
- 上传的文件仅用于演示，不要放敏感数据
