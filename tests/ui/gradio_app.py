from pathlib import Path

import gradio as gr
import httpx


API_BASE = "http://127.0.0.1:8000"


def upload_pdf(file_path: str):
    """
    上传 PDF，并调用 FastAPI 的 /documents/upload 接口完成：
    PDF 保存 -> 解析 -> Chunk -> Embedding -> Qdrant 入库
    """
    if not file_path:
        return "请先选择 PDF。", {}

    path = Path(file_path)

    try:
        with path.open("rb") as f:
            response = httpx.post(
                f"{API_BASE}/documents/upload",
                files={
                    "file": (
                        path.name,
                        f,
                        "application/pdf",
                    )
                },
                timeout=300.0,
            )

        response.raise_for_status()

        data = response.json()

        status = (
            f"✅ 已完成索引：{data.get('filename', path.name)}\n\n"
            f"**document_id：** {data.get('document_id')}\n\n"
            f"**pages：** {data.get('pages', '未知')}\n\n"
            f"**chunks：** "
            f"{data.get('chunks', data.get('chunk_count', '未知'))}\n\n"
            f"**indexed：** {data.get('indexed', False)}"
        )

        return status, data

    except httpx.ConnectError:
        return (
            "❌ 无法连接 FastAPI，请先启动：\n\n"
            "`uvicorn app.main:app --reload`",
            {},
        )

    except httpx.HTTPStatusError as exc:
        return (
            f"❌ 上传失败：HTTP {exc.response.status_code}\n\n"
            f"{exc.response.text}",
            {},
        )

    except Exception as exc:
        return f"❌ 上传发生异常：{exc}", {}


def ask_rag(question: str):
    """
    调用 /rag/query。
    后端完成：
    Query -> Retriever -> Rerank -> Top-K -> Qwen -> Answer
    """
    if not question or not question.strip():
        return "请输入问题。", ""

    try:
        response = httpx.post(
            f"{API_BASE}/rag/query",
            json={
                "question": question.strip(),
                "top_k": 5,
            },
            timeout=180.0,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("answer", "")
        sources = data.get("sources", [])

        evidence = []

        for i, item in enumerate(sources, start=1):
            text = (
                item.get("text", "")
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )

            vector_score = item.get("score", 0.0)

            # 如果后端已经用了 Rerank，就显示 rerank_score。
            # 如果暂时还没有，则退化成原始 score。
            rerank_score = item.get(
                "rerank_score",
                vector_score,
            )

            source = item.get(
                "source",
                "未知来源",
            )

            page = item.get(
                "page",
                "未知页码",
            )

            evidence.append(
                f"### Chunk {i}\n\n"
                f"**向量相似度：** {vector_score:.4f}  \n"
                f"**Rerank 分数：** {rerank_score:.4f}  \n"
                f"**来源：** {source}  \n"
                f"**页码：** {page}\n\n"
                f"> {text[:800]}"
            )

        evidence_text = "\n\n---\n\n".join(evidence)

        if not evidence_text:
            evidence_text = "没有返回检索证据。"

        return answer, evidence_text

    except httpx.ConnectError:
        return (
            "❌ 无法连接 FastAPI，请确认后端已经启动。",
            "",
        )

    except httpx.HTTPStatusError as exc:
        return (
            f"❌ RAG 请求失败：HTTP "
            f"{exc.response.status_code}\n\n"
            f"{exc.response.text}",
            "",
        )

    except Exception as exc:
        return (
            f"❌ RAG 问答发生异常：{exc}",
            "",
        )


with gr.Blocks(
    title="Research RAG Agent"
) as demo:

    gr.Markdown(
        """
# Research RAG Agent

上传科研论文建立知识库，然后进行 RAG 问答。

当前链路：

**PDF → Chunk → Qwen Embedding → Qdrant → Retriever → Rerank → Qwen → Answer**
"""
    )

    with gr.Tab("1. 上传论文"):

        pdf = gr.File(
            label="选择 PDF",
            file_types=[".pdf"],
            type="filepath",
        )

        upload_btn = gr.Button(
            "上传并建立知识库",
            variant="primary",
        )

        upload_status = gr.Markdown()

        upload_json = gr.JSON(
            label="后端返回"
        )

        upload_btn.click(
            fn=upload_pdf,
            inputs=pdf,
            outputs=[
                upload_status,
                upload_json,
            ],
        )

    with gr.Tab("2. RAG 问答"):

        question = gr.Textbox(
            label="问题",
            placeholder=(
                "例如：TiAl 合金等温轧制过程中"
                "影响出口厚度的主要因素是什么？"
            ),
            lines=3,
        )

        ask_btn = gr.Button(
            "发送",
            variant="primary",
        )

        answer = gr.Markdown(
            label="回答"
        )

        evidence = gr.Markdown(
            label="检索证据"
        )

        ask_btn.click(
            fn=ask_rag,
            inputs=question,
            outputs=[
                answer,
                evidence,
            ],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )