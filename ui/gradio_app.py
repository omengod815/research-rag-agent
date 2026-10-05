from pathlib import Path

import httpx
import gradio as gr


API_BASE = "http://127.0.0.1:8000"


def upload_pdf(file_path: str):
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

                # 大论文索引可能需要几分钟
                timeout=900.0,
            )

        response.raise_for_status()

        data = response.json()

        status = (
            f"✅ 已完成索引："
            f"{data.get('filename', path.name)}\n\n"
            f"**document_id：** "
            f"{data.get('document_id')}\n\n"
            f"**pages：** "
            f"{data.get('pages', '未知')}\n\n"
            f"**chunks：** "
            f"{data.get('chunks', '未知')}\n\n"
            f"**indexed_chunks：** "
            f"{data.get('indexed_chunks', '未知')}"
        )

        return status, data

    except httpx.ReadTimeout:
        return (
            "⏳ PDF 索引等待时间过长。\n\n"
            "后端可能仍在执行 Embedding 和 Qdrant 写入。"
            "请查看 FastAPI 终端中的 `Indexing:` 进度，"
            "不要立即重复上传。",
            {},
        )

    except httpx.ConnectError:
        return (
            "❌ 无法连接 FastAPI 后端。\n\n"
            "请确认已经运行：\n\n"
            "`uvicorn app.main:app --reload`",
            {},
        )

    except httpx.HTTPStatusError as exc:
        return (
            f"❌ 上传失败：HTTP "
            f"{exc.response.status_code}\n\n"
            f"{exc.response.text}",
            {},
        )

    except Exception as exc:
        return (
            f"❌ 上传发生异常：{exc}",
            {},
        )


def ask_rag(question: str):
    if not question or not question.strip():
        return "请输入问题。", ""

    try:
        response = httpx.post(
            f"{API_BASE}/rag/query",
            json={
                "question": question.strip(),
                "top_k": 5,
            },

            # 给 Retriever + Rerank + Qwen 足够时间
            timeout=300.0,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "answer",
            "没有返回答案。",
        )

        sources = data.get(
            "sources",
            [],
        )

        evidence = []

        for i, item in enumerate(
            sources,
            start=1,
        ):
            text = (
                item.get("text", "")
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )

            vector_score = float(
                item.get("score", 0.0)
            )

            rerank_score = item.get(
                "rerank_score"
            )

            source = item.get(
                "source",
                "未知来源",
            )

            page = item.get(
                "page",
                "未知页码",
            )

            score_text = (
                f"**向量相似度：** "
                f"{vector_score:.4f}"
            )

            if rerank_score is not None:
                score_text += (
                    f" · **Rerank 分数：** "
                    f"{float(rerank_score):.4f}"
                )

            evidence.append(
                f"### Chunk {i}\n\n"
                f"{score_text}  \n"
                f"**来源：** {source}  \n"
                f"**页码：** {page}\n\n"
                f"> {text[:800]}"
            )

        evidence_text = (
            "\n\n---\n\n".join(evidence)
        )

        if not evidence_text:
            evidence_text = (
                "⚠️ 后端没有返回检索证据。"
            )

        return (
            answer,
            evidence_text,
        )

    except httpx.ReadTimeout:
        return (
            "⏳ RAG 请求超时。\n\n"
            "可能是 Qwen API 当前响应较慢或网络连接异常。"
            "请查看 FastAPI 终端是否出现 "
            "`APITimeoutError` 或 `ConnectTimeout`。",
            "",
        )

    except httpx.ConnectError:
        return (
            "❌ 无法连接 FastAPI 后端。\n\n"
            "请确认 `uvicorn app.main:app --reload` "
            "正在运行。",
            "",
        )

    except httpx.HTTPStatusError as exc:
        try:
            error_data = exc.response.json()

            detail = error_data.get(
                "detail",
                exc.response.text,
            )

        except Exception:
            detail = exc.response.text

        return (
            f"❌ RAG 请求失败\n\n"
            f"HTTP {exc.response.status_code}\n\n"
            f"{detail}",
            "",
        )

    except Exception as exc:
        return (
            f"❌ RAG 问答发生异常：\n\n"
            f"{type(exc).__name__}: {exc}",
            "",
        )


with gr.Blocks(
    title="Research RAG Agent"
) as demo:

    gr.Markdown(
        """
# Research RAG Agent

上传论文，建立知识库，然后从用户视角验证 RAG。

**PDF → Chunk → Embedding → Qdrant → Retriever → Rerank → Qwen → Answer**
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
                "例如：从有限元建模到工业化验证，"
                "论文形成了怎样的完整技术路线？"
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