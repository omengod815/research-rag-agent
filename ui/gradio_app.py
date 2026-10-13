import json
import uuid
from pathlib import Path

import httpx
import gradio as gr

API_BASE = "http://127.0.0.1:8000"


# ================= Tab1: PDF 上传 =================
def upload_pdf(file_path: str):
    if not file_path:
        return "请先选择 PDF。", {}

    path = Path(file_path)
    try:
        with path.open("rb") as f:
            response = httpx.post(
                f"{API_BASE}/documents/upload",
                files={"file": (path.name, f, "application/pdf")},
                timeout=900.0,
            )
        response.raise_for_status()
        data = response.json()

        status = (
            f"✅ 已完成索引：{data.get('filename', path.name)}\n\n"
            f"**document_id：** {data.get('document_id')}\n\n"
            f"**pages：** {data.get('pages', '未知')}\n\n"
            f"**chunks：** {data.get('chunks', '未知')}\n\n"
            f"**indexed_chunks：** {data.get('indexed_chunks', '未知')}"
        )
        return status, data

    except httpx.ReadTimeout:
        return "⏳ PDF 索引等待时间过长，请查看后端终端进度，不要立即重复上传。", {}
    except httpx.ConnectError:
        return "❌ 无法连接 FastAPI 后端，请确认 uvicorn 已启动。", {}
    except httpx.HTTPStatusError as exc:
        return f"❌ 上传失败：HTTP {exc.response.status_code}\n\n{exc.response.text}", {}
    except Exception as exc:
        return f"❌ 上传发生异常：{exc}", {}


# ================= Tab2: RAG 问答 =================
def ask_rag(question: str):
    if not question or not question.strip():
        return "请输入问题。", ""

    try:
        response = httpx.post(
            f"{API_BASE}/rag/query",
            json={"question": question.strip(), "top_k": 5},
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "没有返回答案。")
        sources = data.get("sources", [])
        evidence = []

        for i, item in enumerate(sources, start=1):
            text = item.get("text", "").replace("\r", " ").replace("\n", " ").strip()
            vector_score = float(item.get("score", 0.0))
            rerank_score = item.get("rerank_score")
            source = item.get("source", "未知来源")
            page = item.get("page", "未知页码")

            score_text = f"**向量相似度：** {vector_score:.4f}"
            if rerank_score is not None:
                score_text += f" · **Rerank 分数：** {float(rerank_score):.4f}"

            evidence.append(
                f"### Chunk {i}\n\n{score_text}  \n"
                f"**来源：** {source}  \n**页码：** {page}\n\n> {text[:800]}"
            )

        evidence_text = "\n\n---\n\n".join(evidence) or "⚠️ 后端没有返回检索证据。"
        return answer, evidence_text

    except httpx.ReadTimeout:
        return "⏳ RAG 请求超时，可能是 LLM API 响应较慢。", ""
    except httpx.ConnectError:
        return "❌ 无法连接 FastAPI 后端，请确认 uvicorn 已启动。", ""
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        return f"❌ RAG 请求失败\n\nHTTP {exc.response.status_code}\n\n{detail}", ""
    except Exception as exc:
        return f"❌ RAG 问答发生异常：\n\n{type(exc).__name__}: {exc}", ""


# ================= Tab3: Agent HITL（核心修复） =================
def _parse_sse_lines(line_iterator):
    """
    纯原生 SSE 解析器，替代 sseclient-py
    兼容 httpx.iter_lines() 输出，零外部依赖
    """
    event_type = None
    data_buffer = []

    for raw_line in line_iterator:
        # httpx iter_lines 已去除换行符，但防御性处理
        line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8").strip()

        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_buffer.append(line[len("data:"):].strip())
        elif line == "":
            # 空行 = 事件边界
            if data_buffer:
                yield event_type, "\n".join(data_buffer)
                event_type = None
                data_buffer = []

    # 处理末尾无空行的情况
    if data_buffer:
        yield event_type, "\n".join(data_buffer)


def _stream_agent(question, filename, thread_id, resume=None):
    """Agent SSE 流式通信（纯 httpx 原生实现）"""
    payload = {
        "question": question,
        "filename": filename,
        "thread_id": thread_id,
        "resume_decision": resume,
    }

    log = "🚀 **执行日志：**\n"
    answer = ""
    interrupt_info = None

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/agent/stream",
            json=payload,
            timeout=300.0,
        ) as resp:
            resp.raise_for_status()

            for event_type, data_str in _parse_sse_lines(resp.iter_lines()):
                try:
                    msg = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", event_type)

                if msg_type == "update":
                    node = list(msg["data"].keys())[0] if msg.get("data") else "unknown"
                    log += f"✅ `{node}` 完成\n"
                    if "report" in msg.get("data", {}):
                        answer = msg["data"]["report"].get("answer", "")

                elif msg_type == "interrupt":
                    log += f"⏸️ **{msg['data'].get('message', '等待审批')}**\n"
                    interrupt_info = msg["data"]
                    break  # 中断时停止消费流

                elif msg_type == "end":
                    log += "🏁 流程结束\n"

                elif msg_type == "error":
                    log += f"❌ {msg.get('message', '未知错误')}\n"

    except httpx.ConnectError:
        log += "❌ 无法连接后端，请确认 uvicorn 已启动\n"
    except httpx.ReadTimeout:
        log += "⏳ 请求超时，Agent 执行时间过长\n"
    except Exception as e:
        log += f"❌ 通信异常: {type(e).__name__}: {e}\n"

    return log, answer, interrupt_info


def on_agent_submit(q, f, st):
    tid = st.get("thread_id") or str(uuid.uuid4())
    log, ans, intr = _stream_agent(q, f, tid)
    if intr:
        return log, ans, gr.update(visible=True), gr.update(visible=False), {"thread_id": tid, "pending": True}
    return log, ans, gr.update(visible=False), gr.update(visible=True), {"thread_id": tid, "pending": False}


def on_agent_approve(q, f, st):
    log, ans, _ = _stream_agent(q, f, st["thread_id"], {"approved": True})
    return log, ans, gr.update(visible=False), gr.update(visible=True), {"thread_id": st["thread_id"], "pending": False}


def on_agent_reject(q, f, st):
    log, ans, _ = _stream_agent(q, f, st["thread_id"], {"approved": False})
    return log, ans, gr.update(visible=False), gr.update(visible=True), {"thread_id": st["thread_id"], "pending": False}


# ================= Gradio UI =================
with gr.Blocks(title="Research RAG Agent") as demo:
    gr.Markdown("# Research RAG Agent\n上传论文，建立知识库，验证 RAG 与 Agent 数据分析能力。")

    with gr.Tab("1. 上传论文"):
        pdf = gr.File(label="选择 PDF", file_types=[".pdf"], type="filepath")
        upload_btn = gr.Button("上传并建立知识库", variant="primary")
        upload_status = gr.Markdown()
        upload_json = gr.JSON(label="后端返回")
        upload_btn.click(fn=upload_pdf, inputs=pdf, outputs=[upload_status, upload_json])

    with gr.Tab("2. RAG 问答"):
        question = gr.Textbox(label="问题", placeholder="例如：论文形成了怎样的完整技术路线？", lines=3)
        ask_btn = gr.Button("发送", variant="primary")
        answer = gr.Markdown(label="回答")
        evidence = gr.Markdown(label="检索证据")
        ask_btn.click(fn=ask_rag, inputs=question, outputs=[answer, evidence])

    with gr.Tab("3. Agent 数据分析 (HITL)"):
        gr.Markdown("支持流式进度展示 + 敏感数据操作人工审批")
        agent_state = gr.State({"thread_id": "", "pending": False})

        with gr.Row():
            with gr.Column(scale=1):
                agent_q = gr.Textbox(label="分析问题", placeholder="例如：分析 regression.csv 的拟合误差")
                agent_f = gr.Textbox(label="CSV 文件名", placeholder="regression.csv")
                agent_submit = gr.Button("🚀 提交分析", variant="primary", visible=True)

                with gr.Row(visible=False) as approval_row:
                    approve_btn = gr.Button("✅ 同意执行", variant="primary")
                    reject_btn = gr.Button("❌ 拒绝执行", variant="stop")

            with gr.Column(scale=2):
                agent_log = gr.Markdown(label="执行进度")
                agent_ans = gr.Markdown(label="分析报告")

        agent_submit.click(on_agent_submit, [agent_q, agent_f, agent_state],
                           [agent_log, agent_ans, approval_row, agent_submit, agent_state])
        approve_btn.click(on_agent_approve, [agent_q, agent_f, agent_state],
                          [agent_log, agent_ans, approval_row, agent_submit, agent_state])
        reject_btn.click(on_agent_reject, [agent_q, agent_f, agent_state],
                         [agent_log, agent_ans, approval_row, agent_submit, agent_state])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)