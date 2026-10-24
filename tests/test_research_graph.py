import asyncio

from app.agents import research_graph as rg


def run_graph(monkeypatch, route: str, filename: str = "demo_predictions.csv"):
    async def fake_router(question: str) -> str:
        return route

    async def fake_retrieve(question: str, top_k: int = 5, user_id: str = "local-demo"):
        return [{"source": "paper_a.pdf", "page": 3, "text": "论文说明了实验方法。", "score": 0.9}]

    def fake_data(name: str) -> str:
        assert name == filename
        return "rows=5, MAE=0.620000, RMSE=0.730000, R2=0.980000"

    async def fake_report(question, evidence, analysis="", errors=None):
        return f"QUESTION={question}\nEVIDENCE={bool(evidence)}\nANALYSIS={bool(analysis)}\nERRORS={len(errors or [])}"

    async def fake_chat(prompt: str) -> str:
        return "这是一个普通问题的模拟回答。"

    monkeypatch.setattr(rg, "route_question", fake_router)
    monkeypatch.setattr(rg, "retrieve", fake_retrieve)
    monkeypatch.setattr(rg, "analyze_regression_csv_impl", fake_data)
    monkeypatch.setattr(rg, "build_report", fake_report)
    monkeypatch.setattr(rg, "chat_once", fake_chat)

    return asyncio.run(rg.research_graph.ainvoke({
        "question": "测试问题",
        "filename": filename,
    }))


def test_retrieval_route(monkeypatch):
    result = run_graph(monkeypatch, "retrieval")
    assert result["route"] == "retrieval"
    assert "论文说明了实验方法" in result["evidence"]
    assert result["answer"]


def test_data_route(monkeypatch):
    result = run_graph(monkeypatch, "data")
    assert result["route"] == "data"
    assert "MAE" in result["analysis"]
    assert result["answer"]


def test_combined_route(monkeypatch):
    result = run_graph(monkeypatch, "combined")
    assert result["route"] == "combined"
    assert "论文说明了实验方法" in result["evidence"]
    assert "MAE" in result["analysis"]


def test_general_route(monkeypatch):
    result = run_graph(monkeypatch, "general")
    assert result["route"] == "general"
    assert result["answer"] == "这是一个普通问题的模拟回答。"
