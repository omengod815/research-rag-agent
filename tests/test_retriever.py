import asyncio

from app.rag.retriever import retrieve


def test_retriever():
    query = "TiAl合金等温轧制过程中影响出口厚度的主要因素是什么？"

    results = asyncio.run(
        retrieve(query, top_k=5)
    )

    print("\n用户问题：", query)
    print("检索结果数量：", len(results))

    for i, item in enumerate(results, start=1):
        print(f"\n========== Top {i} ==========")
        print("相似度：", item["score"])
        print("来源：", item["source"])
        print("页码：", item["page"])
        print("内容：", item["text"][:500])

    assert len(results) > 0

    for item in results:
        assert item["text"]
        assert item["source"]
        assert item["page"] is not None