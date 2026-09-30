import asyncio

from app.services.embedding_service import embed_texts


def test_embedding_api():
    vectors = asyncio.run(
        embed_texts(["深度学习温度预测"])
    )

    print("向量数量:", len(vectors))
    print("向量维度:", len(vectors[0]))
    print("前5个值:", vectors[0][:5])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024