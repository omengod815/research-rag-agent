import jieba


def keyword_overlap_rerank(
    query: str,
    hits: list[dict],
) -> list[dict]:
    query_terms = {
        word.strip()
        for word in jieba.lcut(query.lower())
        if word.strip()
    }

    rescored = []

    for hit in hits:
        text_terms = {
            word.strip()
            for word in jieba.lcut(
                hit["text"].lower()
            )
            if word.strip()
        }

        overlap = len(
            query_terms & text_terms
        )

        item = dict(hit)

        item["rerank_score"] = (
            item["score"]
            + 0.03 * overlap
        )

        rescored.append(item)

    return sorted(
        rescored,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )