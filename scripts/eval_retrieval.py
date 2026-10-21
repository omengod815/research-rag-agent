import asyncio
import json
import os
import sys
from pathlib import Path

# 把项目根目录加到 Python 路径，这样在 scripts/ 下运行也能找到 app 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retriever import retrieve

EVAL_FILE = Path("tests/eval_questions.jsonl")
TOP_K = 5
USER_ID = os.getenv("EVAL_USER_ID", "local-demo")
MODE = os.getenv("RETRIEVAL_MODE", "hybrid")


def load_cases():
    cases = []
    for line in EVAL_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


async def main():
    cases = load_cases()
    if not cases:
        raise SystemExit("评测集为空：请先在 tests/eval_questions.jsonl 中添加问题。")

    hits_at_k = 0
    reciprocal_ranks = []
    for case in cases:
        results = await retrieve(case["question"], top_k=TOP_K, user_id=USER_ID, mode=MODE)
        expected_source = Path(case["expected_source"]).name.casefold()
        expected_page = case.get("expected_page")
        rank_found = None

        for rank, result in enumerate(results, start=1):
            actual_source = Path(str(result.get("source", ""))).name.casefold()
            page_matches = expected_page is None or result.get("page") == expected_page
            if actual_source == expected_source and page_matches:
                rank_found = rank
                break

        if rank_found is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank_found)
        else:
            reciprocal_ranks.append(0.0)

        print(f"问题：{case['question']}")
        print(f"命中排名：{rank_found or '未命中'}")
        print("Top-K：", [(x.get("source"), x.get("page")) for x in results])
        print("-" * 60)

    count = len(cases)
    print(f"Mode={MODE}; K={TOP_K}; 样本数={count}")
    print(f"Hit@{TOP_K} = {hits_at_k / count:.4f}")
    print(f"MRR = {sum(reciprocal_ranks) / count:.4f}")


if __name__ == "__main__":
    asyncio.run(main())