from collections import OrderedDict
from threading import RLock

MAX_ENTRIES = 512
_cache: OrderedDict[str, list[float]] = OrderedDict()
_lock = RLock()


def get_cached_embedding(key: str) -> list[float] | None:
    with _lock:
        value = _cache.get(key)
        if value is None:
            return None
        _cache.move_to_end(key)
        return list(value)


def set_cached_embedding(key: str, vector: list[float]) -> None:
    with _lock:
        _cache[key] = list(vector)
        _cache.move_to_end(key)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)


def clear_embedding_cache() -> None:
    """仅供测试或手动排查使用。"""
    with _lock:
        _cache.clear()