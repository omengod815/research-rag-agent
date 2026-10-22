from app.services.embedding_cache import (
    clear_embedding_cache,
    get_cached_embedding,
    set_cached_embedding,
)


def test_embedding_cache_round_trip():
    clear_embedding_cache()
    key = "unit-test-key"
    vector = [0.1, 0.2, 0.3]
    assert get_cached_embedding(key) is None
    set_cached_embedding(key, vector)
    assert get_cached_embedding(key) == vector


def test_embedding_cache_returns_copy():
    clear_embedding_cache()
    set_cached_embedding("copy-key", [1.0, 2.0])
    received = get_cached_embedding("copy-key")
    received[0] = 999.0
    assert get_cached_embedding("copy-key") == [1.0, 2.0]