import pytest
import asyncio

from pipe import Pipe

@pytest.mark.asyncio
async def test_split_text_chunks_simple():
    p = Pipe()
    text = "A" * 2500
    chunks = p._split_text_chunks(text, chunk_size=1000, overlap=200)
    # Should produce 3 chunks: 0-999, 800-1799, 1600-2499
    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    assert len(chunks[2]) == 900

@pytest.mark.asyncio
async def test_get_embeddings_batch_respects_cache():
    p = Pipe()
    texts = ["one", "two", "three"]

    # Pre-seed cache for the second item
    p.embedding_cache.set("two", [0.1, 0.2, 0.3])

    results = await p._get_embeddings_batch(texts, batch_size=2, async_batch=False)

    # We should at least get cached embedding for 'two'
    assert results[1] == [0.1, 0.2, 0.3]
    # For this environment we don't have network to fetch others, so they may be None
    assert isinstance(results, list) and len(results) == 3
