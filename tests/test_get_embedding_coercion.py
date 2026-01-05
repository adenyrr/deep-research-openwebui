import os
import sys
import pytest
import asyncio

# Ensure repository root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide lightweight stubs for optional heavy dependencies to allow unit tests that
# don't exercise network code to import the module without installing all extras.
import types
sys.modules.setdefault('aiohttp', types.ModuleType('aiohttp'))

# Provide a minimal pydantic stub so the module can be imported in environments
# without pydantic installed. This is safe for our tests which don't rely on
# full pydantic behavior.
_pyd = types.ModuleType('pydantic')
class _BaseModel:
    pass

def _Field(*args, **kwargs):
    return None

_pyd.BaseModel = _BaseModel
_pyd.Field = _Field
sys.modules.setdefault('pydantic', _pyd)

# Minimal sklearn stub for imports during testing
_sklearn = types.ModuleType('sklearn')
_sklearn.metrics = types.ModuleType('sklearn.metrics')
_sklearn.metrics.pairwise = types.ModuleType('sklearn.metrics.pairwise')

def _cosine_similarity(a, b):
    # Return 1.0 similarity for identical shapes to avoid errors in tests
    return [[1.0]]

_sklearn.metrics.pairwise.cosine_similarity = _cosine_similarity
_sklearn.decomposition = types.ModuleType('sklearn.decomposition')
class _PCA:
    def __init__(self, n_components=None):
        pass

_sklearn.decomposition.PCA = _PCA
_sklearn.cluster = types.ModuleType('sklearn.cluster')
class _KMeans:
    def __init__(self, n_clusters=None, random_state=None):
        pass

_sklearn.cluster.KMeans = _KMeans
sys.modules.setdefault('sklearn', _sklearn)
sys.modules.setdefault('sklearn.metrics', _sklearn.metrics)
sys.modules.setdefault('sklearn.metrics.pairwise', _sklearn.metrics.pairwise)
sys.modules.setdefault('sklearn.decomposition', _sklearn.decomposition)
sys.modules.setdefault('sklearn.cluster', _sklearn.cluster)

# Minimal stubs for open_webui package to allow importing pipe.py in tests
_open_webui = types.ModuleType('open_webui')
_open_webui.constants = types.ModuleType('open_webui.constants')
_open_webui.constants.TASKS = {}
_open_webui.main = types.ModuleType('open_webui.main')

async def _dummy_generate_chat_completions(*args, **kwargs):
    return {"choices": []}

_open_webui.main.generate_chat_completions = _dummy_generate_chat_completions
_open_webui.models = types.ModuleType('open_webui.models')
_open_webui.models.users = types.ModuleType('open_webui.models.users')

class _User:
    pass

_open_webui.models.users.User = _User

sys.modules.setdefault('open_webui', _open_webui)
sys.modules.setdefault('open_webui.constants', _open_webui.constants)
sys.modules.setdefault('open_webui.main', _open_webui.main)
sys.modules.setdefault('open_webui.models', _open_webui.models)
sys.modules.setdefault('open_webui.models.users', _open_webui.models.users)

from pipe import Pipe, EmbeddingCache


def test_get_embedding_with_empty_dict_returns_none():
    async def _run():
        p = Pipe()
        # Use a lightweight valves object to avoid requiring pydantic in test environment
        from types import SimpleNamespace
        p.valves = SimpleNamespace(EMBEDDING_CHUNK_SIZE=1000, EMBEDDING_CHUNK_OVERLAP=200)
        p.embedding_cache = EmbeddingCache()

        result = await p.get_embedding({})
        assert result is None

    import asyncio

    asyncio.run(_run())


def test_get_embedding_with_dict_text_uses_batch_call_and_returns_embedding():
    async def _run():
        p = Pipe()
        # Use a lightweight valves object to avoid requiring pydantic in test environment
        from types import SimpleNamespace
        p.valves = SimpleNamespace(EMBEDDING_CHUNK_SIZE=1000, EMBEDDING_CHUNK_OVERLAP=200)
        p.embedding_cache = EmbeddingCache()

        # Replace the internal batch call with a dummy that returns a known embedding
        async def dummy_batch(texts, batch_size=None, async_batch=None):
            assert isinstance(texts, list)
            # return a list with one embedding vector
            return [[0.123, 0.456, 0.789]]

        p._get_embeddings_batch = dummy_batch

        inp = {"text": "Sample text to embed"}
        emb = await p.get_embedding(inp)
        assert emb == [0.123, 0.456, 0.789]

    import asyncio

    asyncio.run(_run())
