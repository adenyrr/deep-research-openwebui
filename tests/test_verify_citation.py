import os
import sys
import pytest
import asyncio

# Ensure repository root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reuse the lightweight stubs from other tests to allow importing pipe.py
import types
sys.modules.setdefault('aiohttp', types.ModuleType('aiohttp'))

_pyd = types.ModuleType('pydantic')
class _BaseModel:
    pass

def _Field(*args, **kwargs):
    return None

_pyd.BaseModel = _BaseModel
_pyd.Field = _Field
sys.modules.setdefault('pydantic', _pyd)

_sklearn = types.ModuleType('sklearn')
_sklearn.metrics = types.ModuleType('sklearn.metrics')
_sklearn.metrics.pairwise = types.ModuleType('sklearn.metrics.pairwise')

def _cosine_similarity(a, b):
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

from pipe import Pipe


def test_verify_citation_batch_parses_json_array_without_format_errors():
    async def _run():
        p = Pipe()
        # Provide a tiny dummy valves to avoid pydantic complexity in tests
        from types import SimpleNamespace
        p.valves = SimpleNamespace(TEMPERATURE=0.7, RESEARCH_MODEL='test-model')

        # Mock generate_completion to return a response containing a JSON array
        async def dummy_generate_completion(model, messages, stream=False, temperature=None, stream_handler=None):
            return {
                "choices": [
                    {"message": {"content": '[{"verified": true, "global_id": "citation-1"}]'}}
                ]
            }

        p.generate_completion = dummy_generate_completion

        url = "http://example.com"
        citations = [{"text": "Sample excerpt", "global_id": "citation-1", "section": "Intro"}]
        source_content = "This is the source content which includes the excerpt."

        results = await p.verify_citation_batch(url, citations, source_content)
        assert isinstance(results, list)
        assert len(results) == 1
        r = results[0]
        assert r["verified"] is True
        assert r["global_id"] == "citation-1"
        assert r["url"] == url

    asyncio.run(_run())
