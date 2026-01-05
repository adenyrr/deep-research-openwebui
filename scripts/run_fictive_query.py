import sys
import os
import asyncio
from types import SimpleNamespace

# Ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Provide minimal stubs for optional external packages so this smoke script can run
import types
sys.modules.setdefault('aiohttp', types.ModuleType('aiohttp'))

# Minimal pydantic stub to allow importing Valve BaseModel/Field definitions
_pyd = types.ModuleType('pydantic')
class _BaseModel:
    pass

def _Field(*args, **kwargs):
    return None
_pyd.BaseModel = _BaseModel
_pyd.Field = _Field
sys.modules.setdefault('pydantic', _pyd)

# Minimal sklearn stub
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

# Minimal stubs for open_webui package used by the pipe module
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
    def __init__(self, *args, **kwargs):
        pass
_open_webui.models.users.User = _User
sys.modules.setdefault('open_webui', _open_webui)
sys.modules.setdefault('open_webui.constants', _open_webui.constants)
sys.modules.setdefault('open_webui.main', _open_webui.main)
sys.modules.setdefault('open_webui.models', _open_webui.models)
sys.modules.setdefault('open_webui.models.users', _open_webui.models.users)

from pipe import Pipe, EmbeddingCache

async def main():
    p = Pipe()

    # Minimal valves to keep the run short and non-interactive
    # DefaultValves provides safe defaults for any attribute access not explicitly set.
    class DefaultValves:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getattr__(self, name):
            # Default fallbacks: booleans False, strings empty, numeric 0
            return False

    p.valves = DefaultValves(
        ENABLED=True,
        TEMPERATURE=0.7,
        EMBEDDING_CHUNK_SIZE=1000,
        EMBEDDING_CHUNK_OVERLAP=200,
        STREAM_FINAL_ONLY=False,
        STREAM_WRAP_MASKED=False,
        INTERACTIVE_RESEARCH=False,
        MAX_CYCLES=1,
        MIN_CYCLES=1,
        SEARCH_RESULTS_PER_QUERY=3,
        EXTRA_RESULTS_PER_QUERY=0,
        QUALITY_FILTER_ENABLED=False,
        OPENAI_API_URL="",
        OLLAMA_URL="",
        RESEARCH_MODEL="gemma3:12b",
    )

    # Lightweight caches
    p.embedding_cache = EmbeddingCache()

    # Prevent background vocab-download tasks from running during this smoke test
    async def _noop_load_prebuilt_vocabulary_embeddings():
        return None

    p.load_prebuilt_vocabulary_embeddings = _noop_load_prebuilt_vocabulary_embeddings

    # Ensure model attributes used by the pipe exist
    p.valves.RESEARCH_MODEL = getattr(p.valves, 'RESEARCH_MODEL', 'gemma3:12b')
    p.valves.SYNTHESIS_MODEL = getattr(p.valves, 'SYNTHESIS_MODEL', '')

    # Capture emitted messages/statuses
    emitted = []

    async def fake_emitter(obj):
        emitted.append(obj)
        # Print a compact form for quick feedback
        t = obj.get("type")
        data = obj.get("data")
        print(f"EMIT [{t}]: {data.get('content') if t=='message' else data.get('description')}\n")

    # Stub generate_completion to return predictable responses
    async def fake_generate_completion(model, messages, temperature=None, stream=False, stream_handler=None):
        # Look at the user content to decide which stubbed response to return
        user_msg = messages[-1]["content"] if messages and isinstance(messages, list) else ""
        if "Generate initial search queries" in user_msg or "Generate initial search queries for this user query" in user_msg:
            return {"choices": [{"message": {"content": '{"queries": ["Définition des minéraux critiques UE", "Liste des minéraux critiques UE", "Politiques industrielles récentes minéraux critiques"]}'}}]}
        if "Generate a comprehensive research outline" in user_msg or "Generate a comprehensive research outline that builds on previous research" in user_msg:
            return {"choices": [{"message": {"content": '{"outline": [{"topic": "Minéraux critiques", "subtopics": ["Définition", "Liste selon UE", "Politiques industrielles récentes"]}]}'}}]}
        # Fallback
        return {"choices": [{"message": {"content": "{}"}}]}

    p.generate_completion = fake_generate_completion

    # Stub get_embedding to return a fixed-length embedding
    async def fake_get_embedding(text, *args, **kwargs):
        # Return a simple small embedding vector
        return [0.1] * 384

    p.get_embedding = fake_get_embedding

    # Stub process_query to return a small, valid result
    async def fake_process_query(query, query_embedding, outline_embedding, cycle_feedback, summary_embedding):
        return [{
            "url": f"http://example.com/{query.replace(' ', '_')}",
            "title": f"Result for {query}",
            "content": f"Sample content for {query}",
            "query": query,
            "valid": True,
            "tokens": 400,
            "similarity": 0.8,
        }]

    p.process_query = fake_process_query

    # Run the pipe with a single user message (the French minerals query)
    body = {"messages": [{"id": "m1", "content": "Définition et liste des minéraux critiques selon l'Union européenne et les politiques industrielles récentes"}]}
    user = {"id": "test_user", "name": "Tester"}

    result = await p.pipe(body, user, __event_emitter__=fake_emitter)

    print("\nPIPE RETURNED:\n", result)
    print("\nEMITTED MESSAGES SUMMARY:\n")
    for e in emitted:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())
