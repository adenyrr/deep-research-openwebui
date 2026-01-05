### Deep Research At Home! 

**Version:** 0.5.0

Forked by adenyrr from : https://github.com/atineiatte/deep-research-at-home
For local only, use original script.

This script was designed to perform internet searches with searxng locally, but to use external tools for everything else.

The OCR engine, synthesis models, search models, and embedding models can come from different sources, as long as the API is OpenAI-compatible (format: https://domain.url/api/v1).

## WARNING

THIS SCRIPT HAS BEEN **MASSIVELY** MODIFIED USING AI TOOLS.

I use this script on a local openwebui instance, and I do not guarantee its use or safety in any way. I am simply sharing a script that I use and find useful.


Feel free to fork it! :)


## 🔧 Configuration des embeddings et streaming (exemples)

- Activer une API compatible OpenAI (ex. OpenRouter / Mistral) :

```python
p = Pipe()
p.valves.OPENAI_API_URL = "https://api.openrouter.ai/v1"
p.valves.OPENAI_API_KEY = "sk_..."
# Modèles selon votre fournisseur
p.valves.RESEARCH_MODEL = "mistral-large"
p.valves.EMBEDDING_MODEL = "text-embedding-3-large"
```

- Contrôler le batching / async des embeddings :

```python
p.valves.EMBEDDING_BATCH_SIZE = 1       # 1 = pas de batching
p.valves.EMBEDDING_BATCH_ASYNC = True     # exécute les batches concurrents
p.valves.EMBEDDING_BATCH_TIMEOUT = 30     # timeout par batch (s)
# Délai minimum (secondes) entre requêtes non-batch (single-item) — utile si votre fournisseur impose un rate-limit strict
p.valves.EMBEDDING_NONBATCH_RATE_LIMIT = 1.0
```

- Configurer le chunking des textes longs pour embeddings :

```python
p.valves.EMBEDDING_CHUNK_SIZE = 1200      # taille max (caractères) par chunk
p.valves.EMBEDDING_CHUNK_OVERLAP = 200    # overlap (caractères) entre chunks
```

- Exemple de streaming de complétion (OpenAI-compatible requis) :

```python
async def on_stream(partial):
    if partial is None:
        print("[STREAM DONE]")
        return
    print(partial, end="", flush=True)

await p.generate_completion(
    model=p.get_research_model(),
    messages=[{"role":"user","content":"Fais un résumé court de ceci."}],
    stream=True,
    stream_handler=on_stream
)
```

Cette configuration vous permet d'utiliser OpenRouter/Mistral pour les embeddings et la génération, de paralléliser les batches d'embeddings et de gérer le streaming des réponses en temps réel.

- Masquer les recherches pendant l'exécution (stream final seulement) :

```python
# n'affiche que la synthèse finale et les erreurs, masque les statuts et messages intermédiaires
# Les messages intermédiaires sont toujours envoyés mais marqués `masked=True` et encapsulés
# dans des balises `<think>...</think>` pour que l'interface puisse les afficher sur demande.
p.valves.STREAM_FINAL_ONLY = True
```

- Détails d'affichage :
# Les événements ont maintenant un champ `masked: True`. Les messages intermédiaires sont envoyés en clair mais avec `masked=True` et `masked_stream=True`; par défaut l'interface peut les afficher enveloppés dans `<think>...</think>`. Si vous préférez, le serveur peut lui‑même encapsuler ces messages côté serveur en activant `p.valves.STREAM_WRAP_MASKED = True`. Les erreurs (`level == "error"`) ne sont jamais masquées.

- Utilisation d'Ollama : Ollama est facultatif. Si vous fournissez `OPENAI_API_URL` (et facultativement `OPENAI_API_KEY`), l'API OpenAI-compatible sera utilisée par défaut.

- Configuration avancée par service : vous pouvez maintenant spécifier des endpoints distincts pour les embeddings et la synthèse :

```python
p.valves.EMBEDDING_API_URL = "https://api.openrouter.ai/v1"
p.valves.EMBEDDING_API_KEY = "sk_..."  # facultatif
p.valves.SYNTHESIS_API_URL = "https://api.openrouter.ai/v1"
p.valves.SYNTHESIS_API_KEY = "sk_..."  # facultatif
```

Si une option spécifique n'est pas fournie, le `OPENAI_API_URL` / `OPENAI_API_KEY` servira de fallback.

- OCR Mistral pour PDF : vous pouvez activer l'OCR Mistral pour les PDFs scannés (modèle `mistral-ocr-latest` par défaut) :

```python
p.valves.MISTRAL_OCR_ENABLED = True
p.valves.MISTRAL_API_URL = "https://mistral.ai/api/v1"
p.valves.MISTRAL_API_KEY = "sk_mistral..."  # facultatif
p.valves.MISTRAL_OCR_MODEL = "mistral-ocr-latest"
```

La fonction testera d'abord les extractions textuelles locales (PyPDF2/pdfplumber). Si ces méthodes échouent et que `MISTRAL_OCR_ENABLED=True`, le PDF sera envoyé au endpoint Mistral pour OCR. Notez que les détails d'API peuvent varier selon les fournisseurs; le client essaie plusieurs patterns d'endpoint courants.
