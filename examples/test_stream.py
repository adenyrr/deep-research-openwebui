import asyncio
from pipe import Pipe

async def on_stream(partial):
    if partial is None:
        print("\n[STREAM DONE]")
        return
    print(partial, end="", flush=True)

async def main():
    p = Pipe()
    p.valves.OPENAI_API_URL = "https://api.openrouter.ai/v1"  # or your provider
    p.valves.OPENAI_API_KEY = "YOUR_KEY"
    await p.generate_completion(
        model=p.get_research_model(),
        messages=[{"role":"user","content":"Ecris une courte introduction sur l'IA et la recherche."}],
        stream=True,
        stream_handler=on_stream,
    )

if __name__ == "__main__":
    asyncio.run(main())
