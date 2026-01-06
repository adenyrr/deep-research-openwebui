import pytest
from types import SimpleNamespace
import asyncio

from pipe import Pipe


@pytest.mark.asyncio
async def test_wrap_masked_message_normalizes_think_tags():
    p = Pipe()
    # Minimal user so get_state can generate temporary conversation id
    p.__user__ = SimpleNamespace(id="tester")

    captured = []

    async def emitter(event):
        captured.append(event)

    p.__current_event_emitter__ = emitter

    # Force streaming/masking and server-side wrapper
    p.valves.STREAM_FINAL_ONLY = True
    p.valves.STREAM_WRAP_MASKED = True

    # Message that starts with a stray closing tag (the reported issue)
    bad_message = "</think><think>Query 1: {'query': 'Donald Trump biographie générale et parcours politique', 'topic': \"informations générales sur l'identité, la vie et le rôle politique de Donald Trump\"}"

    await p.emit_message(bad_message)

    assert len(captured) == 1

    data = captured[0]["data"]
    content = data["content"]

    # Should be properly wrapped with a single opening then closing tag
    assert content.startswith("<think>"), f"missing opening tag: {content}"
    assert content.endswith("</think>"), f"missing closing tag: {content}"
    # No stray leading closing tags
    assert not content.lstrip().startswith("</think>"), f"stray closing tag remains: {content}"
    # Exactly one pair of tags
    assert content.count("<think>") == 1 and content.count("</think>") == 1
