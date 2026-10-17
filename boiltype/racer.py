"""The model's side of the race.

The model is asked to repeat the passage verbatim. Every streamed token is
checked against the passage: a token that matches the next stretch of text is
"typed"; anything else is a typo. On a typo the stream is abandoned (everything
after it was conditioned on the mistake), the bad token is banned at that
position, and generation restarts from the last correct character - the
model's equivalent of hitting backspace and retyping.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import Awaitable, Callable

from .runtimes import Runtime, Sampling

SYSTEM_PROMPT = (
    "You are a typist. Reproduce the user's text exactly, character for character. "
    "Do not add, remove or change anything: no quotes, no preamble, no commentary."
)

Emit = Callable[[dict], Awaitable[None]]


@dataclass
class RaceSettings:
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 1.0
    max_tps: float = 0.0  # 0 = unthrottled

    @classmethod
    def from_json(cls, d: dict) -> RaceSettings:
        return cls(
            temperature=max(0.0, float(d.get("temperature", 0.0))),
            top_k=int(d.get("top_k", 40)),
            top_p=float(d.get("top_p", 1.0)),
            max_tps=max(0.0, float(d.get("max_tps", 0.0))),
        )


async def prepare(runtime: Runtime, text: str) -> str:
    prompt = await runtime.format_prompt(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    )
    await runtime.warm(prompt)
    return prompt


async def race(runtime: Runtime, prompt: str, text: str, settings: RaceSettings, emit: Emit) -> None:
    start = time.monotonic()
    min_gap = 1.0 / settings.max_tps if settings.max_tps else 0.0
    last = start
    accepted = 0
    tokens = rejects = 0
    banned: list[int] = []

    async def pace() -> None:
        nonlocal last
        if min_gap:
            await asyncio.sleep(max(0.0, last + min_gap - time.monotonic()))
        last = time.monotonic()

    while accepted < len(text):
        sampling = Sampling(settings.temperature, settings.top_k, settings.top_p, banned=list(banned))
        remaining = text[accepted:]
        typo = None
        # aclosing() drops the HTTP stream as soon as we break, freeing llama's slot for the retry.
        async with aclosing(runtime.stream(prompt + text[:accepted], sampling, len(remaining) + 8)) as stream:
            async for tok in stream:
                await pace()
                if tok.text and remaining.startswith(tok.text):
                    accepted += len(tok.text)
                    remaining = remaining[len(tok.text):]
                    tokens += 1
                    banned.clear()
                    await emit({"ev": "accept", "pos": accepted, "tok": tok.text})
                    if not remaining:
                        break
                elif tok.text.startswith(remaining):
                    # Overshot the end (e.g. trailing newline) - the passage is complete.
                    accepted = len(text)
                    tokens += 1
                    await emit({"ev": "accept", "pos": accepted, "tok": remaining})
                    break
                elif tok.text:
                    typo = tok
                    break
                # Empty text = partial UTF-8 sequence still being assembled; keep reading.

        if typo is not None:
            rejects += 1
            banned.extend(typo.ids)
            await emit({"ev": "reject", "pos": accepted, "tok": typo.text})

    await emit({
        "ev": "finish",
        "ms": round((time.monotonic() - start) * 1000),
        "tokens": tokens,
        "rejects": rejects,
    })
