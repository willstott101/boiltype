"""Inference runtimes.

A runtime owns the model process (if any) and exposes three things the racer
needs: chat-template formatting, prompt warmup, and raw token streaming from a
prompt string. Add a new backend by subclassing `Runtime` and registering it in
`RUNTIMES`.
"""

from __future__ import annotations

import abc
import asyncio
import ctypes
import json
import logging
import os
import shutil
import signal
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import aiohttp

log = logging.getLogger(__name__)


def _die_with_parent() -> None:
    """Have the kernel SIGTERM the model server if we exit without cleaning up (Linux only)."""
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


@dataclass
class Token:
    text: str
    ids: list[int] = field(default_factory=list)


@dataclass
class Sampling:
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 1.0
    min_p: float = 0.0
    # Token ids that may not be sampled (used to stop the model repeating a typo).
    banned: list[int] = field(default_factory=list)


class Runtime(abc.ABC):
    def __init__(self, options: dict[str, Any]):
        self.options = options
        self.ready = asyncio.Event()
        self.error: str | None = None

    @property
    def description(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages to a raw prompt ending at the start of the assistant turn."""

    async def warm(self, prompt: str) -> None:
        """Optionally pre-process `prompt` so the first token comes out fast."""

    @abc.abstractmethod
    def stream(self, prompt: str, sampling: Sampling, max_tokens: int) -> AsyncIterator[Token]:
        """Stream tokens continuing `prompt`. Breaking out of the iterator must cancel generation."""


class LlamaCppRuntime(Runtime):
    """Spawns `llama serve` / `llama-server` and talks to its HTTP API."""

    def __init__(self, options: dict[str, Any]):
        super().__init__(options)
        self.host = options.get("host", "127.0.0.1")
        self.port = int(options.get("port", 9955))
        self.base_url = options.get("url") or f"http://{self.host}:{self.port}"
        self.proc: asyncio.subprocess.Process | None = None
        self.session: aiohttp.ClientSession | None = None
        self._log_task: asyncio.Task | None = None

    @property
    def description(self) -> str:
        model = self.options.get("model_path") or self.options.get("hf_repo") or "?"
        return f"llama.cpp · {os.path.basename(model)} · {self.options.get('profile', 'default')}"

    def command(self) -> list[str]:
        o = self.options
        binary = shutil.which(os.path.expanduser(o.get("binary", "llama-server")))
        if not binary:
            raise RuntimeError(f"llama binary not found: {o.get('binary')}")
        cmd = [binary, *o.get("subcommand", [])]
        if o.get("model_path"):
            cmd += ["-m", os.path.expanduser(o["model_path"])]
        elif o.get("hf_repo"):
            cmd += ["-hf", o["hf_repo"]]
        cmd += ["--host", self.host, "--port", str(self.port)]
        cmd += ["-c", str(o.get("ctx_size", 4096)), "-np", "1"]
        if "gpu_layers" in o:
            cmd += ["-ngl", str(o["gpu_layers"])]
        if "device" in o:
            cmd += ["-dev", str(o["device"])]
        if "threads" in o:
            cmd += ["-t", str(o["threads"])]
        cmd += [str(a) for a in o.get("extra_args", [])]
        return cmd

    async def start(self) -> None:
        self.session = aiohttp.ClientSession(base_url=self.base_url)
        try:
            if self.options.get("spawn", True):
                cmd = self.command()
                log.info("spawning: %s", " ".join(cmd))
                self.proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    preexec_fn=_die_with_parent if sys.platform == "linux" else None,
                )
                self._log_task = asyncio.create_task(self._pump_logs())
            await self._wait_healthy(float(self.options.get("startup_timeout", 600)))
            self.ready.set()
            log.info("runtime ready: %s", self.description)
        except Exception as e:
            self.error = str(e)
            log.exception("runtime failed to start")
            raise

    async def _pump_logs(self) -> None:
        assert self.proc and self.proc.stdout
        llama_log = logging.getLogger("llama")
        async for line in self.proc.stdout:
            llama_log.debug(line.decode(errors="replace").rstrip())

    async def _wait_healthy(self, timeout: float) -> None:
        assert self.session
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self.proc and self.proc.returncode is not None:
                raise RuntimeError(f"llama exited with code {self.proc.returncode}")
            try:
                async with self.session.get("/health") as r:
                    if r.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError("llama server did not become healthy")

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        if self._log_task:
            self._log_task.cancel()

    async def format_prompt(self, messages: list[dict[str, str]]) -> str:
        assert self.session
        body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
        async with self.session.post("/apply-template", json=body) as r:
            r.raise_for_status()
            return (await r.json())["prompt"]

    async def warm(self, prompt: str) -> None:
        assert self.session
        body = {"prompt": prompt, "n_predict": 0, "cache_prompt": True}
        async with self.session.post("/completion", json=body) as r:
            await r.read()

    async def stream(self, prompt: str, sampling: Sampling, max_tokens: int) -> AsyncIterator[Token]:
        assert self.session
        body = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "stream": True,
            "cache_prompt": True,
            # The racer decides when the passage is done; an early EOS would just stall it.
            "ignore_eos": True,
            "temperature": sampling.temperature,
            "top_k": sampling.top_k,
            "top_p": sampling.top_p,
            "min_p": sampling.min_p,
            "logit_bias": [[t, False] for t in sampling.banned],
        }
        # Leaving this context closes the connection, which makes llama.cpp abort the slot.
        async with self.session.post("/completion", json=body) as r:
            r.raise_for_status()
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("content") or chunk.get("tokens"):
                    yield Token(chunk.get("content", ""), chunk.get("tokens", []))
                if chunk.get("stop"):
                    return


RUNTIMES: dict[str, type[Runtime]] = {
    "llama.cpp": LlamaCppRuntime,
}


def make_runtime(config: dict[str, Any], profile: str) -> Runtime:
    profiles = config["profiles"]
    if profile not in profiles:
        raise SystemExit(f"unknown profile {profile!r}; have {sorted(profiles)}")
    options = {**profiles[profile], "profile": profile}
    kind = options.pop("runtime")
    return RUNTIMES[kind](options)
