from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import tomllib
from pathlib import Path

from aiohttp import WSMsgType, web

from . import racer
from .runtimes import Runtime, make_runtime

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
log = logging.getLogger("boiltype")

RUNTIME = web.AppKey("runtime", Runtime)
PASSAGES = web.AppKey("passages", list)


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC / "index.html")


async def websocket(request: web.Request) -> web.WebSocketResponse:
    runtime = request.app[RUNTIME]
    passages = request.app[PASSAGES]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)

    async def send(msg: dict) -> None:
        if not ws.closed:
            await ws.send_json(msg)

    async def status() -> None:
        await send({"type": "status", "ready": runtime.ready.is_set(), "error": runtime.error,
                    "runtime": runtime.description})

    async def run_race(settings: racer.RaceSettings) -> None:
        await runtime.ready.wait()
        text = random.choice(passages)
        await send({"type": "race", "text": text, "countdown": 3})
        # The countdown doubles as time to template + prefill the prompt, so the
        # model doesn't get a head start or a penalty from prompt processing.
        prompt, _ = await asyncio.gather(racer.prepare(runtime, text), asyncio.sleep(3))
        await send({"type": "go"})

        async def emit(ev: dict) -> None:
            await send({"type": "model", **ev})

        await racer.race(runtime, prompt, text, settings, emit)

    async def wait_ready() -> None:
        await runtime.ready.wait()
        await status()

    task: asyncio.Task | None = None
    ready_task = asyncio.create_task(wait_ready())
    await status()
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            if task:
                task.cancel()
                task = None
            if data.get("type") == "start":
                task = asyncio.create_task(run_race(racer.RaceSettings.from_json(data.get("settings", {}))))
                task.add_done_callback(_log_failure)
    finally:
        ready_task.cancel()
        if task:
            task.cancel()
    return ws


def _log_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        log.error("race failed", exc_info=task.exception())


async def runtime_ctx(app: web.Application):
    runtime = app[RUNTIME]
    starter = asyncio.create_task(runtime.start())
    starter.add_done_callback(_log_failure)
    yield
    starter.cancel()
    await runtime.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="Race an LLM at typing.")
    p.add_argument("--config", type=Path, default=ROOT / "config.toml")
    p.add_argument("--profile", help="runtime profile from the config (default: config's default_profile)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true", help="show llama server logs")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if args.verbose:
        logging.getLogger("llama").setLevel(logging.DEBUG)
    else:
        logging.getLogger("llama").setLevel(logging.WARNING)

    config = tomllib.loads(args.config.read_text())
    server_cfg = config.get("server", {})
    app = web.Application()
    app[RUNTIME] = make_runtime(config, args.profile or config["default_profile"])
    app[PASSAGES] = [p.strip() for p in (ROOT / "passages.txt").read_text().split("\n\n") if p.strip()]
    app.cleanup_ctx.append(runtime_ctx)
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.router.add_static("/static", STATIC)
    web.run_app(app, host=args.host or server_cfg.get("host", "127.0.0.1"),
                port=args.port or server_cfg.get("port", 8080))


if __name__ == "__main__":
    main()
