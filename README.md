# boiltype

Race a local LLM at typing.

```sh
uv run boiltype                  # default profile from config.toml (cpu)
uv run boiltype --profile gpu    # or any profile you add
uv run boiltype -v               # include llama server logs
```

Open http://127.0.0.1:8080 and press Enter.

## How the model races

The model is prompted to repeat the passage verbatim, and its output is streamed one token at a time.
A token that matches the next stretch of the passage is accepted. Any other token is a typo: the
stream is dropped, that token is banned at that position, and generation restarts from the last
correct character, just like backspacing and retyping. Restarts reuse llama.cpp's prompt cache, so
a typo costs about one token's worth of time.

Ways to handicap the model (in the UI): **temperature** (it copies perfectly up to ~2; at 3–5 the
typos start), **top-k** (higher means more wrong candidates when the temperature is up), and a
**max tok/s** cap. Hardware is the other handicap: pick a profile.

## Runtimes

`config.toml` defines profiles. Each one names a runtime from `RUNTIMES` in
`boiltype/runtimes.py` and supplies its options (binary, model, GPU layers, threads, extra args).
To add a backend, subclass `Runtime` and implement `start`/`stop`/`format_prompt`/`stream`.
