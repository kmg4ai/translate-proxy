# translate-proxy

A transparent, universal translation relay for AI coding agents. You type in your
language — the model thinks in its language — the answer comes back in your language.

Sits between an AI client (Claude Code, opencode, Hermes, …) and the model endpoint it
already uses:

- **ingress** — your `USER_LANG` prompt (and history prose) is translated to `MODEL_LANG`
  before the model sees it (better reasoning, fewer tokens);
- **egress** — the model's `MODEL_LANG` answer is buffered, translated back to `USER_LANG`
  with a cheap translator model, and streamed to you in the original wire format.

Python standard library only. Works for **any language pair** (`USER_LANG`/`MODEL_LANG`).

## ✨ You need this

*You* write in *your* language. The model *always* thinks in English. The answer comes
back in *your* language. The English in between is invisible — and nearly free.

- 🗣️ Ask in Polish, German, Japanese, Arabic, Hindi, Swahili… ~100 languages, zero setup.
- 🧠 The model only ever sees and produces **English** — its strongest language. Better
  reasoning, and you never waste tokens on "sorry, could you say that in English?".
- 💸 **Save tokens** — say it once, in your language, done.
- 📉 **Save 40%.** 📈 **Save 140%.** (Math says you can't save more than 100%, but our
  heart says otherwise.)
- 🤖 Works with Claude Code, opencode, Hermes — anything that speaks the Anthropic or
  OpenAI wire formats.

### How the magic happens

```text
you (PL):       "Czesc, jak sie masz?"
   └─ proxy ──► model (EN): "Hi, how are you?"
   └─ model ──► proxy:      "I'm doing well, thank you!"
   └─ proxy ──► you (PL):   "U mnie wszystko dobrze, dziękuję!"
```

Your prompt is translated to English **once** (going in), and the model's answer is
translated back **once** (coming out). That's the whole cost.

### 💰 What you actually pay

The main model costs what your provider charges. **The only *extra* cost is the cheap
translation model** — one OpenRouter-style call per translation. One exchange =
2 translations (in + out).

Model prices (OpenRouter, per million tokens):

| Model | Input | Output |
|---|---|---|
| `google/gemini-2.5-flash` (default) | $0.30 | $2.50 |
| `deepseek/deepseek-v4-flash` (fallback) | $0.08 | $0.16 |

Cost per **single translation** (one direction):

| Text size | Example | gemini-2.5-flash (default) | deepseek-v4-flash (fallback) |
|---|---|---|---|
| Tiny (~20 tokens) | "How are you?" | ~$0.00008 | ~$0.00001 |
| Typical (~200 tokens) | a question about your code | ~$0.0006 | ~$0.00005 |
| Long (~800 tokens) | explain this whole file | ~$0.0023 | ~$0.0002 |

A busy coding session (30 exchanges, each a few hundred tokens) costs **a few cents of
translation in total**. Next to the main model's bill, that's rounding error.

## Quickstart

```bash
python3 translate_proxy.py --upstream http://127.0.0.1:8799
```

Point your agent at it:

| Tool | Change |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:8800` in `~/.claude/settings.json` |
| opencode | provider `baseURL` → `http://127.0.0.1:8800` in `opencode.json` |
| Hermes | `model.base_url: http://127.0.0.1:8800` in `config.yaml` |

`GET /health` → `{"ok": true}`. Stop: `python3 translate_proxy.py --port 8800 --stop`.

## Configuration

All config is via environment variables (see `.env.example`).

| Var | Default | Purpose |
|---|---|---|
| `USER_LANG` | `pl` | language you write in |
| `MODEL_LANG` | `en` | language the main model should use |
| `TRANSLATOR` | `openrouter` | primary translator backend (`openrouter` \| `deepseek` \| `cerebras`) |
| `TRANSLATOR_MODEL` | `google/gemini-2.5-flash` | model for the primary backend |
| `TRANSLATOR_FALLBACK` | `deepseek/deepseek-v4-flash` | comma-separated `backend/model` fallback chain |
| `TRANSLATE_HISTORY` | `true` | translate history prose to `MODEL_LANG` on ingress |
| `CACHE_SIZE` | `500` | in-memory translation cache bound |
| `GUARD_STOPWORD_RATIO` | `0.3` | above this English-stopword ratio input is passed through untranslated |
| `PLACEHOLDER` | `…` | shown while the egress answer is being translated |
| `OPENROUTER_API_KEY` | — | needed to translate via `openrouter` (without it that backend is skipped) |
| `DEEPSEEK_API_KEY` | — | needed to translate via `deepseek` (without it that backend is skipped) |

Backends:

- `openrouter` — `https://openrouter.ai/api/v1/chat/completions`, needs `OPENROUTER_API_KEY`;
- `deepseek` — `https://api.deepseek.com/chat/completions`, needs `DEEPSEEK_API_KEY`, model `deepseek-v4-flash`;
- `cerebras` — local gateway, `CEREBRAS_BASE` (default `http://127.0.0.1:8001/v1`), no key.

## Add your own language

Pick any pair — the proxy is for any person, any language. `USER_LANG` alone is enough
for ~100 built-in language codes (the full code list is in `.env.example`), e.g. German → English:

```bash
USER_LANG=de MODEL_LANG=en \
python3 translate_proxy.py --upstream http://127.0.0.1:8799
```

For a code not in the built-in list, also set `USER_LANG_NAME` (the language's English
name), e.g. `USER_LANG=xx USER_LANG_NAME=...`; otherwise it falls back to the code itself
in the prompt. If your `MODEL_LANG` isn't English, the already-English guard is disabled
(nothing to guard against).

## How it works

- **Never translated:** system prompt, `tool_use`, `tool_result`, `thinking` blocks, fenced
  code blocks, inline code, URLs, and text already in `MODEL_LANG`.
- **Fallback:** on translator failure the next backend in `TRANSLATOR_FALLBACK` is tried;
  only when the whole chain fails is the original English (egress) or your prompt (ingress)
  passed through, so you never lose an answer.
- The placeholder is cosmetic; disable it with `PLACEHOLDER=`.

## Security

- Binds `127.0.0.1` only — never expose it publicly.
- API keys live in environment variables, never in the repo.
- Keep it behind your existing local proxy chain (e.g. Claude Code's deepseek proxy).

## Tests

```bash
python3 -m unittest -v test_translate_proxy   # offline, translator mocked
```

## License

MIT.
