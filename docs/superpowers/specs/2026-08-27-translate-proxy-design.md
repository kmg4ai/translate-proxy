# translate-proxy — Design Spec

**Date:** 2026-08-27
**Status:** Approved (brainstorming), pending implementation plan
**Author:** kmg4ai (with Claude)

## Goal

A transparent, universal translation relay for AI coding agents (Claude Code, opencode, Hermes)
and any other client that can point at a custom API base URL.

The user writes in their own language (e.g. Polish). Before the main LLM sees the prompt, the
proxy translates it into the model's language (e.g. English). The model answers in its language
(English — better reasoning, more token-efficient). The proxy translates the answer back into the
user's language using a cheap, configurable model, and the user sees a native-language answer.

Result: better model reasoning + fewer input tokens (English is token-compact vs inflective
languages like Polish), with zero change to the user's experience.

## Background / Why

- Kim's tooling: Claude Code runs through `ANTHROPIC_BASE_URL=http://127.0.0.1:8799`
  (`/opt/deepseek-proxy/proxy.py`, Python stdlib `http.server`). opencode and Hermes are also
  configured with model `base_url`/provider endpoints.
- LLMs reason more reliably in their strongest language (English); Polish is inflective, so a
  single Polish word is often 2+ tokens while English is ~1 token per word. Translating input to
  English shrinks the whole context (especially long history) and improves reasoning quality.
- **Feasibility finding (verified):** Claude Code's hooks *cannot* rewrite the user's prompt.
  `UserPromptSubmit` only injects advisory `additionalContext`; there is no input-rewrite hook
  (open feature request: `anthropics/claude-code#27365`). The Anthropic Messages API has no
  user-turn prefill. The **only** point where the wire payload is editable is a proxy on
  `ANTHROPIC_BASE_URL`. opencode has a native `chat.message` hook that can mutate input, but
  output translation is not cleanly supported by opencode plugins. Hermes is a gateway with
  `base_url` config. → **A standalone translation proxy is the one architecture that covers all
  three tools transparently.**
- The proxy is deliberately **language-agnostic** (configurable `USER_LANG`/`MODEL_LANG`) so it
  can be published to GitHub and used by anyone for any language pair.

## Decisions (made with the user)

| Decision | Choice |
|---|---|
| Output strategy | **Buffered whole-response translation** (v1). Sentence-level streaming later. |
| Language handling | **Config-driven** `USER_LANG`/`MODEL_LANG`, NOT auto-detection. |
| Architecture | **Standalone Python proxy** (stdlib), one mechanism for all tools. |
| Translator backend | **OpenRouter `google/gemini-2.5-flash`** default, fallback **DeepSeek `deepseek-v4-flash`** (2026-08-27: cerebras odrzucony). |
| Language-pair input | `USER_LANG`/`MODEL_LANG` + display names; works for any pair (`pl→en`, `de→en`, `uk→en`, …). |
| Distribution | Public GitHub repo, single-file stdlib Python, English docs, license, example configs. |

## Architecture

```
+----------------+      +------------------------+      +---------------------------+
| Claude Code    |      |  translate-proxy :8800 |      |  upstream (already used)  |
| opencode       | ───► |                        | ───► |  • Claude Code → :8799    |
| Hermes         |      |  Anthropic + OpenAI    |      |  • opencode → OpenRouter  |
+----------------+      |  format-transparent    |      |  • Hermes → DeepSeek      |
                        +-----------+------------+      +---------------------------+
                                    │ translator
                        +-----------▼------------+
                        |  openrouter gemini-2.5-flash (def.) |
                        |  fallback: deepseek-v4-flash        |
                        +------------------------+
```

### Request flow (ingress)

1. Parse the request body into a format-independent list of messages with text parts.
2. Rewrite text for the model:
   - **user text** — translate `USER_LANG → MODEL_LANG`;
   - **assistant prose in history** — the history displayed to the user is in `USER_LANG`
     (that's what the client renders), so translate history prose `USER_LANG → MODEL_LANG` so the
     model's context is uniformly `MODEL_LANG` (token savings, stateless);
   - **system prompt, tool_use, tool_result, code fences** — never translated.
3. Forward to the configured upstream in the **same** wire format (Anthropic or OpenAI), streaming
   or not, unchanged otherwise.

### Response flow (egress, buffered v1)

1. Read the full upstream stream. Accumulate assistant text deltas per content block; pass
   `tool_use` blocks through immediately (verbatim).
2. When the stream finishes: translate the **complete** English text `MODEL_LANG → USER_LANG`
   (one call — full context, best quality).
3. Emit to the client in the original format: `content_block_start` + `text_delta`(Polish) +
   `content_block_stop` + `message_delta` + `message_stop` (Anthropic), or `data:` chunks +
   `finish_reason` (OpenAI).
4. Send a small placeholder to the client first so the UI shows "…" while buffering.

Non-streaming responses take the same path: read the full JSON body, translate the assistant
text, return the same JSON structure with the translated text and passthrough `usage`.

## Components

### `translate_proxy.py` (single file, stdlib only)

- `http.server`-based relay (pattern: `/opt/deepseek-proxy/proxy.py`), binds `127.0.0.1` only.
- CLI: `--port` (default 8800), `--upstream` (required), `--verbose`, `--stop`, `--health`.
- Both wire formats by path: `/v1/messages` (Anthropic) and `/v1/chat/completions` (OpenAI);
  both streaming (SSE) and non-streaming. Format-transparent: translation operates on a normalized
  message model, then re-serializes in the original format.
- `GET /health` → `{"ok": true}` (for systemd/`curl` checks).

### Translation layer

- Interface: `translate(text, src_lang, dst_lang) -> str`.
- Backends (`TRANSLATOR` picks the primary; `TRANSLATOR_FALLBACK` is the fallback chain):
  - `openrouter` (default) — `https://openrouter.ai/api/v1/chat/completions`, model from
    `TRANSLATOR_MODEL` (default `google/gemini-2.5-flash`), key `OPENROUTER_API_KEY`;
  - `deepseek` — `https://api.deepseek.com/chat/completions`, `DEEPSEEK_API_KEY`,
    model `deepseek-v4-flash`;
  - `cerebras` (optional, for self-hosters) — OpenAI-format `chat/completions` to
    `http://127.0.0.1:8001`, model `gpt-oss-120b`. Not part of the default chain.
- **Fallback chain:** on timeout/429/5xx the proxy tries the next entry in `TRANSLATOR_FALLBACK`
  (comma-separated `backend/model` pairs). Only when the entire chain fails does the
  caller-visible fallback apply (see "Error handling").
- In-memory cache keyed `(src, dst, text)` with bounded size (`CACHE_SIZE`, default e.g. 500),
  so repeated blocks (common phrases, repeated prompts) are not re-translated.
- Prompt template built from config language names (not hard-coded):
  - `USER_LANG → MODEL_LANG`: "Translate the following {src} text into {dst}. The input may be
    written without diacritics (e.g. Polish without ą/ę/ś) — reconstruct the intended words, do not
    translate letter-by-letter. Preserve Markdown, code blocks verbatim (never translate code),
    inline code, URLs, numbers, and technical terms unchanged. Output only the translation."
  - `MODEL_LANG → USER_LANG`: same template with "natural, idiomatic {dst}".

### Config (env vars + CLI)

| Var | Default | Purpose |
|---|---|---|
| `USER_LANG` | `pl` | language the user writes in |
| `USER_LANG_NAME` | (from map) | display name for the prompt (e.g. "Polish") |
| `MODEL_LANG` | `en` | language the main model should use |
| `MODEL_LANG_NAME` | (from map) | display name (e.g. "English") |
| `TRANSLATOR` | `openrouter` | primary translator backend |
| `TRANSLATOR_MODEL` | `google/gemini-2.5-flash` | model for the primary backend |
| `TRANSLATOR_FALLBACK` | `deepseek/deepseek-v4-flash` | comma-separated `backend/model` fallback chain |
| `TRANSLATE_HISTORY` | `true` | translate history prose to `MODEL_LANG` on ingress |
| `CACHE_SIZE` | `500` | in-memory translation cache bound |
| `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` | — | translator keys |
| `--upstream` | required | full upstream base URL (proxy appends path) |

Built-in language-name map for common codes (`pl`, `en`, `de`, `fr`, `es`, `it`, `uk`, `ru`,
`cs`, `nl`, `sv`, `da`, `no`, `fi`, `pt`, `tr`, …); unknown codes fall back to the code itself.

### Minimal guard (not full detection)

- If a text part already appears to be in `MODEL_LANG` (English stopword count high),
  skip input translation — avoids double-translating English input pasted into a Polish
  prompt. Implemented predicate (adjudicated 2026-08-27): when `MODEL_LANG=en`, skip only
  when at least **2** English stopword hits AND the stopword ratio ≥ `GUARD_STOPWORD_RATIO`
  (default 0.3). The min-2-hits floor keeps short Polish phrases such as "zrob to" /
  "co to jest" (whose "to"/"i" are English stopwords) from being skipped; pasted English
  blocks produce many hits and are still detected.
- Text inside fenced code blocks and inline code is never translated.

## What is NEVER translated

- system prompt
- `tool_use` blocks (structured JSON / tool params)
- `tool_result` blocks (code, file contents, command output)
- `thinking` blocks (passed through verbatim — hidden from the user anyway)
- content inside fenced code blocks
- text already in `MODEL_LANG` (guard)
- URLs, file paths, numbers

## Wire / protocol notes

- Streaming is preserved end-to-end at the transport level; only the text payload is buffered on
  egress.
- `usage` is passed through from upstream (counts the English output tokens). Claude Code's token
  counter stays correct. The displayed Polish length differs from the counted tokens — accepted.
- Claude Code caveats already in effect via `:8799` (Remote Control disabled for non-Anthropic
  base URL; MCP tool search disabled) — no new limitations are introduced.
- If the upstream rejects beta fields, documented workarounds exist
  (`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`).

## Error handling & edge cases

| Case | Behavior |
|---|---|
| Upstream unreachable | `502 {"error": "upstream unreachable"}` (deepseek-proxy pattern) |
| Translator failure (egress) | Try next model in `TRANSLATOR_FALLBACK`; **only when the whole chain fails: return the original English text** — user never loses the answer; log warning |
| Translator failure (ingress) | Try next model in `TRANSLATOR_FALLBACK`; only when the whole chain fails: forward the untranslated (user-language) prompt |
| Empty/whitespace text | Pass through untouched |
| Non-PL input, pasted English | Guard skips translation |
| Caseless Polish ("bez ogonków") | Translator prompt reconstructs intent; no detection needed |
| Very long answer | Single buffered translation call; `--upstream` timeout configured high |

## Deployment

- systemd unit `translate-proxy.service` (bind `127.0.0.1`, `Restart=always`).
- One proxy instance per upstream (env per instance). Typical wiring:

| Tool | Change | proxy `--upstream` |
|---|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:8800` | `http://127.0.0.1:8799` |
| opencode | provider `baseURL` → `http://127.0.0.1:8800` | current provider endpoint (e.g. OpenRouter) |
| Hermes | `config.yaml` → `model.base_url: http://127.0.0.1:8800` | `https://api.deepseek.com` |

## Testing (offline, stdlib `unittest` — no network)

- Ingress: user text `pl→en` (mocked translator), history prose → en, system/tool blocks untouched,
  code fences untouched, already-English input passthrough.
- Egress: buffered SSE re-emission in both formats, tool blocks preserved in order, usage passthrough,
  translator-failure fallback to English.
- Fallback chain: primary translator fails → next entry in `TRANSLATOR_FALLBACK` used; whole chain
  fails → caller-visible fallback (English / untranslated), logged.
- Guard: English-stopword skip; mixed-language text.
- Config: language pair change (e.g. `de→en`) rewires prompts correctly.
- Health endpoint.

## GitHub publication (part of this project)

- Public repo `translate-proxy` (owner kmg4ai). Repo contents:
  - `translate_proxy.py` (single file, stdlib-only)
  - `test_translate_proxy.py`
  - `README.md` — quickstart; config table; *how to add your own language*; wiring guide for
    Claude Code / opencode / Hermes; security notes (local-only bind, keys via env)
  - `LICENSE` (MIT suggested)
  - `.env.example`, `translate-proxy.service` (systemd example)
- `gh repo create` + push (requires `gh auth` — verify before publishing).

## Non-goals (future work)

- Sentence-level streaming translation (chosen "later").
- Stateful canonical-English history (perfect fidelity for long sessions).
- Web UI / hosted service / multi-user.
- Translating tool results (deliberately never done).

## Resolved at self-review

- Default port **8800**, default cache size **500** (as fixed in the tables above).
- No `v1/models` passthrough (YAGNI — clients that need it can point at the upstream directly).
- Repo name **`translate-proxy`** (as used throughout).
