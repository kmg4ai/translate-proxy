# translate-proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-file Python translation relay (stdlib-only) that rewrites user prompts to the model's language on ingress and translates the model's answer back to the user's language on egress, serving Claude Code, opencode, and Hermes through their `base_url`/`ANTHROPIC_BASE_URL` settings.

**Architecture:** A threaded `http.server` on `127.0.0.1:8800` that is format-transparent (Anthropic `/v1/messages` + OpenAI `/v1/chat/completions`, streaming + non-streaming). Pure transform functions operate on parsed bodies/SSE events so all logic is testable offline with a mocked translator. The translator is a cheap LLM reached via OpenAI `chat/completions` (OpenRouter `google/gemini-2.5-flash` primary, DeepSeek `deepseek-v4-flash` fallback).

**Tech Stack:** Python 3 stdlib only (`http.server`, `urllib`, `argparse`, `json`, `re`, `threading`, `unittest`). No third-party dependencies.

## Global Constraints

Copy these verbatim into every review; they are binding:

- **Python stdlib only** — no third-party imports in `translate_proxy.py`.
- **Bind `127.0.0.1` only** — never `0.0.0.0`.
- **Config defaults** (from spec): `USER_LANG=pl`, `USER_LANG_NAME=Polish`, `MODEL_LANG=en`, `MODEL_LANG_NAME=English`, `TRANSLATOR=openrouter`, `TRANSLATOR_MODEL=google/gemini-2.5-flash`, `TRANSLATOR_FALLBACK=deepseek/deepseek-v4-flash`, `TRANSLATE_HISTORY=true`, `CACHE_SIZE=500`, `GUARD_STOPWORD_RATIO=0.3`, `PLACEHOLDER=…`, default port `8800`.
- **Never translate:** system prompt, `tool_use` blocks, `tool_result` blocks, `thinking` blocks, content inside fenced code blocks, inline code, URLs, and text already in `MODEL_LANG`.
- **Guard:** when `MODEL_LANG=en`, input text with **≥ 2** English stopword hits AND stopword ratio ≥ `GUARD_STOPWORD_RATIO` (0.3) is passed through untranslated (min-2-hits floor: short Polish phrases with "to"/"i" are not skipped; adjudicated 2026-08-27).
- **Fallback semantics:** try primary translator, then each `TRANSLATOR_FALLBACK` entry; only when the **entire chain fails**: egress returns the original English text, ingress forwards the untranslated prompt.
- **Both wire formats, both stream modes** must be handled.
- **Offline tests:** stdlib `unittest`, translator mocked via injected callable; only loopback (`127.0.0.1`) HTTP test servers allowed, never external network.
- Git author `kmg4ai`; every commit ends with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Repo root is the project root; run tests with `python3 -m unittest -v test_translate_proxy`.

---

## File Structure

- `translate_proxy.py` — the whole relay: config, protect/restore, guard, translator client + fallback + cache, ingress/egress transforms, HTTP handler, CLI. One file per spec; internally organized into clearly-labeled sections with pure functions first, HTTP layer last.
- `test_translate_proxy.py` — offline unittest suite, one test class per feature; grows with each task.
- `README.md` — quickstart, config table, *how to add your own language*, per-tool wiring (Claude Code / opencode / Hermes), security notes. Created in Task 8.
- `LICENSE` — MIT, `kmg4ai` 2026. Created in Task 8.
- `.env.example` — documented env var template. Created in Task 8.
- `translate-proxy.service` — systemd example. Created in Task 8.
- `.gitignore` — exists (`__pycache__/`, `*.pyc`, `.env`, `.claude/`). Leave as is.

Every task commits a green suite. The implementer appends new test classes to `test_translate_proxy.py` and implements in `translate_proxy.py`; each task leaves the repo runnable (`python3 translate_proxy.py --help` works after Task 4).

---
### Task 1: Config + language map

**Files:**
- Create: `translate_proxy.py`
- Create: `test_translate_proxy.py`

**Interfaces:**
- Produces: `LANG_NAMES`, `lang_name(code) -> str`, `parse_bool(v, default) -> bool`, `parse_fallback(s) -> list[(backend, model)]`, `BACKENDS` dict, `class Config`, `load_config(argv=None) -> Config`.

- [ ] **Step 1: Write the failing test**

```python
import os
import unittest

from translate_proxy import load_config, lang_name, parse_bool, parse_fallback


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config(["--upstream", "http://127.0.0.1:8799"])
        self.assertEqual(cfg.port, 8800)
        self.assertEqual(cfg.user_lang, "pl")
        self.assertEqual(cfg.user_lang_name, "Polish")
        self.assertEqual(cfg.model_lang, "en")
        self.assertEqual(cfg.model_lang_name, "English")
        self.assertEqual(cfg.translator, "openrouter")
        self.assertEqual(cfg.translator_model, "google/gemini-2.5-flash")
        self.assertEqual(cfg.translator_fallback, [("deepseek", "deepseek-v4-flash")])
        self.assertTrue(cfg.translate_history)
        self.assertEqual(cfg.cache_size, 500)
        self.assertEqual(cfg.placeholder, "…")

    def test_env_overrides(self):
        os.environ["USER_LANG"] = "de"
        os.environ["TRANSLATOR"] = "deepseek"
        os.environ["TRANSLATE_HISTORY"] = "false"
        os.environ["CACHE_SIZE"] = "10"
        try:
            cfg = load_config(["--upstream", "u"])
            self.assertEqual(cfg.user_lang, "de")
            self.assertEqual(cfg.user_lang_name, "German")
            self.assertEqual(cfg.translator, "deepseek")
            self.assertEqual(cfg.translator_model, "deepseek-v4-flash")
            self.assertFalse(cfg.translate_history)
            self.assertEqual(cfg.cache_size, 10)
        finally:
            for k in ("USER_LANG", "TRANSLATOR", "TRANSLATE_HISTORY", "CACHE_SIZE"):
                os.environ.pop(k, None)

    def test_requires_upstream(self):
        with self.assertRaises(SystemExit):
            load_config([])

    def test_lang_name_unknown_falls_back(self):
        self.assertEqual(lang_name("xx"), "xx")

    def test_parse_fallback_forms(self):
        self.assertEqual(
            parse_fallback("deepseek/deepseek-v4-flash, cerebras"),
            [("deepseek", "deepseek-v4-flash"), ("cerebras", "gpt-oss-120b")],
        )
        self.assertEqual(parse_fallback(""), [])

    def test_parse_bool(self):
        self.assertTrue(parse_bool("true"))
        self.assertFalse(parse_bool("no"))
        self.assertTrue(parse_bool(None, True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v test_translate_proxy`
Expected: FAIL with `ModuleNotFoundError: No module named 'translate_proxy'`.

- [ ] **Step 3: Write minimal implementation**

Create `translate_proxy.py`:

```python
"""translate-proxy — transparent translation relay for LLM coding agents.

Sits between an AI client (Claude Code, opencode, Hermes, ...) and the model
endpoint it already uses. Rewrites the user's language (USER_LANG) into the
model's language (MODEL_LANG) on the way in, and translates the model's answer
back to USER_LANG on the way out with a cheap, configurable translator model.

Python standard library only. Bind to 127.0.0.1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "0.1.0"

LANG_NAMES = {
    "pl": "Polish", "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "uk": "Ukrainian", "ru": "Russian", "cs": "Czech", "nl": "Dutch",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "pt": "Portuguese",
    "tr": "Turkish", "hu": "Hungarian", "ro": "Romanian", "sk": "Slovak", "bg": "Bulgarian",
    "el": "Greek", "hr": "Croatian", "sr": "Serbian", "lt": "Lithuanian", "lv": "Latvian",
    "et": "Estonian", "sl": "Slovenian",
}


def lang_name(code: str) -> str:
    """Human-readable language name for a code, falling back to the code itself."""
    return LANG_NAMES.get(code, code)


def parse_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


BACKENDS = {
    # name -> (default OpenAI chat/completions URL, default model, key env var or None)
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "google/gemini-2.5-flash", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/chat/completions", "deepseek-v4-flash", "DEEPSEEK_API_KEY"),
    "cerebras": (None, "gpt-oss-120b", None),  # URL comes from CEREBRAS_BASE (+ /chat/completions)
}


def parse_fallback(s: str):
    """'deepseek/deepseek-v4-flash, cerebras' -> [('deepseek','deepseek-v4-flash'), ('cerebras','gpt-oss-120b')]"""
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "/" in part:
            backend, model = part.split("/", 1)
            out.append((backend.strip(), model.strip()))
        else:
            model = BACKENDS[part][1] if part in BACKENDS else ""
            out.append((part, model))
    return out


class Config:
    def __init__(self, port, upstream, verbose, placeholder, user_lang, user_lang_name,
                 model_lang, model_lang_name, translator, translator_model, translator_fallback,
                 translate_history, cache_size, guard_ratio, translator_timeout, upstream_timeout,
                 cerebras_base, stop_requested=False, health_only=False, stop_file=None):
        self.port = port
        self.upstream = upstream
        self.verbose = verbose
        self.placeholder = placeholder
        self.user_lang = user_lang
        self.user_lang_name = user_lang_name
        self.model_lang = model_lang
        self.model_lang_name = model_lang_name
        self.translator = translator
        self.translator_model = translator_model
        self.translator_fallback = translator_fallback
        self.translate_history = translate_history
        self.cache_size = cache_size
        self.guard_ratio = guard_ratio
        self.translator_timeout = translator_timeout
        self.upstream_timeout = upstream_timeout
        self.cerebras_base = cerebras_base
        self.stop_requested = stop_requested
        self.health_only = health_only
        self.stop_file = stop_file


def load_config(argv=None) -> Config:
    ap = argparse.ArgumentParser(
        prog="translate-proxy",
        description="Translation relay for LLM coding agents (USER_LANG <-> MODEL_LANG).",
    )
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8800")))
    ap.add_argument("--upstream", default=os.environ.get("UPSTREAM"), help="upstream base URL, e.g. http://127.0.0.1:8799")
    ap.add_argument("--verbose", action="store_true", default=parse_bool(os.environ.get("VERBOSE")))
    ap.add_argument("--placeholder", default=os.environ.get("PLACEHOLDER", "…"))
    ap.add_argument("--stop", action="store_true", help="request the running instance to stop")
    ap.add_argument("--health", action="store_true", help="print GET /health of the running instance")
    args = ap.parse_args(argv)
    if not args.upstream:
        ap.error("--upstream is required (or UPSTREAM env)")

    user_lang = os.environ.get("USER_LANG", "pl").strip().lower()
    model_lang = os.environ.get("MODEL_LANG", "en").strip().lower()
    translator = os.environ.get("TRANSLATOR", "openrouter").strip().lower()
    translator_model = os.environ.get("TRANSLATOR_MODEL", BACKENDS[translator][1] if translator in BACKENDS else "")
    fallback_raw = os.environ.get("TRANSLATOR_FALLBACK", "deepseek/deepseek-v4-flash")

    return Config(
        port=args.port,
        upstream=args.upstream,
        verbose=args.verbose,
        placeholder=args.placeholder,
        user_lang=user_lang,
        user_lang_name=os.environ.get("USER_LANG_NAME", lang_name(user_lang)),
        model_lang=model_lang,
        model_lang_name=os.environ.get("MODEL_LANG_NAME", lang_name(model_lang)),
        translator=translator,
        translator_model=translator_model,
        translator_fallback=parse_fallback(fallback_raw),
        translate_history=parse_bool(os.environ.get("TRANSLATE_HISTORY"), True),
        cache_size=int(os.environ.get("CACHE_SIZE", "500")),
        guard_ratio=float(os.environ.get("GUARD_STOPWORD_RATIO", "0.3")),
        translator_timeout=int(os.environ.get("TRANSLATOR_TIMEOUT", "60")),
        upstream_timeout=int(os.environ.get("UPSTREAM_TIMEOUT", "300")),
        cerebras_base=os.environ.get("CEREBRAS_BASE", "http://127.0.0.1:8001/v1"),
        stop_requested=args.stop,
        health_only=args.health,
        stop_file=f"/tmp/translate-proxy-{args.port}.stop",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest -v test_translate_proxy`
Expected: 6 tests PASS (`ConfigTests`).

- [ ] **Step 5: Commit**

```bash
git add translate_proxy.py test_translate_proxy.py
git commit -m "feat: config loading + language map

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Protect/restore code spans

**Files:**
- Modify: `translate_proxy.py` (append after `parse_fallback`)
- Modify: `test_translate_proxy.py` (append class)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_MARK`, `_MARK_END`, `protect(text) -> (str, list[str])`, `restore(text, spans) -> str`.

- [ ] **Step 1: Write the failing test**

Append to `test_translate_proxy.py`:

```python
from translate_proxy import protect, restore


class ProtectTests(unittest.TestCase):
    def test_fenced_block_verbatim(self):
        t = "Zmien ten kod:\n```python\nprint('x')\n```\nDzieki."
        p, spans = protect(t)
        self.assertEqual(p, "Zmien ten kod:\n⟦0⟧\nDzieki.")
        self.assertEqual(spans[0], "```python\nprint('x')\n```")
        self.assertEqual(restore(p, spans), t)

    def test_inline_code(self):
        t = "Uzyj `npm install -g y` i gotowe."
        p, spans = protect(t)
        self.assertEqual(p, "Uzyj ⟦0⟧ i gotowe.")
        self.assertEqual(spans[0], "`npm install -g y`")

    def test_url(self):
        t = "Zobacz https://example.com/abc i https://example.com/x?a=1."
        p, spans = protect(t)
        self.assertEqual(p, "Zobacz ⟦0⟧ i ⟦1⟧.")
        self.assertEqual(len(spans), 2)
        self.assertEqual(restore(p, spans), t)

    def test_marker_inside_fence_not_double_matched(self):
        t = "```\n`inline`\n```"
        p, spans = protect(t)
        self.assertEqual(p, "⟦0⟧")
        self.assertEqual(len(spans), 1)

    def test_restore_best_effort_missing_marker(self):
        p, spans = protect("a `x` b")
        out = restore("nie ma tu markera", spans)
        self.assertEqual(out, "nie ma tu markera")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v test_translate_proxy.ProtectTests`
Expected: FAIL with `ImportError: cannot import name 'protect'`.

- [ ] **Step 3: Write minimal implementation**

Append to `translate_proxy.py`:

```python
_MARK = "⟦"
_MARK_END = "⟧"


def protect(text: str):
    """Extract fenced code blocks, inline code and URLs into ⟦N⟧ placeholders."""
    spans = []

    def repl(m):
        spans.append(m.group(0).rstrip(".,;:!?"))
        return f"{_MARK}{len(spans) - 1}{_MARK_END}"

    t = re.sub(r"```[\s\S]*?```", repl, text)
    t = re.sub(r"`[^`\n]+`", repl, t)
    t = re.sub(r"https?://\S+(?<![.,;:!?])", repl, t)
    return t, spans


def restore(text: str, spans):
    """Put the extracted spans back where their ⟦N⟧ placeholders are."""
    for i, span in enumerate(spans):
        text = text.replace(f"{_MARK}{i}{_MARK_END}", span)
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest -v test_translate_proxy.ProtectTests`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add translate_proxy.py test_translate_proxy.py
git commit -m "feat: protect/restore code spans (fences, inline code, URLs)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Already-English input guard

**Files:**
- Modify: `translate_proxy.py` (append after `restore`)
- Modify: `test_translate_proxy.py` (append class)

**Interfaces:**
- Produces: `STOPWORDS_EN` (frozenset), `guard_skip(text, cfg) -> bool`. Test helper `_cfg()` builds a minimal `Config` (used by later tasks too — keep it copy-pastable).

- [ ] **Step 1: Write the failing test**

Append to `test_translate_proxy.py`:

```python
from translate_proxy import Config, guard_skip


class GuardTests(unittest.TestCase):
    def _cfg(self, ratio=0.3, model_lang="en"):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang="pl", user_lang_name="Polish",
                      model_lang=model_lang, model_lang_name="English",
                      translator="openrouter", translator_model="m", translator_fallback=[],
                      translate_history=True, cache_size=10, guard_ratio=ratio,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c")

    def test_english_skipped(self):
        self.assertTrue(guard_skip(
            "Please show me the list of files in this directory and tell me what they are.",
            self._cfg()))

    def test_polish_not_skipped(self):
        self.assertFalse(guard_skip(
            "Pokaz mi liste plikow w tym katalogu i powiedz co one robia.",
            self._cfg()))

    def test_polish_without_diacritics_not_skipped(self):
        self.assertFalse(guard_skip("czy mozesz sprawdzic ten plik i dac mi raport", self._cfg()))

    def test_empty_not_skipped(self):
        self.assertFalse(guard_skip("   ", self._cfg()))

    def test_non_en_model_lang_never_skips(self):
        self.assertFalse(guard_skip("This is clearly English text.", self._cfg(model_lang="de")))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v test_translate_proxy.GuardTests`
Expected: FAIL with `ImportError: cannot import name 'guard_skip'`.

- [ ] **Step 3: Write minimal implementation**

Append to `translate_proxy.py`:

```python
STOPWORDS_EN = frozenset("""a an and are as at be but by did do does for from had has have he her his
i if in is it its me my not of on or our she so that the their them then there these they this to
was we were what when where which who will with you your""".split())


def guard_skip(text: str, cfg: Config) -> bool:
    """True when text already looks like MODEL_LANG (English stopword ratio high)."""
    if cfg.model_lang != "en":
        return False
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in STOPWORDS_EN)
    # Min. 2 hits: short Polish phrases ("zrob to", "co to jest") contain "to"
    # (an English stopword) and at ratio-only would be skipped as "already English".
    return hits >= 2 and (hits / len(words)) >= cfg.guard_ratio
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest -v test_translate_proxy.GuardTests`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add translate_proxy.py test_translate_proxy.py
git commit -m "feat: already-English input guard

Co-Authored-By: Claude <noreply@anthropic.com>"
```
### Task 4: Translator client with fallback chain + cache

**Files:**
- Modify: `translate_proxy.py` (append after `guard_skip`)
- Modify: `test_translate_proxy.py` (append class)

**Interfaces:**
- Produces: `TranslatorError`, `_cache`/`_cache_lock`, `_cache_get(key)`, `_cache_put(key, val, cap)`, `clear_cache()`, `build_prompt(src_name, dst_name, direction) -> str`, `_call_backend(backend, model, prompt, user_text, cfg) -> str`, `call_with_fallback(prompt, user_text, cfg, call_backend=None) -> str|None`, `translate_text(text, cfg, src, dst, direction="ingress", call_backend=None) -> str`. All later tasks call `translate_text(text, cfg, src, dst, direction, call_backend)`; `call_backend` is injected in tests and defaults to the real `_call_backend`.

- [ ] **Step 1: Write the failing test**

Append to `test_translate_proxy.py`:

```python
from translate_proxy import Config, TranslatorError, clear_cache, translate_text


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _cfg(self, fallback=None):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang="pl", user_lang_name="Polish",
                      model_lang="en", model_lang_name="English",
                      translator="openrouter", translator_model="m",
                      translator_fallback=fallback or [("deepseek", "deepseek-v4-flash")],
                      translate_history=True, cache_size=5, guard_ratio=0.3,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c")

    def test_primary_success(self):
        def fake(backend, model, prompt, text, cfg):
            self.assertEqual(backend, "openrouter")
            return "The translated text."
        out = translate_text("Przetlumacz mnie.", self._cfg(), "pl", "en", call_backend=fake)
        self.assertEqual(out, "The translated text.")

    def test_fallback_used_when_primary_fails(self):
        calls = []

        def fake(backend, model, prompt, text, cfg):
            calls.append(backend)
            if backend == "openrouter":
                raise TranslatorError("boom")
            return "Fallback result."
        out = translate_text("cos", self._cfg(), "pl", "en", call_backend=fake)
        self.assertEqual(out, "Fallback result.")
        self.assertEqual(calls, ["openrouter", "deepseek"])

    def test_whole_chain_fails_returns_original(self):
        def fake(backend, model, prompt, text, cfg):
            raise TranslatorError("boom")
        out = translate_text("Nie tlumacz mnie.", self._cfg(), "pl", "en", call_backend=fake)
        self.assertEqual(out, "Nie tlumacz mnie.")

    def test_cache_avoids_second_call(self):
        calls = []

        def fake(backend, model, prompt, text, cfg):
            calls.append(backend)
            return "wynik"
        cfg = self._cfg()
        translate_text("ala ma kota", cfg, "pl", "pl", call_backend=fake)
        translate_text("ala ma kota", cfg, "pl", "pl", call_backend=fake)
        self.assertEqual(len(calls), 1)

    def test_code_fence_preserved_through_translation(self):
        def fake(backend, model, prompt, text, cfg):
            return text.replace("Napisz funkcje", "Write a function")  # keeps ⟦0⟧

        cfg = self._cfg()
        src = "Napisz funkcje:\n```python\nprint(1)\n```\nDzieki."
        out = translate_text(src, cfg, "pl", "en", call_backend=fake)
        self.assertIn("print(1)", out)
        self.assertIn("```python", out)

    def test_english_input_guard_skips_backend(self):
        cfg = self._cfg()

        def fake(backend, model, prompt, text, cfg):
            raise AssertionError("must not call backend for English input")

        out = translate_text("Please show me the list of files in this directory and tell me what they are.",
                             cfg, "pl", "en", call_backend=fake)
        self.assertEqual(out, "Please show me the list of files in this directory and tell me what they are.")

    def test_empty_text_passthrough(self):
        def never(*a):
            raise AssertionError("must not call backend for empty text")

        out = translate_text("", self._cfg(), "pl", "en", call_backend=never)
        out2 = translate_text("   \n  ", self._cfg(), "pl", "en", call_backend=never)
        self.assertEqual(out, "")
        self.assertEqual(out2, "   \n  ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v test_translate_proxy.TranslatorTests`
Expected: FAIL with `ImportError: cannot import name 'translate_text'`.

- [ ] **Step 3: Write minimal implementation**

Append to `translate_proxy.py`:

```python
class TranslatorError(Exception):
    pass


_cache = {}
_cache_lock = threading.Lock()


def _cache_get(key):
    with _cache_lock:
        return _cache.get(key)


def _cache_put(key, value, cap):
    with _cache_lock:
        if key not in _cache and len(_cache) >= cap:
            _cache.pop(next(iter(_cache)))  # FIFO eviction of the oldest entry
        _cache[key] = value


def clear_cache():
    with _cache_lock:
        _cache.clear()


def build_prompt(src_name, dst_name, direction="ingress"):
    guard = ("The input may be written without diacritics (e.g. Polish without ą/ę/ś) — "
             "reconstruct the intended words, do not translate letter-by-letter. ")
    if direction == "egress":
        head = f"Translate the following {src_name} text into {dst_name}, in natural, idiomatic {dst_name}. "
    else:
        head = f"Translate the following {src_name} text into {dst_name}. "
    return (f"You are a professional translator. {head}{guard}"
            "Keep every ⟦N⟧ placeholder exactly as written (they are code blocks, inline code, or "
            "URLs that must not change). Keep URLs, numbers, and technical terms unchanged. "
            "Output only the translation, nothing else.")


def _call_backend(backend, model, prompt, user_text, cfg):
    if backend not in BACKENDS:
        raise TranslatorError(f"unknown backend: {backend}")
    url, default_model, key_env = BACKENDS[backend]
    if url is None:
        url = cfg.cerebras_base.rstrip("/") + "/chat/completions"
    if not model:
        model = default_model
    key = os.environ.get(key_env) if key_env else None
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=cfg.translator_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise TranslatorError(f"{backend} failed: {e}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise TranslatorError(f"{backend} returned unexpected payload: {data}") from e


def call_with_fallback(prompt, user_text, cfg, call_backend=None):
    """Try the primary backend then each fallback; return first non-empty translation or None."""
    chain = [(cfg.translator, cfg.translator_model)] + list(cfg.translator_fallback)
    for backend, model in chain:
        try:
            out = (call_backend or _call_backend)(backend, model, prompt, user_text, cfg)
        except TranslatorError as e:
            print(f"[translate-proxy] translator {backend}/{model or 'default'} failed: {e}", file=sys.stderr)
            continue
        if out:
            return out.strip()
    return None


def translate_text(text, cfg, src, dst, direction="ingress", call_backend=None):
    """Translate one text span. Returns the original text when the whole chain fails."""
    if not text or not text.strip():
        return text
    if direction == "ingress" and guard_skip(text, cfg):
        return text
    protected, spans = protect(text)
    key = (src, dst, protected)
    cached = _cache_get(key)
    if cached is not None:
        return restore(cached, spans)
    prompt = build_prompt(lang_name(src), lang_name(dst), direction)
    out = call_with_fallback(prompt, protected, cfg, call_backend)
    if out is None:
        return text  # whole chain failed → original
    _cache_put(key, out, cfg.cache_size)
    return restore(out, spans)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest -v test_translate_proxy.TranslatorTests`
Expected: 7 tests PASS. Then run the full suite once: `python3 -m unittest -v test_translate_proxy` → 23 PASS total.

- [ ] **Step 5: Commit**

```bash
git add translate_proxy.py test_translate_proxy.py
git commit -m "feat: translator client with fallback chain + cache

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Ingress translation (Anthropic + OpenAI)

**Files:**
- Modify: `translate_proxy.py` (append after `translate_text`)
- Modify: `test_translate_proxy.py` (append class)

**Interfaces:**
- Produces: `translate_anthropic_messages(body, cfg, call_backend=None) -> dict`, `translate_openai_messages(body, cfg, call_backend=None) -> dict`. Both mutate a copy-friendly way and return the body. Later tasks (HTTP layer) call these on the parsed request before forwarding upstream.

- [ ] **Step 1: Write the failing test**

Append to `test_translate_proxy.py`:

```python
from translate_proxy import clear_cache, translate_anthropic_messages, translate_openai_messages


class IngressTests(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _cfg(self, history=True):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang="pl", user_lang_name="Polish",
                      model_lang="en", model_lang_name="English",
                      translator="openrouter", translator_model="m", translator_fallback=[],
                      translate_history=history, cache_size=5, guard_ratio=0.3,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c")

    def fake(self, calls):
        return lambda backend, model, prompt, text, cfg: calls.append(text) or ("TRANS:" + text)

    def never(self):
        return lambda *a: (_ for _ in ()).throw(AssertionError("must not call backend"))

    def test_anthropic_user_string(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "czesc swiecie"}]}
        out = translate_anthropic_messages(body, self._cfg(), call_backend=self.fake([]))
        self.assertEqual(out["messages"][0]["content"], "TRANS:czesc swiecie")

    def test_anthropic_system_and_tool_untouched(self):
        calls = []
        body = {"system": [{"type": "text", "text": "You are a robot."}],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "co to jest"},
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ls /etc"},
                ]}]}
        out = translate_anthropic_messages(body, self._cfg(), call_backend=self.fake(calls))
        self.assertEqual(out["system"][0]["text"], "You are a robot.")
        self.assertEqual(out["messages"][0]["content"][0]["text"], "TRANS:co to jest")
        self.assertEqual(out["messages"][0]["content"][1]["content"], "ls /etc")
        self.assertEqual(calls, ["co to jest"])

    def test_anthropic_code_fence_not_translated(self):
        body = {"messages": [{"role": "user", "content": "Zrob:\n```py\nx=1\n```\nok?"}]}
        out = translate_anthropic_messages(body, self._cfg(), call_backend=self.fake([]))
        self.assertIn("```py", out["messages"][0]["content"])
        self.assertIn("x=1", out["messages"][0]["content"])

    def test_anthropic_english_passthrough(self):
        body = {"messages": [{"role": "user",
                              "content": "Please show me the list of files in this directory."}]}
        out = translate_anthropic_messages(body, self._cfg(), call_backend=self.never())
        self.assertEqual(out["messages"][0]["content"],
                         "Please show me the list of files in this directory.")

    def test_anthropic_history_disabled(self):
        body = {"messages": [{"role": "assistant", "content": "Odpowiedz po polsku."}]}
        out = translate_anthropic_messages(body, self._cfg(history=False), call_backend=self.never())
        self.assertEqual(out["messages"][0]["content"], "Odpowiedz po polsku.")

    def test_anthropic_history_translated(self):
        body = {"messages": [{"role": "assistant", "content": "To jest odpowiedz modelu."}]}
        out = translate_anthropic_messages(body, self._cfg(history=True), call_backend=self.fake([]))
        self.assertEqual(out["messages"][0]["content"], "TRANS:To jest odpowiedz modelu.")

    def test_openai_user_string_and_tool_calls_untouched(self):
        body = {"messages": [
            {"role": "user", "content": "szybko policz"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "x", "type": "function",
                             "function": {"name": "calc", "arguments": "{1+1}"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "2"},
        ]}
        out = translate_openai_messages(body, self._cfg(), call_backend=self.fake([]))
        self.assertEqual(out["messages"][0]["content"], "TRANS:szybko policz")
        self.assertEqual(out["messages"][1]["tool_calls"][0]["function"]["name"], "calc")
        self.assertEqual(out["messages"][2]["content"], "2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v test_translate_proxy.IngressTests`
Expected: FAIL with `ImportError: cannot import name 'translate_anthropic_messages'`.

- [ ] **Step 3: Write minimal implementation**

Append to `translate_proxy.py`:

```python
def translate_anthropic_messages(body, cfg, call_backend=None):
    """Translate user + assistant text (USER_LANG -> MODEL_LANG) in an Anthropic /v1/messages body."""
    # system (string or list of blocks), tools, tool_choice: never touched.
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and not cfg.translate_history:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = translate_text(content, cfg, cfg.user_lang, cfg.model_lang, "ingress", call_backend)
        elif isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    blk["text"] = translate_text(blk.get("text", ""), cfg, cfg.user_lang, cfg.model_lang, "ingress", call_backend)
                # tool_use / tool_result / thinking blocks untouched
    return body


def translate_openai_messages(body, cfg, call_backend=None):
    """Translate user + assistant text in an OpenAI /v1/chat/completions body."""
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and not cfg.translate_history:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = translate_text(content, cfg, cfg.user_lang, cfg.model_lang, "ingress", call_backend)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = translate_text(part.get("text", ""), cfg, cfg.user_lang, cfg.model_lang, "ingress", call_backend)
                # tool_calls / other parts untouched
    return body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest -v test_translate_proxy.IngressTests`
Expected: 7 tests PASS. Then full suite: 30 PASS.

- [ ] **Step 5: Commit**

```bash
git add translate_proxy.py test_translate_proxy.py
git commit -m "feat: ingress translation (Anthropic + OpenAI)

Co-Authored-By: Claude <noreply@anthropic.com>"
```
### Task 6: Egress translation (streaming + non-streaming, both formats)

**Files:**
- Modify: `translate_proxy.py` (append after `translate_openai_messages`)
- Modify: `test_translate_proxy.py` (append class)

**Interfaces:**
- Produces: `_ev(event, data_dict=None, data_str=None) -> dict`, `parse_sse(raw: bytes) -> list[dict]`, `encode_sse(events) -> bytes`, `translate_anthropic_stream(events, cfg, call_backend=None) -> list[dict]`, `translate_openai_stream(chunks, cfg, call_backend=None) -> list[dict]`, `translate_anthropic_nonstream(body, cfg, call_backend=None) -> dict`, `translate_openai_nonstream(body, cfg, call_backend=None) -> dict`. Events are `{"event": str|None, "data": str}` (data is the raw SSE data line). `chunks` are parsed JSON dicts from OpenAI `data:` lines.

- [ ] **Step 1: Write the failing test**

Append to `test_translate_proxy.py`:

```python
import json

from translate_proxy import (Config, TranslatorError, clear_cache, encode_sse,
                             parse_sse, translate_anthropic_nonstream,
                             translate_anthropic_stream, translate_openai_nonstream,
                             translate_openai_stream)


class EgressTests(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _cfg(self):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang="pl", user_lang_name="Polish",
                      model_lang="en", model_lang_name="English",
                      translator="openrouter", translator_model="m", translator_fallback=[],
                      translate_history=True, cache_size=5, guard_ratio=0.3,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c")

    def fake(self, repl="Przetlumaczone."):
        return lambda backend, model, prompt, text, cfg: repl

    def never(self):
        return lambda *a: (_ for _ in ()).throw(AssertionError("must not call backend"))

    def test_anthropic_stream_text_then_tool_use(self):
        events = [
            {"event": "content_block_start", "data": json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})},
            {"event": "text_delta", "data": json.dumps({"type": "text_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}})},
            {"event": "text_delta", "data": json.dumps({"type": "text_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}})},
            {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": 0})},
            {"event": "content_block_start", "data": json.dumps({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "ls", "input": {}}})},
            {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": 1})},
            {"event": "message_delta", "data": json.dumps({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 10, "output_tokens": 5}})},
            {"event": "message_stop", "data": json.dumps({"type": "message_stop"})},
        ]
        out = translate_anthropic_stream(events, self._cfg(), call_backend=self.fake())
        texts = [json.loads(e["data"])["delta"]["text"] for e in out if e["event"] == "text_delta"]
        self.assertEqual(texts, ["…", "Przetlumaczone."])
        self.assertIn("tool_use", out[4]["data"])
        usage = [json.loads(e["data"]) for e in out if e["event"] == "message_delta"][0]
        self.assertEqual(usage["usage"]["output_tokens"], 5)

    def test_anthropic_stream_no_text_only_tools(self):
        events = [
            {"event": "content_block_start", "data": json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "bash", "input": {}}})},
            {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": 0})},
        ]
        out = translate_anthropic_stream(events, self._cfg(), call_backend=self.never())
        self.assertEqual([e for e in out if e["event"] == "text_delta"], [])

    def test_anthropic_nonstream(self):
        body = {"content": [{"type": "text", "text": "Hello world."},
                            {"type": "tool_use", "id": "t", "name": "ls", "input": {}}]}
        out = translate_anthropic_nonstream(body, self._cfg(), call_backend=self.fake())
        self.assertEqual(out["content"][0]["text"], "Przetlumaczone.")
        self.assertEqual(out["content"][1]["name"], "ls")

    def test_openai_stream(self):
        chunks = [
            {"id": "a", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"id": "a", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hi "}}]},
            {"id": "a", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "there"}}]},
            {"id": "a", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        out = translate_openai_stream(chunks, self._cfg(), call_backend=self.fake())
        contents = [c["choices"][0]["delta"].get("content") for c in out]
        self.assertEqual(contents, ["", "…", "Przetlumaczone.", None])
        self.assertTrue(any(c["choices"][0].get("finish_reason") == "stop" for c in out))

    def test_openai_nonstream(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "Hello world.",
                                         "tool_calls": [{"function": {"name": "x"}}]}}]}
        out = translate_openai_nonstream(body, self._cfg(), call_backend=self.fake())
        self.assertEqual(out["choices"][0]["message"]["content"], "Przetlumaczone.")
        self.assertEqual(out["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "x")

    def test_sse_roundtrip(self):
        raw = b'event: message_delta\ndata: {"x": 1}\n\n'
        events = parse_sse(raw)
        self.assertEqual(events[0]["event"], "message_delta")
        self.assertEqual(encode_sse(events), raw)

    def test_egress_translator_failure_returns_english(self):
        def fake(backend, model, prompt, text, cfg):
            raise TranslatorError("boom")
        body = {"content": [{"type": "text", "text": "Keep this English."}]}
        out = translate_anthropic_nonstream(body, self._cfg(), call_backend=fake)
        self.assertEqual(out["content"][0]["text"], "Keep this English.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v test_translate_proxy.EgressTests`
Expected: FAIL with `ImportError: cannot import name 'parse_sse'`.

- [ ] **Step 3: Write minimal implementation**

Append to `translate_proxy.py`:

```python
def _ev(event, data_dict=None, data_str=None):
    data = data_str if data_str is not None else json.dumps(data_dict, ensure_ascii=False)
    return {"event": event, "data": data}


def parse_sse(raw):
    """bytes -> list of {'event': str|None, 'data': str}"""
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev = None
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        events.append({"event": ev, "data": "\n".join(data_lines)})
    return events


def encode_sse(events):
    out = []
    for ev in events:
        lines = []
        if ev.get("event"):
            lines.append("event: " + ev["event"])
        lines.append("data: " + ev["data"])
        out.append("\n".join(lines) + "\n\n")
    return "".join(out).encode("utf-8")


def translate_anthropic_stream(events, cfg, call_backend=None):
    out = []
    buffer = None
    for ev in events:
        etype = ev.get("event")
        data = None
        if ev.get("data") and ev["data"].lstrip().startswith("{"):
            try:
                data = json.loads(ev["data"])
            except ValueError:
                data = None
        if etype == "content_block_start" and data and data.get("content_block", {}).get("type") == "text":
            buffer = {"index": data["index"], "text": ""}
            out.append(ev)
            if cfg.placeholder:
                d = {"type": "text_delta", "index": buffer["index"], "delta": {"type": "text_delta", "text": cfg.placeholder}}
                out.append(_ev("text_delta", d))
        elif etype == "text_delta" and buffer and data and data.get("index") == buffer["index"]:
            buffer["text"] += data.get("delta", {}).get("text", "")
        elif etype == "content_block_stop" and buffer and data and data.get("index") == buffer["index"]:
            translated = translate_text(buffer["text"], cfg, cfg.model_lang, cfg.user_lang, "egress", call_backend)
            d = {"type": "text_delta", "index": buffer["index"], "delta": {"type": "text_delta", "text": translated}}
            out.append(_ev("text_delta", d))
            out.append(ev)
            buffer = None
        else:
            out.append(ev)
    if buffer is not None:
        translated = translate_text(buffer["text"], cfg, cfg.model_lang, cfg.user_lang, "egress", call_backend)
        d = {"type": "text_delta", "index": buffer["index"], "delta": {"type": "text_delta", "text": translated}}
        out.append(_ev("text_delta", d))
        out.append(_ev("content_block_stop", {"type": "content_block_stop", "index": buffer["index"]}))
    return out


def translate_openai_stream(chunks, cfg, call_backend=None):
    meta = None
    texts = []
    role = None
    finish = None
    usage = None
    for c in chunks:
        if not isinstance(c, dict):
            continue
        choices = c.get("choices")
        if choices:
            if meta is None:
                meta = {k: v for k, v in c.items() if k != "choices"}
            for ch in choices:
                delta = ch.get("delta") or {}
                if delta.get("role"):
                    role = delta["role"]
                if delta.get("content"):
                    texts.append(delta["content"])
                if ch.get("finish_reason"):
                    finish = c
        if c.get("usage"):
            usage = c
    full = "".join(texts)
    translated = translate_text(full, cfg, cfg.model_lang, cfg.user_lang, "egress", call_backend) if full else ""

    def chunk(content, role=None, finish_reason=None):
        delta = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        ch = {"index": 0, "delta": delta}
        if finish_reason is not None:
            ch["finish_reason"] = finish_reason
        return dict(meta, choices=[ch]) if meta else {"choices": [ch]}

    out = []
    if role:
        out.append(chunk("", role=role))
    if full and cfg.placeholder:
        out.append(chunk(cfg.placeholder))
    out.append(chunk(translated))
    if finish:
        out.append(finish)
    if usage:
        out.append(usage)
    return out


def translate_anthropic_nonstream(body, cfg, call_backend=None):
    for blk in body.get("content", []):
        if isinstance(blk, dict) and blk.get("type") == "text":
            blk["text"] = translate_text(blk.get("text", ""), cfg, cfg.model_lang, cfg.user_lang, "egress", call_backend)
    return body


def translate_openai_nonstream(body, cfg, call_backend=None):
    for choice in body.get("choices", []):
        msg = choice.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            msg["content"] = translate_text(msg["content"], cfg, cfg.model_lang, cfg.user_lang, "egress", call_backend)
    return body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest -v test_translate_proxy.EgressTests`
Expected: 7 tests PASS. Then full suite: 37 PASS.

- [ ] **Step 5: Commit**

```bash
git add translate_proxy.py test_translate_proxy.py
git commit -m "feat: egress translation (streaming + non-streaming)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: HTTP relay, /health, CLI stop/health

**Files:**
- Modify: `translate_proxy.py` (append `Handler`, `_serve`, `main`)
- Modify: `test_translate_proxy.py` (append class)

**Interfaces:**
- Consumes: `translate_anthropic_messages`, `translate_openai_messages`, `translate_anthropic_stream`, `translate_openai_stream`, `translate_anthropic_nonstream`, `translate_openai_nonstream`, `parse_sse`, `encode_sse`, `Config`.
- Produces: `Handler(BaseHTTPRequestHandler)` with class attribute `call_backend = None` (test injection seam), `_serve(server, stop_file)`, `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `test_translate_proxy.py`:

```python
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from translate_proxy import Handler


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # -- mock upstream -------------------------------------------------
        class Up(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                data = b'{"up": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                path = self.path
                if path == "/v1/messages" and body.get("stream"):
                    resp = (
                        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                        'event: text_delta\ndata: {"type":"text_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
                        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
                        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":5,"output_tokens":1}}\n\n'
                        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp.encode("utf-8"))
                elif path == "/v1/messages":
                    resp = {"content": [{"type": "text", "text": "Hello non-stream."},
                                        {"type": "tool_use", "id": "t1", "name": "ls", "input": {}}],
                            "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}
                    data = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif path == "/v1/chat/completions":
                    resp = {"choices": [{"message": {"role": "assistant", "content": "Hello openai."}}],
                            "usage": {"total_tokens": 3}}
                    data = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
        t = threading.Thread(target=cls.up.serve_forever, daemon=True)
        t.start()

        # -- proxy under test ----------------------------------------------
        cls.cfg = Config(port=0, upstream=f"http://127.0.0.1:{cls.up.server_address[1]}",
                         verbose=False, placeholder="…",
                         user_lang="pl", user_lang_name="Polish",
                         model_lang="en", model_lang_name="English",
                         translator="openrouter", translator_model="m", translator_fallback=[],
                         translate_history=True, cache_size=5, guard_ratio=0.3,
                         translator_timeout=60, upstream_timeout=300, cerebras_base="c")

        def echo_backend(backend, model, prompt, text, cfg):
            return text  # echo: offline, deterministic

        class Proxy(Handler):
            pass

        Proxy.cfg = cls.cfg
        Proxy.call_backend = echo_backend
        cls.proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        t2 = threading.Thread(target=cls.proxy.serve_forever, daemon=True)
        t2.start()

        # bad-upstream proxy (for the 502 test)
        class BadProxy(Handler):
            pass

        BadCfg = Config(port=0, upstream="http://127.0.0.1:1", verbose=False, placeholder="…",
                        user_lang="pl", user_lang_name="Polish",
                        model_lang="en", model_lang_name="English",
                        translator="openrouter", translator_model="m", translator_fallback=[],
                        translate_history=True, cache_size=5, guard_ratio=0.3,
                        translator_timeout=60, upstream_timeout=2, cerebras_base="c")
        BadProxy.cfg = BadCfg
        BadProxy.call_backend = echo_backend
        cls.bad = ThreadingHTTPServer(("127.0.0.1", 0), BadProxy)
        t3 = threading.Thread(target=cls.bad.serve_forever, daemon=True)
        t3.start()

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.bad.shutdown()
        cls.up.shutdown()

    def _post(self, server, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}{path}", data=data,
                                     headers={"Content-Type": "application/json",
                                              "Anthropic-Version": "2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8"), r.headers.get("Content-Type")

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.proxy.server_address[1]}/health") as r:
            self.assertEqual(json.loads(r.read().decode("utf-8")), {"ok": True})

    def test_unknown_path_404(self):
        status, body, _ = self._post(self.proxy, "/v1/nope", {"a": 1})
        self.assertEqual(status, 404)

    def test_anthropic_stream_translated(self):
        status, body, ctype = self._post(self.proxy, "/v1/messages",
                                         {"model": "m", "stream": True,
                                          "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", ctype)
        self.assertIn("Hello", body)   # echoed translated egress text
        self.assertIn("…", body)       # placeholder
        self.assertIn("message_stop", body)

    def test_anthropic_nonstream_translated(self):
        status, body, _ = self._post(self.proxy, "/v1/messages",
                                     {"model": "m", "stream": False,
                                      "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        out = json.loads(body)
        self.assertEqual(out["content"][0]["text"], "Hello non-stream.")
        self.assertEqual(out["content"][1]["name"], "ls")

    def test_openai_nonstream_translated(self):
        status, body, _ = self._post(self.proxy, "/v1/chat/completions",
                                     {"model": "m", "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertIn("Hello openai.", body)

    def test_upstream_down_502(self):
        status, body, _ = self._post(self.bad, "/v1/messages",
                                     {"model": "m", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 502)
        self.assertIn("upstream unreachable", body)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v test_translate_proxy.HttpTests`
Expected: FAIL with `ImportError: cannot import name 'Handler'`.

- [ ] **Step 3: Write minimal implementation**

Append to `translate_proxy.py`:

```python
class Handler(BaseHTTPRequestHandler):
    cfg = None
    call_backend = None  # test injection seam: translator callable (backend, model, prompt, text, cfg) -> str
    server_version = "translate-proxy/" + VERSION

    def log_message(self, fmt, *args):
        if self.cfg and self.cfg.verbose:
            super().log_message(fmt, *args)

    def _send(self, status, headers, body):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, {"Content-Type": "application/json; charset=utf-8"}, body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        cb = self.call_backend
        try:
            if self.path == "/v1/messages":
                body = json.loads(raw.decode("utf-8"))
                body = translate_anthropic_messages(body, self.cfg, call_backend=cb)
                self._forward(self.path, body, bool(body.get("stream")), cb)
            elif self.path == "/v1/chat/completions":
                body = json.loads(raw.decode("utf-8"))
                body = translate_openai_messages(body, self.cfg, call_backend=cb)
                self._forward(self.path, body, bool(body.get("stream")), cb)
            else:
                self._json(404, {"error": "unknown path: " + self.path})
        except (ValueError, json.JSONDecodeError) as e:
            self._json(400, {"error": "bad json: " + str(e)})

    def _forward(self, path, body, stream, cb):
        url = self.cfg.upstream.rstrip("/") + path
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": self.headers.get("Accept") or "*/*",
        }
        for h in ("Authorization", "X-Api-Key", "Anthropic-Version", "Anthropic-Beta"):
            v = self.headers.get(h)
            if v:
                headers[h] = v
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.cfg.upstream_timeout)
        except urllib.error.HTTPError as e:
            self._send(e.code, {"Content-Type": e.headers.get("Content-Type") or "application/json"}, e.read())
            return
        except Exception as e:
            self._json(502, {"error": "upstream unreachable: " + str(e)})
            return
        raw = resp.read()
        ctype = resp.headers.get("Content-Type") or ""
        status = resp.status
        if stream and "text/event-stream" in ctype:
            events = parse_sse(raw)
            if self.path == "/v1/messages":
                events = translate_anthropic_stream(events, self.cfg, call_backend=cb)
                self._send(status, {"Content-Type": "text/event-stream; charset=utf-8"}, encode_sse(events))
            else:
                done = any(e.get("data") == "[DONE]" for e in events)
                parsed = [json.loads(e["data"]) for e in events if e.get("data") and e["data"] != "[DONE]"]
                translated = translate_openai_stream(parsed, self.cfg, call_backend=cb)
                out_events = [{"event": None, "data": json.dumps(c, ensure_ascii=False)} for c in translated]
                if done:
                    out_events.append({"event": None, "data": "[DONE]"})
                self._send(status, {"Content-Type": "text/event-stream; charset=utf-8"}, encode_sse(out_events))
        else:
            if self.path == "/v1/messages":
                out = translate_anthropic_nonstream(json.loads(raw.decode("utf-8")), self.cfg, call_backend=cb)
            else:
                out = translate_openai_nonstream(json.loads(raw.decode("utf-8")), self.cfg, call_backend=cb)
            self._send(status, {"Content-Type": ctype or "application/json; charset=utf-8"},
                       json.dumps(out, ensure_ascii=False).encode("utf-8"))


def _serve(server, stop_file):
    print(f"[translate-proxy] listening on http://127.0.0.1:{server.server_port}", file=sys.stderr)
    server.timeout = 0.5
    try:
        while not os.path.exists(stop_file):
            server.handle_request()
    finally:
        server.server_close()


def main(argv=None):
    cfg = load_config(argv)
    if cfg.stop_requested:
        open(cfg.stop_file, "w").close()
        print(f"[translate-proxy] stop requested (wrote {cfg.stop_file})")
        return 0
    if cfg.health_only:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cfg.port}/health", timeout=5) as r:
                print(r.read().decode("utf-8"))
                return 0 if r.status == 200 else 1
        except Exception as e:
            print(f"health check failed: {e}", file=sys.stderr)
            return 1
    Handler.cfg = cfg
    server = ThreadingHTTPServer(("127.0.0.1", cfg.port), Handler)
    _serve(server, cfg.stop_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest -v test_translate_proxy.HttpTests`
Expected: 6 tests PASS. Then full suite: 44 PASS (38 pre-task + 6 HttpTests; Task 6 added one regression test beyond the plan's 37).

Smoke: `python3 translate_proxy.py --upstream http://127.0.0.1:8799 --health` should print `{"ok": false}` (nothing listening on :8800 yet) — expected, the real service comes later. `python3 translate_proxy.py --upstream http://127.0.0.1:8799 --port 18800 --health` → `{"ok": false}`.

- [ ] **Step 5: Commit**

```bash
git add translate_proxy.py test_translate_proxy.py
git commit -m "feat: HTTP relay, /health, CLI stop/health

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Packaging and docs (README, LICENSE, .env.example, systemd)

**Files:**
- Create: `README.md`, `LICENSE`, `.env.example`, `translate-proxy.service`

**Interfaces:**
- No new code. Verifies the public-facing surface that ships to GitHub.

- [ ] **Step 1: Create `README.md`**

```markdown
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
| `OPENROUTER_API_KEY` | — | required for `openrouter` |
| `DEEPSEEK_API_KEY` | — | required for `deepseek` |

Backends:

- `openrouter` — `https://openrouter.ai/api/v1/chat/completions`, key `OPENROUTER_API_KEY`;
- `deepseek` — `https://api.deepseek.com/chat/completions`, key `DEEPSEEK_API_KEY`, model `deepseek-v4-flash`;
- `cerebras` — local gateway, `CEREBRAS_BASE` (default `http://127.0.0.1:8001/v1`), no key.

## Add your own language

Pick any pair, e.g. German → English:

```bash
USER_LANG=de USER_LANG_NAME=German MODEL_LANG=en MODEL_LANG_NAME=English \
python3 translate_proxy.py --upstream http://127.0.0.1:8799
```

Unknown language codes fall back to the code itself for the prompt. If your `MODEL_LANG`
isn't English, the already-English guard is disabled (nothing to guard against).

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
```

- [ ] **Step 2: Create `LICENSE`**

```text
MIT License

Copyright (c) 2026 kmg4ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create `.env.example`**

```bash
# --- language pair -----------------------------------------------------------
USER_LANG=pl
USER_LANG_NAME=Polish
MODEL_LANG=en
MODEL_LANG_NAME=English

# --- translator --------------------------------------------------------------
TRANSLATOR=openrouter
TRANSLATOR_MODEL=google/gemini-2.5-flash
TRANSLATOR_FALLBACK=deepseek/deepseek-v4-flash

# --- behavior ----------------------------------------------------------------
TRANSLATE_HISTORY=true
CACHE_SIZE=500
GUARD_STOPWORD_RATIO=0.3
PLACEHOLDER=…

# --- keys (keep out of git) --------------------------------------------------
OPENROUTER_API_KEY=
DEEPSEEK_API_KEY=
```

- [ ] **Step 4: Create `translate-proxy.service`**

```ini
[Unit]
Description=translate-proxy — transparent translation relay for LLM coding agents
After=network.target

[Service]
Type=simple
EnvironmentFile=/etc/translate-proxy.env
# Example: Claude Code -> translate-proxy :8800 -> existing deepseek proxy :8799
ExecStart=/usr/bin/python3 /opt/translate-proxy/translate_proxy.py --port 8800 --upstream http://127.0.0.1:8799
Restart=always
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/tmp

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 5: Verify the full suite + CLI smoke**

Run: `python3 -m unittest -v test_translate_proxy`
Expected: 43 tests PASS.

Run: `python3 translate_proxy.py --help`
Expected: argparse help with `--port`, `--upstream`, `--verbose`, `--placeholder`, `--stop`, `--health`.

Run: `git status --short` — only the new files; no `.env`, no `__pycache__`, no `.claude/`.

- [ ] **Step 6: Commit**

```bash
git add README.md LICENSE .env.example translate-proxy.service
git commit -m "docs: README, LICENSE, .env.example, systemd unit

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Post-plan notes

- The repo ships 8 commits, all green under `python3 -m unittest -v test_translate_proxy`.
- GitHub publication is a follow-up (after Kim confirms the repo name): `gh repo create translate-proxy --public --source . --push`.
- Local deployment wiring (Kim's server): one systemd unit per upstream, e.g. Claude Code via
  `--upstream http://127.0.0.1:8799`, with `ANTHROPIC_BASE_URL=http://127.0.0.1:8800` in
  `~/.claude/settings.json`.
