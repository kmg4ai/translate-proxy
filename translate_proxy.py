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


STOPWORDS_EN = frozenset("""a an and are as at be but by did do does for from had has have he her his
i if in is it its me my not of on or our she so that the their them then there these they this to
was we were what when where which who will with you your""".split())


def guard_skip(text: str, cfg: Config) -> bool:
    """True when text already looks like MODEL_LANG (English stopword ratio high).

    Requires at least 2 English-stopword hits in addition to the ratio. Short Polish
    phrases like "zrob to" and "co to jest" contain "to" (an English stopword) and would
    otherwise be wrongly skipped by the ratio alone; pasted English (a long block with
    many hits) is still detected.
    """
    if cfg.model_lang != "en":
        return False
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in STOPWORDS_EN)
    return hits >= 2 and (hits / len(words)) >= cfg.guard_ratio


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


def _translate_content(content, cfg, call_backend):
    """Translate the text parts of a message's content (str or list of blocks).

    Non-text blocks (tool_use, tool_result, thinking, image_url, tool_calls)
    pass through untouched.
    """
    if isinstance(content, str):
        return translate_text(content, cfg, cfg.user_lang, cfg.model_lang, "ingress", call_backend)
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                blk["text"] = translate_text(blk.get("text", ""), cfg, cfg.user_lang, cfg.model_lang, "ingress", call_backend)
        return content
    return content


def translate_anthropic_messages(body, cfg, call_backend=None):
    """Translate user + assistant text (USER_LANG -> MODEL_LANG) in an Anthropic /v1/messages body."""
    # system (string or list of blocks), tools, tool_choice: never touched.
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and not cfg.translate_history:
            continue
        msg["content"] = _translate_content(msg.get("content"), cfg, call_backend)
    return body


def translate_openai_messages(body, cfg, call_backend=None):
    """Translate user + assistant text in an OpenAI /v1/chat/completions body."""
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and not cfg.translate_history:
            continue
        msg["content"] = _translate_content(msg.get("content"), cfg, call_backend)
    return body
