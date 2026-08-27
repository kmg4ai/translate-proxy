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
