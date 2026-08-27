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
    # every-language UX: USER_LANG alone should name the language; obscure codes
    # not listed here fall back to the code itself (or set USER_LANG_NAME).
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "he": "Hebrew",
    "hi": "Hindi", "bn": "Bengali", "ur": "Urdu", "pa": "Punjabi", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "ne": "Nepali", "si": "Sinhala", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian", "ms": "Malay", "tl": "Filipino", "km": "Khmer", "my": "Burmese",
    "lo": "Lao", "mn": "Mongolian", "jv": "Javanese", "ceb": "Cebuano",
    "mi": "Maori", "haw": "Hawaiian", "sm": "Samoan", "fj": "Fijian",
    "sw": "Swahili", "am": "Amharic", "ha": "Hausa", "yo": "Yoruba", "ig": "Igbo",
    "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans", "so": "Somali", "mg": "Malagasy",
    "wo": "Wolof", "rw": "Kinyarwanda", "sn": "Shona", "st": "Sesotho", "tn": "Tswana",
    "ee": "Ewe", "ak": "Akan", "ti": "Tigrinya", "om": "Oromo",
    "fa": "Persian", "ps": "Pashto", "ku": "Kurdish", "uz": "Uzbek", "kk": "Kazakh",
    "az": "Azerbaijani", "hy": "Armenian", "ka": "Georgian", "tg": "Tajik",
    "tk": "Turkmen", "ky": "Kyrgyz",
    "cy": "Welsh", "ga": "Irish", "gd": "Scottish Gaelic", "mt": "Maltese",
    "sq": "Albanian", "mk": "Macedonian", "be": "Belarusian", "is": "Icelandic",
    "bs": "Bosnian", "ca": "Catalan", "gl": "Galician", "eu": "Basque",
    "lb": "Luxembourgish", "fy": "Frisian",
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
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "deepseek/deepseek-v4-flash", "OPENROUTER_API_KEY"),
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
    # --stop and --health only need the port; upstream is required to serve.
    if not args.upstream and not args.stop and not args.health:
        ap.error("--upstream is required (or UPSTREAM env)")

    user_lang = os.environ.get("USER_LANG", "pl").strip().lower()
    model_lang = os.environ.get("MODEL_LANG", "en").strip().lower()
    translator = os.environ.get("TRANSLATOR", "openrouter").strip().lower()
    translator_model = os.environ.get("TRANSLATOR_MODEL", BACKENDS[translator][1] if translator in BACKENDS else "")
    fallback_raw = os.environ.get("TRANSLATOR_FALLBACK", "openrouter/google/gemini-2.5-flash")

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
    """Extract fenced code blocks, inline code and URLs into ⟦N⟧ placeholders.

    A single left-to-right pass matches each code/URL span OR a literal ⟦N⟧ marker that
    is not inside one (a marker inside a fence/backtick/URL is consumed as part of that
    span). Spans are indexed strictly by appearance order, and re.sub never re-scans an
    inserted placeholder, so:
      - a literal marker inside a code/URL span stays inside the span value and
        round-trips unchanged;
      - a literal marker in prose is wrapped as a span mapping back to itself, so a later
        restore() can never substitute a real code/URL span into it — its literal index
        value never collides with a generated placeholder's index because indices are
        positional, not derived from the marker text.
    """
    spans = []

    def repl(m):
        if m.group(1) is not None:
            spans.append(m.group(0))  # literal ⟦N⟧ marker in prose → self-span back to itself
        else:
            spans.append(m.group(0).rstrip(".,;:!?"))
        return f"{_MARK}{len(spans) - 1}{_MARK_END}"

    t = re.sub(
        r"```[\s\S]*?```|`[^`\n]+`|https?://\S+(?<![.,;:!?])|⟦(\d+)⟧",
        repl, text,
    )
    return t, spans


def restore(text: str, spans):
    """Put the extracted spans back where their ⟦N⟧ placeholders are.

    Single-pass substitution so span VALUES are never re-scanned: the old loop's
    str.replace re-matched placeholders that appeared inside already-inserted span values.
    """
    def repl(m):
        i = int(m.group(1))
        if 0 <= i < len(spans):
            return spans[i]
        return m.group(0)  # unknown/missing span → leave the marker as-is (best effort)
    return re.sub(rf"{_MARK}(\d+){_MARK_END}", repl, text)


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
    if cap <= 0:
        return  # cache disabled (mirrors functools.lru_cache(maxsize=0))
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


def _translate_content(content, cfg, call_backend, src, dst, direction):
    """Translate the text parts of a content value (str or list of blocks).

    Non-text blocks (tool_use, tool_result, thinking, image_url, tool_calls)
    pass through untouched.
    """
    if isinstance(content, str):
        return translate_text(content, cfg, src, dst, direction, call_backend)
    if isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                blk["text"] = translate_text(blk.get("text", ""), cfg, src, dst, direction, call_backend)
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
        msg["content"] = _translate_content(msg.get("content"), cfg, call_backend,
                                            cfg.user_lang, cfg.model_lang, "ingress")
    return body


def translate_openai_messages(body, cfg, call_backend=None):
    """Translate user + assistant text in an OpenAI /v1/chat/completions body."""
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and not cfg.translate_history:
            continue
        msg["content"] = _translate_content(msg.get("content"), cfg, call_backend,
                                            cfg.user_lang, cfg.model_lang, "ingress")
    return body


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
    """Buffer each text content block and emit its translation at block stop.

    The placeholder ("…") is intentionally constant: SSE has no delta retraction, so the
    whole response is buffered and a single translated delta is emitted at the end; the fixed
    placeholder signals "translation in progress" while the model's English streams in.
    Non-text blocks (tool_use, etc.) pass through unchanged.
    """
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
        elif buffer and data and data.get("index") == buffer["index"] and (
            etype == "text_delta"
            or (etype == "content_block_delta" and data.get("delta", {}).get("type") == "text_delta")
        ):
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
    """Reassemble content deltas and emit the translation as a single chunk.

    The placeholder ("…") is intentionally constant: SSE has no delta retraction, so the
    whole response is buffered and the translation is emitted in one chunk; the fixed
    placeholder signals "translation in progress". tool_calls deltas pass through unchanged,
    between the text chunk and the finish chunk.
    """
    meta = None
    texts = []
    role = None
    finish = None
    usage = None
    tool_calls_chunks = []
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
                if delta.get("tool_calls") and not ch.get("finish_reason"):
                    tool_calls_chunks.append(c)
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
    if translated:
        out.append(chunk(translated))
    out.extend(tool_calls_chunks)
    if finish:
        out.append(finish)
    if usage:
        out.append(usage)
    return out


def translate_anthropic_nonstream(body, cfg, call_backend=None):
    content = body.get("content")
    if content is not None:
        body["content"] = _translate_content(content, cfg, call_backend,
                                             cfg.model_lang, cfg.user_lang, "egress")
    return body


def translate_openai_nonstream(body, cfg, call_backend=None):
    for choice in body.get("choices", []):
        msg = choice.get("message")
        if isinstance(msg, dict):
            msg["content"] = _translate_content(msg.get("content"), cfg, call_backend,
                                                cfg.model_lang, cfg.user_lang, "egress")
    return body

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
        cb = type(self).call_backend  # plain fn, not bound method
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
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
            self._json(400, {"error": "bad request: " + str(e)})

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
            with urllib.request.urlopen(req, timeout=self.cfg.upstream_timeout) as resp:
                raw = resp.read()
                ctype = resp.headers.get("Content-Type") or ""
                status = resp.status
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read()
            finally:
                e.close()
            self._send(e.code, {"Content-Type": e.headers.get("Content-Type") or "application/json"}, err_body)
            return
        except Exception as e:
            print(f"[translate-proxy] upstream unreachable: {e}", file=sys.stderr)
            self._json(502, {"error": "upstream unreachable"})
            return
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
    # A previous --stop may have left a stale stop file behind; clear it so a fresh
    # start on the same port actually serves instead of exiting 0 immediately.
    try:
        os.remove(stop_file)
    except FileNotFoundError:
        pass
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
    key_env = BACKENDS[cfg.translator][2] if cfg.translator in BACKENDS else None
    if key_env and not os.environ.get(key_env):
        print(f"[translate-proxy] warning: {key_env} is not set — {cfg.translator} translation "
              "will fall back to the next backend or pass through untranslated", file=sys.stderr)
    Handler.cfg = cfg
    server = ThreadingHTTPServer(("127.0.0.1", cfg.port), Handler)
    _serve(server, cfg.stop_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
