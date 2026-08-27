import os
import unittest

from translate_proxy import load_config, lang_name, parse_bool, parse_fallback

# Every env var load_config reads; ConfigTests snapshots and clears these so the
# suite is hermetic even when the developer's shell sets them (e.g. USER_LANG=pl).
CONFIG_ENV_VARS = (
    "USER_LANG", "USER_LANG_NAME", "MODEL_LANG", "MODEL_LANG_NAME",
    "TRANSLATOR", "TRANSLATOR_MODEL", "TRANSLATOR_FALLBACK", "TRANSLATE_HISTORY",
    "CACHE_SIZE", "GUARD_STOPWORD_RATIO", "PLACEHOLDER", "UPSTREAM", "PORT",
    "VERBOSE", "CEREBRAS_BASE", "TRANSLATOR_TIMEOUT", "UPSTREAM_TIMEOUT",
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = {k: os.environ.get(k) for k in CONFIG_ENV_VARS}
        for k in CONFIG_ENV_VARS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

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

    def test_stop_without_upstream(self):
        cfg = load_config(["--stop"])
        self.assertTrue(cfg.stop_requested)
        self.assertIsNone(cfg.upstream)

    def test_health_without_upstream(self):
        cfg = load_config(["--health"])
        self.assertTrue(cfg.health_only)
        self.assertIsNone(cfg.upstream)

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

    def test_marker_inside_fence_roundtrips(self):
        t = "```\n⟦7⟧\n```"
        p, spans = protect(t)
        self.assertEqual(p, "⟦0⟧")
        self.assertEqual(spans[0], t)
        self.assertEqual(restore(p, spans), t)

    def test_marker_inside_inline_code_roundtrips(self):
        t = "`⟦3⟧`"
        p, spans = protect(t)
        self.assertEqual(p, "⟦0⟧")
        self.assertEqual(spans[0], t)
        self.assertEqual(restore(p, spans), t)

    def test_standalone_prose_marker_roundtrips(self):
        t = "hello ⟦0⟧ world"
        p, spans = protect(t)
        self.assertEqual(spans, ["⟦0⟧"])
        self.assertEqual(restore(p, spans), t)

    def test_prose_marker_mixed_with_code_roundtrips(self):
        t = "⟦3⟧ and `code`"
        p, spans = protect(t)
        self.assertEqual(p, "⟦0⟧ and ⟦1⟧")
        self.assertEqual(spans, ["⟦3⟧", "`code`"])
        self.assertEqual(restore(p, spans), t)

    def test_fence_inline_url_all_roundtrip(self):
        t = "```fence``` and plain `inline` and https://x.dev text"
        p, spans = protect(t)
        self.assertEqual(p, "⟦0⟧ and plain ⟦1⟧ and ⟦2⟧ text")
        self.assertEqual(spans, ["```fence```", "`inline`", "https://x.dev"])
        self.assertEqual(restore(p, spans), t)
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

    def test_literal_marker_survives_translation(self):
        def fake(backend, model, prompt, text, cfg):
            return text  # echo the protected text back unchanged

        cfg = self._cfg()
        src = "Zobacz ⟦0⟧ i https://example.com."
        out = translate_text(src, cfg, "pl", "en", call_backend=fake)
        # the literal ⟦0⟧ must survive intact — not substituted with the URL span
        self.assertEqual(out, src)

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

    def test_anthropic_tool_use_and_thinking_untouched(self):
        calls = []
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "co to jest"},
            {"type": "thinking", "thinking": "Mysle po angielsku."},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
        ]}]}
        out = translate_anthropic_messages(body, self._cfg(), call_backend=self.fake(calls))
        self.assertEqual(out["messages"][0]["content"][0]["text"], "TRANS:co to jest")
        self.assertEqual(out["messages"][0]["content"][1]["thinking"], "Mysle po angielsku.")
        self.assertEqual(out["messages"][0]["content"][2]["input"]["cmd"], "ls")
        self.assertEqual(calls, ["co to jest"])

    def test_openai_content_list_translates_only_text(self):
        calls = []
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "co jest na obrazku"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]}]}
        out = translate_openai_messages(body, self._cfg(), call_backend=self.fake(calls))
        self.assertEqual(out["messages"][0]["content"][0]["text"], "TRANS:co jest na obrazku")
        self.assertEqual(out["messages"][0]["content"][1]["image_url"]["url"], "data:image/png;base64,AAA")
        self.assertEqual(calls, ["co jest na obrazku"])

    def test_openai_assistant_history_translated(self):
        body = {"messages": [{"role": "assistant", "content": "To jest historia."}]}
        out = translate_openai_messages(body, self._cfg(history=True), call_backend=self.fake([]))
        self.assertEqual(out["messages"][0]["content"], "TRANS:To jest historia.")

    def test_openai_assistant_history_disabled(self):
        body = {"messages": [{"role": "assistant", "content": "To jest historia."}]}
        out = translate_openai_messages(body, self._cfg(history=False), call_backend=self.never())
        self.assertEqual(out["messages"][0]["content"], "To jest historia.")

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

    def test_openai_stream_tool_calls_preserved(self):
        chunks = [
            {"id": "a", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"id": "a", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "t1", "function": {"name": "ls", "arguments": ""}}]}}]},
            {"id": "a", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        out = translate_openai_stream(chunks, self._cfg(), call_backend=self.never())
        tool_deltas = [c["choices"][0]["delta"].get("tool_calls") for c in out]
        self.assertTrue(any(td is not None for td in tool_deltas))
        self.assertTrue(any(c["choices"][0].get("finish_reason") == "tool_calls" for c in out))
        # exactly one tool_calls chunk, and it must precede the finish_reason chunk
        tool_chunks = [c for c in out if c["choices"][0]["delta"].get("tool_calls") is not None]
        self.assertEqual(len(tool_chunks), 1)
        finish_idx = next(i for i, c in enumerate(out) if c["choices"][0].get("finish_reason"))
        self.assertIn(tool_chunks[0], out[:finish_idx])
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
                elif path == "/v1/chat/completions" and body.get("stream"):
                    resp = (
                        'data: {"id":"a","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
                        'data: {"id":"a","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello "}}]}\n\n'
                        'data: {"id":"a","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"streaming"}}]}\n\n'
                        'data: {"id":"a","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                        'data: [DONE]\n\n'
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp.encode("utf-8"))
                elif path == "/v1/chat/completions" and body.get("trigger_error"):
                    resp = '{"error": {"message": "upstream exploded"}}'
                    data = resp.encode("utf-8")
                    self.send_response(500)
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
        cls.proxy.server_close()
        cls.bad.server_close()
        cls.up.server_close()

    def _post(self, server, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}{path}", data=data,
                                     headers={"Content-Type": "application/json",
                                              "Anthropic-Version": "2023-06-01"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode("utf-8"), r.headers.get("Content-Type")
        except urllib.request.HTTPError as e:  # 4xx/5xx: urlopen raises; surface status+body
            code, body, ctype = e.code, e.read().decode("utf-8"), e.headers.get("Content-Type")
            e.close()
            return code, body, ctype

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

    def test_openai_stream_translated(self):
        status, body, ctype = self._post(self.proxy, "/v1/chat/completions",
                                         {"model": "m", "stream": True,
                                          "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", ctype)
        self.assertIn("Hello streaming", body)  # echoed translated egress text
        self.assertIn("…", body)                # placeholder
        self.assertIn("[DONE]", body)

    def test_upstream_http_error_passthrough(self):
        status, body, _ = self._post(self.proxy, "/v1/chat/completions",
                                     {"model": "m", "trigger_error": True,
                                      "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 500)
        self.assertIn("upstream exploded", body)

    def test_upstream_down_502(self):
        status, body, _ = self._post(self.bad, "/v1/messages",
                                     {"model": "m", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 502)
        self.assertIn("upstream unreachable", body)
