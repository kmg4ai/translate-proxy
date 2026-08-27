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
