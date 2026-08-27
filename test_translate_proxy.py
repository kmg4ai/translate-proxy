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
