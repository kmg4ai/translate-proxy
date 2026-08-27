import os
import tempfile
import unittest

from translate_proxy import (
    decrypt_env_file,
    encrypt_env_file,
    lang_name,
    load_config,
    load_dotenv,
    parse_bool,
    parse_fallback,
)

# Every env var load_config reads; ConfigTests snapshots and clears these so the
# suite is hermetic even when the developer's shell sets them (e.g. USER_LANG=pl).
CONFIG_ENV_VARS = (
    "USER_LANG", "USER_LANG_NAME", "MODEL_LANG", "MODEL_LANG_NAME",
    "TRANSLATOR", "TRANSLATOR_MODEL", "TRANSLATOR_FALLBACK", "TRANSLATE_HISTORY",
    "CACHE_SIZE", "GUARD_STOPWORD_RATIO", "PLACEHOLDER", "UPSTREAM", "PORT",
    "VERBOSE", "CEREBRAS_BASE", "TRANSLATOR_TIMEOUT", "UPSTREAM_TIMEOUT",
    "SKIP_TRANSLATION_MODELS", "GOOGLE_TRANSLATE_API_KEY", "LIBRETRANSLATE_BASE",
    "LIBRETRANSLATE_API_KEY", "DEEPL_API_KEY", "DEEPL_API_BASE",
    "AZURE_TRANSLATOR_KEY", "AZURE_TRANSLATOR_REGION",
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
        self.assertEqual(cfg.translator, "deepseek")
        self.assertEqual(cfg.translator_model, "deepseek-v4-flash")
        self.assertEqual(cfg.translator_fallback, [("openrouter", "deepseek/deepseek-v4-flash")])
        self.assertEqual(cfg.skip_translation_models, ["deepseek-v4-flash", "deepseek-pro"])
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

    def test_skip_models_override(self):
        os.environ["SKIP_TRANSLATION_MODELS"] = "deepseek-chat, claude-sonnet-4"
        try:
            cfg = load_config(["--upstream", "u"])
            self.assertEqual(cfg.skip_translation_models, ["deepseek-chat", "claude-sonnet-4"])
        finally:
            os.environ.pop("SKIP_TRANSLATION_MODELS", None)

    def test_skip_models_empty_means_translate_all(self):
        os.environ["SKIP_TRANSLATION_MODELS"] = ""
        try:
            cfg = load_config(["--upstream", "u"])
            self.assertEqual(cfg.skip_translation_models, [])
        finally:
            os.environ.pop("SKIP_TRANSLATION_MODELS", None)

    def test_libretranslate_base_default(self):
        cfg = load_config(["--upstream", "u"])
        self.assertEqual(cfg.libretranslate_base, "http://127.0.0.1:5000")

    def test_libretranslate_base_override(self):
        os.environ["LIBRETRANSLATE_BASE"] = "http://192.168.1.10:5000"
        try:
            cfg = load_config(["--upstream", "u"])
            self.assertEqual(cfg.libretranslate_base, "http://192.168.1.10:5000")
        finally:
            os.environ.pop("LIBRETRANSLATE_BASE", None)

    def test_machine_backend_no_implicit_fallback(self):
        # user picks a free machine-translation backend -> NO implicit fallback:
        # the choice is authoritative, nothing silently switches on failure
        for backend in ("google", "libretranslate", "deepl", "azure"):
            os.environ["TRANSLATOR"] = backend
            try:
                cfg = load_config(["--upstream", "u"])
                self.assertEqual(cfg.translator_fallback, [])
            finally:
                os.environ.pop("TRANSLATOR", None)

    def test_llm_backend_keeps_implicit_fallback(self):
        os.environ["TRANSLATOR"] = "deepseek"
        try:
            cfg = load_config(["--upstream", "u"])
            self.assertEqual(cfg.translator_fallback, [("openrouter", "deepseek/deepseek-v4-flash")])
        finally:
            os.environ.pop("TRANSLATOR", None)

    def test_machine_backend_explicit_fallback_honored(self):
        # explicit TRANSLATOR_FALLBACK always wins, even for a machine backend
        os.environ["TRANSLATOR"] = "google"
        os.environ["TRANSLATOR_FALLBACK"] = "deepseek/deepseek-chat"
        try:
            cfg = load_config(["--upstream", "u"])
            self.assertEqual(cfg.translator_fallback, [("deepseek", "deepseek-chat")])
        finally:
            os.environ.pop("TRANSLATOR", None)
            os.environ.pop("TRANSLATOR_FALLBACK", None)

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

    def test_lang_name_world_coverage(self):
        # any arbitrary person's language should name itself from USER_LANG alone
        for code, name in [("ja", "Japanese"), ("zh", "Chinese"), ("ko", "Korean"),
                           ("ar", "Arabic"), ("hi", "Hindi"), ("vi", "Vietnamese"),
                           ("sw", "Swahili"), ("fa", "Persian")]:
            self.assertEqual(lang_name(code), name)

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
from translate_proxy import Config, guard_skip, should_translate_model


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


class SkipModelTests(unittest.TestCase):
    def _cfg(self, skip=None):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang="pl", user_lang_name="Polish",
                      model_lang="en", model_lang_name="English",
                      translator="openrouter", translator_model="m", translator_fallback=[],
                      translate_history=True, cache_size=5, guard_ratio=0.3,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c",
                      skip_translation_models=skip)

    def test_default_skip_list(self):
        cfg = self._cfg()
        self.assertEqual(cfg.skip_translation_models, ["deepseek-v4-flash", "deepseek-pro"])

    def test_cheap_model_skipped(self):
        self.assertFalse(should_translate_model("deepseek-v4-flash", self._cfg()))

    def test_deepseek_pro_skipped(self):
        self.assertFalse(should_translate_model("deepseek-pro", self._cfg()))

    def test_provider_prefixed_cheap_skipped(self):
        self.assertFalse(should_translate_model("deepseek/deepseek-v4-flash", self._cfg()))

    def test_other_model_translated(self):
        self.assertTrue(should_translate_model("claude-sonnet-4", self._cfg()))
        self.assertTrue(should_translate_model("gpt-4o", self._cfg()))

    def test_missing_model_translated(self):
        self.assertTrue(should_translate_model(None, self._cfg()))
        self.assertTrue(should_translate_model("", self._cfg()))

    def test_case_insensitive(self):
        self.assertFalse(should_translate_model("DEEPSEEK-V4-FLASH", self._cfg()))

    def test_custom_skip_list(self):
        cfg = self._cfg(skip=["deepseek-chat"])
        self.assertTrue(should_translate_model("deepseek-v4-flash", cfg))
        self.assertFalse(should_translate_model("deepseek-chat", cfg))

    def test_empty_skip_list_translates_all(self):
        self.assertTrue(should_translate_model("deepseek-v4-flash", self._cfg(skip=[])))
from translate_proxy import Config, TranslatorError, clear_cache, translate_text


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _cfg(self, fallback=None):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang="pl", user_lang_name="Polish",
                      model_lang="en", model_lang_name="English",
                      translator="openrouter", translator_model="m",
                      translator_fallback=fallback or [("openrouter", "deepseek/deepseek-v4-flash")],
                      translate_history=True, cache_size=5, guard_ratio=0.3,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c")

    def test_primary_success(self):
        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
            self.assertEqual(backend, "openrouter")
            return "The translated text."
        out = translate_text("Przetlumacz mnie.", self._cfg(), "pl", "en", call_backend=fake)
        self.assertEqual(out, "The translated text.")

    def test_fallback_used_when_primary_fails(self):
        calls = []

        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
            calls.append((backend, model))
            if model == "m":  # primary translator model
                raise TranslatorError("boom")
            return "Fallback result."
        out = translate_text("cos", self._cfg(), "pl", "en", call_backend=fake)
        self.assertEqual(out, "Fallback result.")
        self.assertEqual(calls, [("openrouter", "m"), ("openrouter", "deepseek/deepseek-v4-flash")])

    def test_whole_chain_fails_returns_original(self):
        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
            raise TranslatorError("boom")
        out = translate_text("Nie tlumacz mnie.", self._cfg(), "pl", "en", call_backend=fake)
        self.assertEqual(out, "Nie tlumacz mnie.")

    def test_cache_avoids_second_call(self):
        calls = []

        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
            calls.append(backend)
            return "wynik"
        cfg = self._cfg()
        translate_text("ala ma kota", cfg, "pl", "pl", call_backend=fake)
        translate_text("ala ma kota", cfg, "pl", "pl", call_backend=fake)
        self.assertEqual(len(calls), 1)

    def test_code_fence_preserved_through_translation(self):
        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
            return text.replace("Napisz funkcje", "Write a function")  # keeps ⟦0⟧

        cfg = self._cfg()
        src = "Napisz funkcje:\n```python\nprint(1)\n```\nDzieki."
        out = translate_text(src, cfg, "pl", "en", call_backend=fake)
        self.assertIn("print(1)", out)
        self.assertIn("```python", out)

    def test_literal_marker_survives_translation(self):
        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
            return text  # echo the protected text back unchanged

        cfg = self._cfg()
        src = "Zobacz ⟦0⟧ i https://example.com."
        out = translate_text(src, cfg, "pl", "en", call_backend=fake)
        # the literal ⟦0⟧ must survive intact — not substituted with the URL span
        self.assertEqual(out, src)

    def test_english_input_guard_skips_backend(self):
        cfg = self._cfg()

        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
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

import json
import urllib.parse
from unittest import mock

from translate_proxy import call_with_fallback


class MachineBackendTests(unittest.TestCase):
    """Google Cloud Translation + LibreTranslate dispatch through the real
    _call_backend, with urllib.request.urlopen mocked (no network)."""

    def setUp(self):
        clear_cache()
        self._keys = {k: os.environ.get(k) for k in (
            "GOOGLE_TRANSLATE_API_KEY", "LIBRETRANSLATE_API_KEY",
            "DEEPL_API_KEY", "DEEPL_API_BASE", "AZURE_TRANSLATOR_KEY",
            "AZURE_TRANSLATOR_REGION")}
        for k in self._keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._keys.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _cfg(self, translator, fallback=None, libretranslate_base=None):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang="pl", user_lang_name="Polish",
                      model_lang="en", model_lang_name="English",
                      translator=translator, translator_model="",
                      translator_fallback=fallback or [],
                      translate_history=True, cache_size=5, guard_ratio=0.3,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c",
                      libretranslate_base=libretranslate_base)

    def _fake_urlopen(self, responses):
        """Return (fake, calls). responses: list of dict (-> JSON reply) or
        Exception (-> raise). calls fills with {url, data, headers} per request."""
        state = {"i": 0, "calls": []}

        class FakeResp:
            def __init__(self, payload):
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self._payload

        def fake(req, timeout=None):
            raw = req.data.decode("utf-8") if req.data else None
            try:
                data = json.loads(raw) if raw is not None else None
            except (ValueError, TypeError):
                data = raw  # non-JSON body (e.g. DeepL form-encoded)
            # note: urllib normalizes header keys (capitalize()); store lowercase
            state["calls"].append({
                "url": req.full_url,
                "data": data,
                "headers": {k.lower(): v for k, v in req.headers.items()},
            })
            item = responses[min(state["i"], len(responses) - 1)]
            state["i"] += 1
            if isinstance(item, Exception):
                raise item
            return FakeResp(json.dumps(item).encode("utf-8"))

        return fake, state["calls"]

    def test_google_success(self):
        os.environ["GOOGLE_TRANSLATE_API_KEY"] = "secret-key"
        fake, calls = self._fake_urlopen([{"data": {"translations": [{"translatedText": "Hello"}]}}])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("google"), src="pl", dst="en")
        self.assertEqual(out, "Hello")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["url"].startswith(
            "https://translation.googleapis.com/language/translate/v2?key=secret-key"))
        self.assertEqual(calls[0]["data"]["q"], "Czesc")
        self.assertEqual(calls[0]["data"]["source"], "pl")
        self.assertEqual(calls[0]["data"]["target"], "en")
        self.assertEqual(calls[0]["data"]["format"], "text")

    def test_google_missing_key_no_network(self):
        def boom(*a):
            raise AssertionError("urlopen must not be called without an API key")

        with mock.patch("translate_proxy.urllib.request.urlopen", boom):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("google"), src="pl", dst="en")
        self.assertIsNone(out)  # whole chain failed -> None (translate_text keeps the original)

    def test_google_failure_falls_back_to_llm(self):
        os.environ["GOOGLE_TRANSLATE_API_KEY"] = "secret-key"
        fake, calls = self._fake_urlopen([
            OSError("google down"),
            {"choices": [{"message": {"content": "Hallo"}}]},
        ])
        cfg = self._cfg("google", fallback=[("openrouter", "deepseek/deepseek-v4-flash")])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc", cfg, src="pl", dst="en")
        self.assertEqual(out, "Hallo")
        self.assertEqual(len(calls), 2)

    def test_libretranslate_success(self):
        fake, calls = self._fake_urlopen([{"translatedText": "Hallo"}])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc",
                                     self._cfg("libretranslate"), src="pl", dst="de")
        self.assertEqual(out, "Hallo")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:5000/translate")
        self.assertEqual(calls[0]["data"]["source"], "pl")
        self.assertEqual(calls[0]["data"]["target"], "de")
        self.assertNotIn("api_key", calls[0]["data"])

    def test_libretranslate_custom_base_trailing_slash(self):
        fake, calls = self._fake_urlopen([{"translatedText": "Hallo"}])
        cfg = self._cfg("libretranslate", libretranslate_base="http://192.168.1.10:5000/")
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc", cfg, src="pl", dst="de")
        self.assertEqual(out, "Hallo")
        self.assertEqual(calls[0]["url"], "http://192.168.1.10:5000/translate")

    def test_libretranslate_api_key_in_payload(self):
        os.environ["LIBRETRANSLATE_API_KEY"] = "sekret"
        fake, calls = self._fake_urlopen([{"translatedText": "Hallo"}])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc",
                                     self._cfg("libretranslate"), src="pl", dst="de")
        self.assertEqual(out, "Hallo")
        self.assertEqual(calls[0]["data"]["api_key"], "sekret")

    def test_deepl_success(self):
        os.environ["DEEPL_API_KEY"] = "deep-key"
        fake, calls = self._fake_urlopen([
            {"translations": [{"detected_source_language": "PL", "text": "Hello"}]}])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("deepl"), src="pl", dst="en")
        self.assertEqual(out, "Hello")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["url"], "https://api-free.deepl.com/v2/translate")
        self.assertEqual(calls[0]["headers"]["authorization"], "DeepL-Auth-Key deep-key")
        body = urllib.parse.parse_qs(calls[0]["data"])  # form-encoded, stored raw
        self.assertEqual(body["text"], ["Czesc"])
        self.assertEqual(body["source_lang"], ["PL"])   # DeepL wants uppercase codes
        self.assertEqual(body["target_lang"], ["EN"])

    def test_deepl_custom_base(self):
        os.environ["DEEPL_API_KEY"] = "deep-key"
        os.environ["DEEPL_API_BASE"] = "https://api.deepl.com/v2"
        fake, calls = self._fake_urlopen([{"translations": [{"text": "Hello"}]}])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("deepl"), src="pl", dst="en")
        self.assertEqual(out, "Hello")
        self.assertEqual(calls[0]["url"], "https://api.deepl.com/v2/translate")

    def test_deepl_missing_key_no_network(self):
        def boom(*a):
            raise AssertionError("urlopen must not be called without a DeepL key")

        with mock.patch("translate_proxy.urllib.request.urlopen", boom):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("deepl"), src="pl", dst="en")
        self.assertIsNone(out)

    def test_azure_success(self):
        os.environ["AZURE_TRANSLATOR_KEY"] = "az-key"
        os.environ["AZURE_TRANSLATOR_REGION"] = "westeurope"
        fake, calls = self._fake_urlopen([
            [{"translations": [{"text": "Hello", "to": "en"}]}]])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("azure"), src="pl", dst="en")
        self.assertEqual(out, "Hello")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["url"],
                         "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&from=pl&to=en")
        self.assertEqual(calls[0]["headers"]["ocp-apim-subscription-key"], "az-key")
        self.assertEqual(calls[0]["headers"]["ocp-apim-subscription-region"], "westeurope")
        self.assertEqual(calls[0]["data"], [{"Text": "Czesc"}])  # JSON array body

    def test_azure_no_region_ok(self):
        os.environ["AZURE_TRANSLATOR_KEY"] = "az-key"
        fake, calls = self._fake_urlopen([[{"translations": [{"text": "Hello", "to": "en"}]}]])
        with mock.patch("translate_proxy.urllib.request.urlopen", fake):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("azure"), src="pl", dst="en")
        self.assertEqual(out, "Hello")
        self.assertNotIn("ocp-apim-subscription-region", calls[0]["headers"])

    def test_azure_missing_key_no_network(self):
        def boom(*a):
            raise AssertionError("urlopen must not be called without an Azure key")

        with mock.patch("translate_proxy.urllib.request.urlopen", boom):
            out = call_with_fallback("translate prompt", "Czesc", self._cfg("azure"), src="pl", dst="en")
        self.assertIsNone(out)

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
        return lambda backend, model, prompt, text, cfg, src=None, dst=None: calls.append(text) or ("TRANS:" + text)

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
        return lambda backend, model, prompt, text, cfg, src=None, dst=None: repl

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

    def test_anthropic_stream_content_block_delta_text(self):
        # deepseek-proxy-style upstream emits text as event: content_block_delta
        # with delta.type == "text_delta" (not event: text_delta). It must be
        # buffered and translated, never passed through in the model's language;
        # thinking_delta (same envelope, different delta.type) passes through.
        events = [
            {"event": "content_block_start", "data": json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}})},
            {"event": "content_block_delta", "data": json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Hmm"}})},
            {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": 0})},
            {"event": "content_block_start", "data": json.dumps({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}})},
            {"event": "content_block_delta", "data": json.dumps({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello "}})},
            {"event": "content_block_delta", "data": json.dumps({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "world"}})},
            {"event": "content_block_stop", "data": json.dumps({"type": "content_block_stop", "index": 1})},
            {"event": "message_delta", "data": json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 10, "output_tokens": 5}})},
            {"event": "message_stop", "data": json.dumps({"type": "message_stop"})},
        ]
        out = translate_anthropic_stream(events, self._cfg(), call_backend=self.fake())
        texts = [json.loads(e["data"])["delta"]["text"] for e in out if e["event"] == "text_delta"]
        self.assertEqual(texts, ["…", "Przetlumaczone."])
        # raw English content_block_delta must not reach the client
        self.assertFalse(any(e["event"] == "content_block_delta" and json.loads(e["data"])["delta"].get("type") == "text_delta" for e in out))
        # thinking delta passes through untouched
        thinking = [json.loads(e["data"])["delta"]["thinking"] for e in out if e["event"] == "content_block_delta"]
        self.assertEqual(thinking, ["Hmm"])

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
        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
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
                cls.up_last = body  # capture for pass-through tests
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

        def echo_backend(backend, model, prompt, text, cfg, src=None, dst=None):
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

    # --- pass-through: cheap main models skip translation --------------------
    def test_cheap_anthropic_stream_passthrough(self):
        status, body, ctype = self._post(self.proxy, "/v1/messages",
                                         {"model": "deepseek-v4-flash", "stream": True,
                                          "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertNotIn("…", body)                              # no placeholder
        self.assertIn("Hello", body)                             # raw upstream text
        self.assertEqual(self.up_last["messages"][0]["content"], "Czesc")  # ingress untouched

    def test_cheap_provider_prefixed_passthrough(self):
        status, body, _ = self._post(self.proxy, "/v1/messages",
                                     {"model": "deepseek/deepseek-v4-flash", "stream": False,
                                      "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertIn("Hello non-stream.", body)
        self.assertEqual(self.up_last["messages"][0]["content"], "Czesc")

    def test_cheap_openai_nonstream_passthrough(self):
        status, body, _ = self._post(self.proxy, "/v1/chat/completions",
                                     {"model": "deepseek-pro",
                                      "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertIn("Hello openai.", body)
        self.assertEqual(self.up_last["messages"][0]["content"], "Czesc")

    def test_deepseek_pro_stream_passthrough(self):
        status, body, _ = self._post(self.proxy, "/v1/chat/completions",
                                     {"model": "deepseek-pro", "stream": True,
                                      "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertNotIn("…", body)      # chunks not buffered for translation
        self.assertIn("Hello ", body)    # raw upstream chunks untouched
        self.assertIn("streaming", body)
        self.assertIn("[DONE]", body)

    def test_other_model_still_translated(self):
        # claude-sonnet-4 is NOT on the skip list -> the buffered translate path
        # runs, which is observable as the "…" placeholder in stream mode (same
        # skip flag also drives the ingress translation).
        status, body, _ = self._post(self.proxy, "/v1/messages",
                                     {"model": "claude-sonnet-4", "stream": True,
                                      "messages": [{"role": "user", "content": "Czesc"}]})
        self.assertEqual(status, 200)
        self.assertIn("…", body)
        self.assertEqual(self.up_last["model"], "claude-sonnet-4")


class DotenvTests(unittest.TestCase):
    """load_dotenv: plaintext .env parsing (pure stdlib, no crypto needed)."""

    def _write(self, text):
        fd, path = tempfile.mkstemp(prefix="dotenv-", suffix=".env")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_missing_file_is_noop(self):
        load_dotenv("/definitely/not/here/.env")  # must not raise

    def test_directory_is_noop(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        load_dotenv(d)  # not a file -> no-op

    def test_loads_simple_pairs(self):
        path = self._write("FOO=bar\nBAZ=qux\n")
        for k in ("FOO", "BAZ"):
            os.environ.pop(k, None)
        load_dotenv(path)
        self.assertEqual(os.environ["FOO"], "bar")
        self.assertEqual(os.environ["BAZ"], "qux")

    def test_real_env_wins(self):
        os.environ["FOO"] = "shell"
        self.addCleanup(os.environ.pop, "FOO", None)
        load_dotenv(self._write("FOO=dotenv\n"))
        self.assertEqual(os.environ["FOO"], "shell")

    def test_comments_export_quotes_and_trailing_comment(self):
        for k in ("X", "Y", "Z", "H", "E"):
            os.environ.pop(k, None)
        path = self._write(
            "# a comment\n"
            "\n"
            "export X=1\n"
            'Y="two words"\n'
            "Z=value # trailing comment\n"
            'H="#quoted hash"\n'
            "E=\n"
        )
        load_dotenv(path)
        self.assertEqual(os.environ["X"], "1")
        self.assertEqual(os.environ["Y"], "two words")
        self.assertEqual(os.environ["Z"], "value")
        self.assertEqual(os.environ["H"], "#quoted hash")
        self.assertEqual(os.environ.get("E"), "")

    def test_shell_value_not_overwritten_by_empty(self):
        os.environ["E"] = "keepme"
        self.addCleanup(os.environ.pop, "E", None)
        load_dotenv(self._write("E=\n"))
        self.assertEqual(os.environ["E"], "keepme")


try:
    import cryptography  # noqa: F401
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


@unittest.skipUnless(_HAS_CRYPTO, "cryptography not installed")
class EnvEncTests(unittest.TestCase):
    """Optional encrypted .env: .env.enc (Fernet + PBKDF2) round-trip."""

    def _dir(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        return d

    def _with_passphrase(self, pw="test-passphrase"):
        os.environ["ENV_PASSPHRASE"] = pw
        self.addCleanup(os.environ.pop, "ENV_PASSPHRASE", None)

    def test_round_trip_and_ciphertext_hides_secret(self):
        d = self._dir()
        plain = os.path.join(d, ".env")
        enc = os.path.join(d, ".env.enc")
        out = os.path.join(d, ".env.out")
        self._with_passphrase()
        with open(plain, "w", encoding="utf-8") as f:
            f.write("AZURE_TRANSLATOR_KEY=sekret\nTRANSLATOR=azure\n")
        encrypt_env_file(plain, enc)
        self.assertTrue(os.path.isfile(enc))
        with open(enc, encoding="utf-8") as f:
            self.assertNotIn("sekret", f.read())  # secret never in ciphertext
        decrypt_env_file(enc, out)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), "AZURE_TRANSLATOR_KEY=sekret\nTRANSLATOR=azure\n")
        with open(plain, encoding="utf-8") as f:  # source never auto-deleted
            self.assertEqual(f.read(), "AZURE_TRANSLATOR_KEY=sekret\nTRANSLATOR=azure\n")

    def test_wrong_passphrase_rejected(self):
        d = self._dir()
        plain = os.path.join(d, ".env")
        enc = os.path.join(d, ".env.enc")
        self._with_passphrase("right")
        with open(plain, "w", encoding="utf-8") as f:
            f.write("KEY=v\n")
        encrypt_env_file(plain, enc)
        os.environ["ENV_PASSPHRASE"] = "wrong"
        with self.assertRaises(ValueError):
            decrypt_env_file(enc, os.path.join(d, "out.env"))

    def test_load_dotenv_enc_in_memory(self):
        from translate_proxy import _load_dotenv_enc
        d = self._dir()
        plain = os.path.join(d, ".env")
        enc = os.path.join(d, ".env.enc")
        self._with_passphrase("pw")
        with open(plain, "w", encoding="utf-8") as f:
            f.write("AZURE_TRANSLATOR_KEY=sekret\n")
        encrypt_env_file(plain, enc)
        os.environ.pop("AZURE_TRANSLATOR_KEY", None)
        os.unlink(plain)  # no plaintext left on disk, like the real workflow
        _load_dotenv_enc(enc)
        self.assertEqual(os.environ["AZURE_TRANSLATOR_KEY"], "sekret")

    def test_load_dotenv_enc_wrong_passphrase_skips(self):
        from translate_proxy import _load_dotenv_enc
        d = self._dir()
        plain = os.path.join(d, ".env")
        enc = os.path.join(d, ".env.enc")
        self._with_passphrase("pw")
        with open(plain, "w", encoding="utf-8") as f:
            f.write("K=1\n")
        encrypt_env_file(plain, enc)
        os.environ.pop("K", None)
        os.environ["ENV_PASSPHRASE"] = "wrong"
        _load_dotenv_enc(enc)  # must not raise, must not load
        self.assertNotIn("K", os.environ)

    def test_missing_source_file(self):
        d = self._dir()
        self._with_passphrase()
        with self.assertRaises(FileNotFoundError):
            encrypt_env_file(os.path.join(d, "nope.env"), os.path.join(d, ".env.enc"))


class EnvTemplateTests(unittest.TestCase):
    """_ensure_env_file: first run auto-creates .env from the shipped template."""

    def _dir(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        return d

    def test_creates_from_template_when_missing(self):
        from translate_proxy import _ensure_env_file
        d = self._dir()
        created = _ensure_env_file(".env", dest_dir=d)
        self.assertTrue(created)
        with open(created, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("DEEPSEEK_API_KEY", content)
        self.assertIn("AZURE_TRANSLATOR_KEY", content)
        self.assertIn("TRANSLATOR=deepseek", content)

    def test_skips_when_env_enc_exists(self):
        from translate_proxy import _ensure_env_file
        d = self._dir()
        with mock.patch("translate_proxy._find_env_file",
                        side_effect=[None, "/x/.env.enc"]):
            created = _ensure_env_file(".env", dest_dir=d)
        self.assertIsNone(created)

    def test_returns_none_when_env_exists(self):
        from translate_proxy import _ensure_env_file
        d = self._dir()
        with mock.patch("translate_proxy._find_env_file", return_value="/x/.env"):
            created = _ensure_env_file(".env", dest_dir=d)
        self.assertIsNone(created)


@unittest.skipUnless(_HAS_CRYPTO, "cryptography not installed")
class EnvEncHashTests(unittest.TestCase):
    """.env.enc carries a PBKDF2 hash of the passphrase, never the passphrase."""

    def _dir(self):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d)
        return d

    def _with_passphrase(self, pw="tajne"):
        os.environ["ENV_PASSPHRASE"] = pw
        self.addCleanup(os.environ.pop, "ENV_PASSPHRASE", None)

    def test_passphrase_hashed_not_stored(self):
        d = self._dir()
        plain = os.path.join(d, ".env")
        enc = os.path.join(d, ".env.enc")
        self._with_passphrase()
        with open(plain, "w", encoding="utf-8") as f:
            f.write("K=v\n")
        encrypt_env_file(plain, enc)
        with open(enc, encoding="utf-8") as f:
            payload = next(ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith("#"))
        parts = payload.split(":")
        self.assertEqual(len(parts), 4)  # verif_salt:verif_hash:enc_salt:token
        self.assertNotIn("tajne", payload)  # passphrase never written to disk
        self.assertNotEqual(parts[1], "tajne")
        self.assertEqual(len(parts[1]), 64)  # sha256 hex digest


class MultiLanguageTests(unittest.TestCase):
    """Language-agnostic: any USER_LANG flows through exactly like Polish."""

    def setUp(self):
        clear_cache()

    def _cfg(self, user_lang, name):
        return Config(port=8800, upstream="u", verbose=False, placeholder="…",
                      user_lang=user_lang, user_lang_name=name,
                      model_lang="en", model_lang_name="English",
                      translator="openrouter", translator_model="m", translator_fallback=[],
                      translate_history=True, cache_size=5, guard_ratio=0.3,
                      translator_timeout=60, upstream_timeout=300, cerebras_base="c")

    def fake(self, calls):
        def fake(backend, model, prompt, text, cfg, src=None, dst=None):
            calls.append((src, dst))
            return "EN:" + text
        return fake

    def test_german_ingress_sends_de_to_en(self):
        calls = []
        out = translate_text("Kannst du mir helfen?", self._cfg("de", "German"),
                             "de", "en", call_backend=self.fake(calls))
        self.assertEqual(calls, [("de", "en")])
        self.assertTrue(out.startswith("EN:"))

    def test_japanese_ingress_sends_ja_to_en(self):
        calls = []
        out = translate_text("これは何ですか", self._cfg("ja", "Japanese"),
                             "ja", "en", call_backend=self.fake(calls))
        self.assertEqual(calls, [("ja", "en")])
        self.assertTrue(out.startswith("EN:"))

    def test_arabic_ingress_sends_ar_to_en(self):
        calls = []
        out = translate_text("ما هذا؟", self._cfg("ar", "Arabic"),
                             "ar", "en", call_backend=self.fake(calls))
        self.assertEqual(calls, [("ar", "en")])
        self.assertTrue(out.startswith("EN:"))

    def test_english_egress_back_to_chinese(self):
        # egress: the model's English answer is translated back to the user's zh
        calls = []
        out = translate_text("The answer is 42.", self._cfg("zh", "Chinese"),
                             "en", "zh", direction="egress", call_backend=self.fake(calls))
        self.assertEqual(calls, [("en", "zh")])
        self.assertTrue(out.startswith("EN:"))

    def test_german_not_skipped_by_english_guard(self):
        # the guard only skips text already in MODEL_LANG (en); a German prompt
        # must always reach the translator
        calls = []
        translate_text("Kannst du mir helfen?", self._cfg("de", "German"),
                       "de", "en", call_backend=self.fake(calls))
        self.assertEqual(calls, [("de", "en")])
