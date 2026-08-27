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

*You* write in *your* language. On mid-range and premium models the model thinks in
English and the answer comes back in *your* language. On cheap models
(`deepseek-v4-flash`, `deepseek-pro`) the proxy notices and passes you straight
through — no translation, nothing to pay.

- 🗣️ Ask in Polish, German, Japanese, Arabic, Hindi, Swahili… ~100 languages, zero setup.
- 🧠 On mid-range and premium models the model only ever sees and produces **English** —
  its strongest language. Better reasoning, and you never waste tokens on "sorry,
  could you say that in English?".
- 💸 **Save tokens** — say it once, in your language, done.
- 📉 **Save ~35–65%** on mid-range and premium models (measured). 📈 **Save 140%.**
  (Math says you can't save more than 100%, but our heart says otherwise.)
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

> On a cheap main model (`deepseek-v4-flash`, `deepseek-pro`) there is no English
> detour at all — the proxy relays your words and the answer back untouched. See
> [Real cost](#-what-you-actually-pay) below for why.

### 💰 What you actually pay

The main model costs what your provider charges. **The only *extra* cost is the cheap
translation model** — one API call per translation. One exchange = 2 translations
(in + out).

Model price (per million tokens):

| Model | Input | Output |
|---|---|---|
| `deepseek-v4-flash` (default) | $0.08 | $0.16 |

Cost per **single translation** (one direction):

| Text size | Example | Cost |
|---|---|---|
| Tiny (~20 tokens) | "How are you?" | ~$0.000005 |
| Typical (~200 tokens) | a question about your code | ~$0.00005 |
| Long (~800 tokens) | explain this whole file | ~$0.0002 |

### Real cost — with and without the proxy

Measured on this repo: a typical exchange is a ~120-token prompt and a ~250-token
answer. "Without" = you write in your language, the model answers in your language.
"With" = your words reach the model in English and the answer comes back translated
(deepseek-v4-flash translator, ≈$0.0001 per exchange). Numbers cover the whole
exchange — main model + translation — at current DeepSeek prices:

| Main model ($/M in / out) | 1 exchange, without | 1 exchange, with proxy | 30 exchanges, without | 30 exchanges, with proxy |
|---|---|---|---|---|
| `deepseek-v4-flash` ($0.08 / $0.16) — passed through | $0.00008 | $0.00008 | $0.0025 | $0.0025 |
| mid-tier, e.g. Sonnet ($3 / $15) | $0.0075 | $0.0042 | $0.226 | $0.127 |
| premium, e.g. Opus ($15 / $75) | $0.038 | $0.021 | $1.13 | $0.62 |

#### Cost by your language — before vs after the proxy

How much your language costs next to English on the main model's tokenizer, and what
the proxy changes. Measured on an identical code-related exchange (a Python question +
explanation; prose-only chat runs a bit higher). Baseline: one ~120-token prompt + one
~250-token answer, premium main ($15 / $75). Ratios measured with the **cl100k
tokenizer** (Anthropic family) for the main-model leg and **deepseek-v4-flash's own
tokenizer** for the translator leg:

| Your language | tokens vs English | 1 exchange, without | 1 exchange, with proxy | 30 exchanges, without | 30 exchanges, with proxy | you save |
|---|---|---|---|---|---|---|
| English | 1.0× | $0.021 | $0.021 | $0.62 | $0.62 | — |
| Polish | 1.8× | $0.038 | $0.021 | $1.13 | $0.62 | ~45% |
| German | 1.6× | $0.034 | $0.021 | $1.01 | $0.62 | ~39% |
| Japanese | 2.3× | $0.047 | $0.021 | $1.40 | $0.62 | ~56% |
| Arabic | 2.8× | $0.058 | $0.021 | $1.73 | $0.62 | ~64% |
| Chinese | 1.6× | $0.032 | $0.021 | $0.96 | $0.62 | ~36% |

The "with proxy" column is ~the same for every language — that's the point. Translation
itself costs ≈$0.0001 no matter your language; the entire saving is the main model
billing **English instead of your (longer) language**. Same shape on mid-tier mains
(Sonnet $3 / $15): ~35–63% per language. Fun fact: deepseek's tokenizer is optimized
for Chinese — it bills Chinese *under* English (0.92×) — so Chinese pays the least for
translation and still saves on the main model.

#### Where the money goes — the two translation legs vs the saving

Same scenario, DeepSeek translator only (Google Translate / DeepL / Azure bill the same
way, at their own rates). "→ EN" = your question being translated in; "EN → you" = the
answer being translated back. The main model always bills **English** — a fixed $0.0206
for every language; the only per-language line is the translation:

| Your language | translate question → EN | translate answer EN → you | translation total | **with proxy** (main + translation) | without proxy | you save |
|---|---|---|---|---|---|---|
| Polish | $0.00004 | $0.00009 | $0.00012 | $0.0207 | $0.038 | **~45%** |
| German | $0.00003 | $0.00008 | $0.00012 | $0.0207 | $0.034 | ~39% |
| Japanese | $0.00003 | $0.00008 | $0.00011 | $0.0207 | $0.047 | ~56% |
| Arabic | $0.00003 | $0.00008 | $0.00012 | $0.0207 | $0.058 | **~64%** |
| Chinese | $0.00003 | $0.00006 | $0.00008 | $0.0206 | $0.032 | ~36% |

Read it top to bottom: the **translation is 0.4–0.6% of the exchange** — a ~$0.0001
rounding error — while the main-model saving is **36–64%**. The answer leg costs ~2×
the question leg (the answer is longer, and DeepSeek bills output at double the input
rate). So on mid and premium mains it *always* pays: you pay a half-percent tax on a
thirty-to-sixty-percent discount.

The one case where it does **not** pay is `deepseek-v4-flash` / `deepseek-pro` as the
*main* model — the proxy skips translation entirely (see above), because translating
an exchange that cheap would cost ~1.5× the exchange itself.

What this means:

- **Cheap main model** (like `deepseek-v4-flash` itself): passed through untranslated,
  so the proxy adds **$0**. A whole 30-exchange session is **$0.0025** — you pay exactly
  what you'd pay without the proxy.
- **Mid and premium mains**: the proxy is **~45% cheaper** for Polish — English takes
  ~1.8× fewer tokens than Polish on the main model, and that saving beats the
  translation cost. A 30-exchange premium session: **$0.62 with proxy vs $1.13 without**.

#### Why the proxy skips DeepSeek — translation costs ≈2× the exchange itself

`deepseek-v4-flash`, `deepseek-pro` (and anything else in `SKIP_TRANSLATION_MODELS`)
pass through **untranslated**. It's not about quality — it's arithmetic.

That ~$0.00008 exchange is ~630 Polish tokens on the deepseek tokenizer. Translating it
costs **two extra round-trips through the same model**, and the translated text is about
the size of the original (in: ~205 tokens read + ~120 written; out: ~250 read + ~425
written). That's roughly **$0.0001 of translation** — *more than the $0.00008 exchange
itself*. Translate both sides and you've **~doubled your bill** for no reasoning gain:
a model that cheap has no cheaper language to think in.

On mid-range and premium models the same two translations are a rounding error next to
the main-model saving — English is ~1.8× cheaper than Polish (up to ~2.8× for Arabic),
so the proxy is still ~36–64% cheaper overall. DeepSeek is the one case where the
arithmetic flips.

So the proxy **detects the main model automatically** and skips translation when it's
not worth it: cheap models (`deepseek-v4-flash`, `deepseek-pro`, default) pass through
as a plain relay — your words go to the model untouched and its answer comes back
untouched. Every other model is translated as described. Tune with
`SKIP_TRANSLATION_MODELS` (substring match, so `deepseek/deepseek-v4-flash` is caught
too); set it to `""` to translate for every model.

### What you need to run it

| You need | How |
|---|---|
| **Python 3** | already on most systems — check with `python3 --version` |
| **A model endpoint to relay to** | any Anthropic/OpenAI-compatible server (e.g. your Claude Code → DeepSeek bridge) — pass it with `--upstream` |
| **A DeepSeek API key** | the default translator (`deepseek-v4-flash`) talks directly to `api.deepseek.com` — set `DEEPSEEK_API_KEY` |
| *(optional)* **An OpenRouter API key** | only if you switch the translator to OpenRouter — set `OPENROUTER_API_KEY` |
| **Nothing else** | no pip installs, no database, no Docker — one Python file |

**Keys:** the proxy reads your shell environment — or, easiest, a local `.env`
file. `cp .env.example .env`, paste your keys in it, run. It's gitignored and
auto-loaded, and keys you've already exported in your shell always win. No `.env`
yet? On the first run the proxy **creates one for you** from the template — just
paste your keys in and restart.

## Quickstart

```bash
cp .env.example .env   # paste your API keys here (the proxy auto-loads it)
python3 translate_proxy.py --upstream http://127.0.0.1:8799
```

> Ran it without a `.env`? The proxy made one from the template on its own — fill in
> your keys and run again.

Point your agent at it — only the URL changes; your agent keeps its normal protocol
and model. Each tool also needs the API key your upstream accepts, and (for Claude
Code) a model name your upstream understands:

| Tool | Config file | Change |
|---|---|---|
| **Claude Code** | `~/.claude/settings.json` → `"env"` | `"ANTHROPIC_BASE_URL": "http://127.0.0.1:8800"` |
| **opencode** | `opencode.json` → `provider` | `"baseURL": "http://127.0.0.1:8800"` |
| **Hermes** | `config.yaml` → `model` | `base_url: http://127.0.0.1:8800` |

Two settings complete the chain:

- **API key** — the request must carry a token your upstream accepts. Claude Code:
  `ANTHROPIC_AUTH_TOKEN` in `settings.json`'s `env`; opencode: the provider's
  `apiKey`; Hermes: its provider key. With a real Anthropic/OpenAI API that's just
  your normal key.
- **Model name** — use one your upstream understands. Claude Code asks for
  `claude-sonnet-*` by default; point it at your model with
  `ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-flash` in `settings.json`'s `env`
  (add `ANTHROPIC_DEFAULT_OPUS_MODEL` / `ANTHROPIC_DEFAULT_HAIKU_MODEL` for the
  other tiers). opencode: set `"model"`; Hermes: set `model.default`.

`GET /health` → `{"ok": true}`. Stop: `python3 translate_proxy.py --port 8800 --stop`.

## Configuration

All config is via environment variables, auto-loaded from a local `.env` file
(`cp .env.example .env` and fill it in — the proxy loads it automatically; a
missing file is fine and keys you export yourself always win).

| Var | Default | Purpose |
|---|---|---|
| `USER_LANG` | `pl` | language you write in |
| `MODEL_LANG` | `en` | language the main model should use |
| `TRANSLATOR` | `deepseek` | **your choice** of translator backend: `deepseek` \| `openrouter` \| `cerebras` \| `google` \| `libretranslate` \| `deepl` \| `azure` |
| `TRANSLATOR_MODEL` | `deepseek-v4-flash` | model for the primary backend (ignored by machine-translation backends) |
| `TRANSLATOR_FALLBACK` | *LLM only* | comma-separated `backend/model` backup chain. **Machine-translation backends have no implicit fallback** — the `TRANSLATOR` choice is authoritative; a fallback only runs if you set it here |
| `TRANSLATE_HISTORY` | `true` | translate history prose to `MODEL_LANG` on ingress |
| `CACHE_SIZE` | `500` | in-memory translation cache bound |
| `GUARD_STOPWORD_RATIO` | `0.3` | above this English-stopword ratio input is passed through untranslated |
| `PLACEHOLDER` | `…` | shown while the egress answer is being translated |
| `SKIP_TRANSLATION_MODELS` | `deepseek-v4-flash,deepseek-pro` | main models that pass through untranslated (substring match; `""` = translate for every model) |
| `DEEPSEEK_API_KEY` | — | needed to translate via the default `deepseek` backend (without it translation is skipped) |
| `OPENROUTER_API_KEY` | — | needed to translate via `openrouter` (an LLM backend) |
| `GOOGLE_TRANSLATE_API_KEY` | — | needed to translate via the `google` backend (Cloud Translation v2) |
| `LIBRETRANSLATE_BASE` | `http://127.0.0.1:5000` | base URL of your LibreTranslate instance (the `libretranslate` backend appends `/translate`) |
| `LIBRETRANSLATE_API_KEY` | — | optional API key sent to LibreTranslate when the instance requires one |
| `DEEPL_API_KEY` | — | needed to translate via the `deepl` backend |
| `DEEPL_API_BASE` | `https://api-free.deepl.com/v2` | DeepL endpoint; use `https://api.deepl.com/v2` for a Pro account |
| `AZURE_TRANSLATOR_KEY` | — | needed to translate via the `azure` backend (Microsoft Translator) |
| `AZURE_TRANSLATOR_REGION` | — | Azure resource region (e.g. `westeurope`) — required by regional Azure resources |

Backends:

- `openrouter` — `https://openrouter.ai/api/v1/chat/completions`, needs `OPENROUTER_API_KEY`;
- `deepseek` — `https://api.deepseek.com/chat/completions`, needs `DEEPSEEK_API_KEY`, model `deepseek-v4-flash` (default);
- `cerebras` — local gateway, `CEREBRAS_BASE` (default `http://127.0.0.1:8001/v1`), no key;
- `google` — [Google Cloud Translation v2](https://cloud.google.com/translate) (`translation.googleapis.com`), needs `GOOGLE_TRANSLATE_API_KEY`; billed per character, **500K characters/month free forever**, then ~$20 per million chars;
- `libretranslate` — self-hosted, **completely free** ([Argos Translate](https://github.com/argosopentech/argos-translate)), default `http://127.0.0.1:5000`, optional `LIBRETRANSLATE_API_KEY`;
- `deepl` — [DeepL API v2](https://developers.deepl.com) (`api-free.deepl.com/v2`, Pro via `DEEPL_API_BASE`), needs `DEEPL_API_KEY`; best quality for European pairs, free endpoint gives a one-time 1M-char evaluation;
- `azure` — [Microsoft Translator (Azure AI Translator)](https://learn.microsoft.com/en-us/azure/ai-services/translator/), needs `AZURE_TRANSLATOR_KEY` (+ `AZURE_TRANSLATOR_REGION` for regional resources); **permanent free tier 2M chars/month**, then $10 per million chars.

### Which translator do you pick? You pick — `TRANSLATOR`

`TRANSLATOR` is your choice, and it's authoritative. LLM backends
(`deepseek`, `openrouter`, `cerebras`) keep an implicit OpenRouter safety-net
fallback; **machine-translation backends (`google`, `libretranslate`, `deepl`,
`azure`) have no implicit fallback** — if your chosen API fails, the original
text passes through untranslated rather than silently switching to a different
(possibly paid) provider. Add a backup yourself only if you want one:

```bash
# Google Cloud Translation (no model to pay — free tier 500K chars/month)
TRANSLATOR=google GOOGLE_TRANSLATE_API_KEY=AIza... \
python3 translate_proxy.py --upstream http://127.0.0.1:8799

# LibreTranslate (free, self-hosted)
TRANSLATOR=libretranslate \
python3 translate_proxy.py --upstream http://127.0.0.1:8799

# DeepL (best quality for European pairs; Pro endpoint via DEEPL_API_BASE)
TRANSLATOR=deepl DEEPL_API_KEY=... \
python3 translate_proxy.py --upstream http://127.0.0.1:8799

# Microsoft Translator (permanent 2M chars/month free)
TRANSLATOR=azure AZURE_TRANSLATOR_KEY=... AZURE_TRANSLATOR_REGION=westeurope \
python3 translate_proxy.py --upstream http://127.0.0.1:8799

# ...and if you want an explicit backup only when Google fails:
TRANSLATOR=google TRANSLATOR_FALLBACK=libretranslate ...
```

Machine-translation backends have no model — the text and the language pair go
straight to the API (`Google` v2, your LibreTranslate `/translate`, DeepL
`/translate`, or Azure `translate`). Free-tier scale: a typical exchange is
~1,500 characters (a ~500-char prompt + ~1,000-char answer), a long answer
~3,500 — so Google's 500K/month ≈ ~330 typical exchanges, Azure's permanent
2M/month ≈ ~1,300, and DeepL's one-time 1M ≈ ~660. Code blocks, inline code,
URLs and already-English text are **not** counted (they're passed through
verbatim).

### Use OpenRouter instead

The default talks straight to DeepSeek. To translate through OpenRouter instead (one
key, many models), set two env vars and provide `OPENROUTER_API_KEY`:

```bash
TRANSLATOR=openrouter \
TRANSLATOR_MODEL=deepseek/deepseek-v4-flash \
OPENROUTER_API_KEY=sk-or-... \
python3 translate_proxy.py --upstream http://127.0.0.1:8799
```

That's the whole change — everything else works the same.

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
- **Fallback:** on translator failure the next backend in `TRANSLATOR_FALLBACK` is tried
  (LLM backends get an implicit OpenRouter backup; machine-translation backends get none —
  your `TRANSLATOR` choice is authoritative). Only when the whole chain fails is the
  original English (egress) or your prompt (ingress) passed through, so you never lose an answer.
- The placeholder is cosmetic; disable it with `PLACEHOLDER=`.

## Security

- Binds `127.0.0.1` only — never expose it publicly.
- API keys live in environment variables, or in a gitignored `.env` (auto-loaded), never in the
  repo — optionally encrypted at rest as `.env.enc` (see below).
- Keep it behind your existing local proxy chain (e.g. Claude Code's deepseek proxy).

### 🔐 Optional: encrypted `.env`

Keep your keys encrypted at rest, so even a leaked file is useless:

```bash
python3 translate_proxy.py --make-env-enc   # prompts for a passphrase → writes .env.enc
rm .env                                      # leave only the ciphertext
ENV_PASSPHRASE=... python3 translate_proxy.py --upstream http://127.0.0.1:8799
```

No `.env` yet? `--make-env-enc` creates one from the template first — fill in your
keys, then run it again to encrypt. The proxy decrypts `.env.enc` **in memory** and
never writes plaintext back to disk. To change keys later:

```bash
python3 translate_proxy.py --env-decrypt     # back to plaintext .env for editing
# ...edit .env... then re-encrypt with --make-env-enc, and rm .env again
```

The payload is a Fernet (AES) token with a passphrase-derived key
(PBKDF2-HMAC-SHA256, 600k iterations). Your passphrase is **never stored** — the
file holds only a PBKDF2 hash of it, checked (in constant time) to verify you
typed the right passphrase before the token is decrypted. It's off by default and
optional: plain `.env` needs nothing; this mode needs `pip install cryptography`.
It protects the file *at rest* — anyone who can read the running process still has
the keys, so keep the machine itself safe.

## Tests

```bash
python3 -m unittest -v test_translate_proxy   # offline, translator mocked
```

113 tests, zero network calls — the translator is mocked, so running the suite
costs nothing. Language pairs are covered explicitly, so adding a new `USER_LANG`
can't silently break:

| `USER_LANG` | Flow tested | Assertion | Test |
|---|---|---|---|
| `pl` (Polish) | `pl`→`en` in, `en`→`pl` out | full round trip | `TranslatorTests`, `IngressTests` |
| `de` (German) | `de`→`en` in | model gets English; German is **not** guard-skipped | `test_german_ingress_sends_de_to_en` |
| `ja` (Japanese) | `ja`→`en` in | model gets English | `test_japanese_ingress_sends_ja_to_en` |
| `ar` (Arabic) | `ar`→`en` in | model gets English | `test_arabic_ingress_sends_ar_to_en` |
| `zh` (Chinese) | `en`→`zh` out | answer comes back in Chinese | `test_english_egress_back_to_chinese` |

## Built with

- 🔮🪄 **Imagination** — the main ingredient
- **DeepSeek V4 Flash** — the model that does the thinking (also the default translator)
- **DeepClaude** — local bridge that routes Claude Code to DeepSeek
- **Claude Code** — the agent that wrote every line
- **Python standard library** — zero dependencies, nothing else

## License

MIT.
