
<div align="center">



# NafsAI
### The Arabic-First Memory Layer for Local AI
### طبقة الذاكرة العربية للذكاء الاصطناعي المحلي

[![PyPI](https://img.shields.io/pypi/v/nafsai)](https://pypi.org/project/nafsai)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Arabic-First](https://img.shields.io/badge/Arabic--First-✓-green.svg)]()

**Arabic · English · No Cloud · No API · Your Device Only**

*Built in Saudi Arabia* <img src="https://flagcdn.com/sa.svg" width="20"/> *— المملكة العربية السعودية*

[![Watch Demo](https://img.shields.io/badge/▶_Watch_Demo-black?style=for-the-badge)](https://github.com/user-attachments/assets/fb93518b-092c-468f-befc-a2e36ed90ae3)

> 
>

</div>

---

## The Problem

Every AI chatbot you build forgets everything after each conversation.

كل chatbot تبنيه ينسى كل شيء بعد كل محادثة.

## The Solution

```python
from nafsai import Agent

agent = Agent()
answer = agent.chat("What do you remember about me?", your_llm)
# Remembers. Always. In Arabic and English.
# يتذكر. دائماً. بالعربية والإنجليزية.
```

**Permanent memory. No cloud. Arabic-first. Works in English too.**

**ذاكرة دائمة. بدون سحابة. عربي أولاً. يعمل بالإنجليزية أيضاً.**

---

## Why NafsAI?

| | mem0 | Zep | Letta | **NafsAI** |
|--|------|-----|-------|------------|
| Arabic-First | ❌ | ❌ | ❌ | **✅** |
| No Cloud Required | ⚠️ | ⚠️ | ⚠️ | **✅** |
| pip install only | ❌ | ❌ | ❌ | **✅** |
| No Docker | ❌ | ❌ | ❌ | **✅** |
| Local & Private | ⚠️ | ⚠️ | ⚠️ | **✅** |
| Works Offline | ❌ | ❌ | ❌ | **✅** |

---

## Installation

```bash
pip install nafsai
```

That's it. No Docker. No server. No API key.

هذا كل شيء. بدون Docker. بدون خادم. بدون API key.

---

## Quick Start

```python
from nafsai import Agent

agent = Agent()

def my_llm(prompt: str) -> str:
    # your LLM here — Ollama, OpenAI, Gemini, anything
    return your_llm_function(prompt)

# First session
agent.chat("اسمي عبدالله وأنا مطور Python", my_llm)
agent.chat("My name is John and I work in Riyadh", my_llm)

# Close and reopen — memory persists
# أغلق وأعد التشغيل — الذاكرة تبقى

# New session
agent2 = Agent()
agent2.chat("هل تتذكرني؟", my_llm)
# → "نعم عبدالله، أنت مطور Python"

agent2.chat("Who am I?", my_llm)
# → "You are John, you work in Riyadh"
```

---

## Where is My Data Stored?

```
your-project/
└── nafsai_memory.db   ← all memory stored here (SQLite)
```

- **Default location:** same folder where you run your script
- **Custom location:**

```python
agent = Agent(db_path="/your/custom/path/memory.db")
```

- **To move your memory** to another device: copy `nafsai_memory.db`
- **To reset memory:** delete `nafsai_memory.db`

---

## How It Works

```
Your App
    ↓
Agent   ← combines all components
    ↓
Router  ← classifies question  (coding / reasoning / general)
Memory  ← permanent storage    (SQLite + FTS5 + vectors)
Cache   ← smart TTL cache
Session ← current conversation context
    ↓
Your LLM ← Ollama / OpenAI / Gemini / anything
```

### Three Memory Modes

NafsAI detects your environment and uses the best available mode:

| Mode | Engine | Performance |
|------|--------|-------------|
| `full` | FTS5 + sqlite-vec | Best |
| `fts_only` | FTS5 only | Good |
| `numpy_only` | numpy cosine | Always works |

No configuration needed. It just works.

---

## Components

| Component | Description | الوصف |
|-----------|-------------|-------|
| `Memory` | Permanent local memory | ذاكرة دائمة محلية |
| `Router` | Smart question routing | توجيه ذكي للأسئلة |
| `Normalizer` | Arabic text normalization | تطبيع النص العربي |
| `Cache` | TTL-based smart cache | كاش ذكي |
| `Session` | Conversation context | سياق المحادثة |
| `Agent` | All-in-one interface | واجهة موحدة |

---

## Works With Any LLM

```python
# Ollama (local & free)
import ollama

def llm(prompt: str) -> str:
    return ollama.chat(
        model="command-r",
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]

# OpenAI
from openai import OpenAI

client = OpenAI()

def llm(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

# Any function that takes str and returns str
agent.chat("your question", llm)
```

---

## Tests & Quality

All 96 tests pass across all memory modes:

| Test File | Coverage |
|-----------|----------|
| `test_agent.py` | Agent pipeline, cache hits, LLM errors, prompt building |
| `test_memory.py` | Save/search, user facts, normalization, persistence, all 3 modes |
| `test_cache.py` | TTL rules, route isolation, bad responses, Arabic normalization |
| `test_router.py` | Keyword rules, semantic routing, Arabic & English questions |
| `test_normalizer.py` | Hamza, Tashkeel, numbers, math symbols, mixed text |
| `test_session.py` | Turn limits, context window, f-string format, truncation |

```bash
pip install -e ".[dev]"
pytest
# ✓ 96 passed
```

---

## Privacy

```
✓ Everything stored locally — your device only
✓ No data leaves your machine
✓ No telemetry
✓ No internet required after installation
✓ GDPR-friendly by design
```

```
✓ كل شيء يُخزن محلياً على جهازك فقط
✓ لا بيانات تغادر جهازك
✓ لا تتبع
✓ لا إنترنت مطلوب بعد التثبيت
```

---

## Roadmap

- [x] SQLite + FTS5 + vector search
- [x] Arabic-English bilingual support
- [x] Smart fallback system
- [x] Arabic text normalization
- [ ] MCP server support

---

## Contributing

```bash
git clone https://github.com/NafsAI/NafsAI
cd NafsAI
pip install -e ".[dev]"
pytest
```

All contributions welcome — Arabic and English.

كل المساهمات مرحب بها — بالعربية والإنجليزية.

---

<div align="center">

**Built in Saudi Arabia 🇸🇦 — For the Arabic AI Ecosystem**

[PyPI](https://pypi.org/project/nafsai) ·
[Documentation](https://github.com/NafsAI/NafsAI/wiki) ·
[Issues](https://github.com/NafsAI/NafsAI/issues) ·
[Discussions](https://github.com/NafsAI/NafsAI/discussions) ·
[Discord](https://discord.gg/fUqpDfVeY)

*"The Arabic world deserves its own AI tools — built here, for here, by us."*

*"العالم العربي يستحق أدوات AI — مبنية هنا، لنا، منّا."*

</div>
