
# NafsAI — Quick Start

**Arabic-First Memory for Local AI**
**ذاكرة عربية محلية للذكاء الاصطناعي**

⏱ **5 minutes** — No Docker. No servers. No API key.

---

## What You Need

| Tool | Version | Link |
|------|---------|------|
| Python | 3.10+ | https://python.org |
| RAM | 4GB+ | — |
| Any LLM | Ollama / OpenAI / Gemini | — |

---

## Step 1 — Install

```bash
pip install nafsai
```

---

## Step 2 — Choose Your LLM

**Option A — Ollama (Free & Local — Recommended)**

```bash
# Install Ollama from https://ollama.com
ollama pull command-r
```

```python
import ollama

def my_llm(prompt: str) -> str:
    return ollama.chat(
        model="command-r",
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]
```

**Option B — OpenAI**

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

def my_llm(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content
```

**Option C — Any LLM**

```python
# Any function that takes str and returns str
def my_llm(prompt: str) -> str:
    return your_model(prompt)
```

---

## Step 3 — First Conversation

```python
from nafsai import Agent

agent = Agent()

# Tell it about yourself
agent.chat("اسمي عبدالله وأنا مطور Python", my_llm)
agent.chat("I work on AI projects in Riyadh", my_llm)

# Ask it what it knows
print(agent.chat("ما الذي تعرفه عني؟", my_llm))
print(agent.chat("What do you know about me?", my_llm))
```

---

## Step 4 — The Real Test

Close your terminal. Open a new one.

```python
from nafsai import Agent

agent = Agent()  # Fresh start — new session

# Does it remember?
print(agent.chat("هل تتذكرني؟", my_llm))
# → Yes. Always.
```

**This is NafsAI. Permanent memory. Your device only.**
**هذا هو NafsAI. ذاكرة دائمة. جهازك فقط.**

---

## Where is My Data Stored?

```
your-project/
└── nafsai_memory.db   ← all memory here
```

- **Move memory** to another device: copy `nafsai_memory.db`
- **Reset memory:** delete `nafsai_memory.db`
- **Custom path:**

```python
agent = Agent(db_path="/your/path/memory.db")
```

---

## Memory Modes

NafsAI auto-detects your environment and picks the best available mode:

| Mode | Engine | Performance |
|------|--------|-------------|
| `full` | FTS5 + vectors | Best |
| `fts_only` | FTS5 only | Good |
| `numpy_only` | numpy cosine | Always works |

No configuration needed. It chooses automatically.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: nafsai` | Run `pip install nafsai` |
| Ollama not responding | Run `ollama serve` first |
| Memory not persisting | Check `nafsai_memory.db` exists in your folder |
| Slow first run | Normal — model loads once, then cached |

---

## Next Steps

- 🎯 Run the full demo: `python examples/ollama_demo.py`
- 📖 Read the [full documentation](https://github.com/NafsAI/NafsAI/wiki)
- 💬 Ask a question: [open an issue](https://github.com/NafsAI/NafsAI/issues)
```

---

### ملخص التحسينات

| العنصر | قبل | بعد |
|--------|-----|-----|
| وقت الإنجاز | ❌ غائب | ✅ 5 دقائق |
| الموديل الموصى به | llama3.2 ضعيف | ✅ command-r |
| أين البيانات | ❌ غائب | ✅ موجود |
| Troubleshooting | ❌ غائب | ✅ جدول كامل |
| Next Steps | سطر واحد | ✅ 3 خطوات مفيدة |
| متطلبات RAM | ❌ غائب | ✅ موجود |
