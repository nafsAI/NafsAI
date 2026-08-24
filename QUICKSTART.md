
# NafsAI — Quick Start

**Arabic-First Memory for Local AI**
**ذاكرة عربية محلية للذكاء الاصطناعي**

⏱ **5 minutes** — No Docker. No servers. No API key.

---

## What You Need

| Tool | Version | Link |
|------|---------|------|
| Python | 3.10+ | https://python.org |
| Ollama | Latest | https://ollama.com |
| RAM | 4GB+ | — |

---

## Step 1 — Install

```bash
pip install nafsai
```

---

## Step 2 — Choose Your Model

| Device RAM | Model | Size | Arabic Quality |
|------------|-------|------|----------------|
| Any device | `llama3.2` | 2GB | Basic |
| 8GB+ | `aya-expanse` | 4.7GB | Good |
| 16GB+ | `aya-expanse:8b` | 8GB | Best |

```bash
# Pull your chosen model
ollama pull llama3.2        # lightweight — any device
ollama pull aya-expanse     # recommended — 8GB RAM
ollama pull aya-expanse:8b  # best Arabic — 16GB RAM
```

---

## Step 3 — Connect Your LLM

**Option A — Ollama (Free & Local — Recommended)**

```python
import ollama

def my_llm(prompt: str) -> str:
    return ollama.chat(
        model="aya-expanse",   # change to your model
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]
```

**Option B — OpenAI**

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

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

## Step 4 — First Conversation

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

## Step 5 — The Real Test

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

## Full Demo

```bash
# Run with default model
python examples/ollama_demo.py

# Run with specific model
python examples/ollama_demo.py --model aya-expanse

# Show all recommended models
python examples/ollama_demo.py --list-models


```

## Next Steps

- 🎯 Run the full demo: `python examples/ollama_demo.py`
- 📖 Read the [full documentation](https://github.com/NafsAI/NafsAI/wiki)
- 💬 Ask a question: [open an issue](https://github.com/NafsAI/NafsAI/issues)

