"""
NafsAI — Live Demo with Ollama
Tests memory persistence across sessions.

Installation:
    pip install ollama
    ollama pull <your-model>

Recommended models:
    ollama pull aya-expanse     # 4.7GB — Best Arabic (8GB RAM)
    ollama pull aya-expanse:8b  # 8GB   — Best Arabic (16GB RAM)
    ollama pull llama3.2        # 2GB   — Lightweight (any device)

Usage:
    python examples/ollama_demo.py
    python examples/ollama_demo.py --model aya-expanse
"""

import argparse
import sys
import ollama
from nafsai import Agent


# ─── config ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama3.2"

RECOMMENDED_MODELS = {
    "aya-expanse":    "Best Arabic quality — requires 8GB RAM",
    "aya-expanse:8b": "Best Arabic quality — requires 16GB RAM",
    "llama3.2":       "Lightweight — works on any device",
    "gemma3:4b":      "Good balance — requires 8GB RAM",
    "command-r":      "Excellent Arabic — requires 32GB RAM",
}


# ─── helpers ──────────────────────────────────────────────────────────────────


def separator(title: str = ""):
    if title:
        pad = (48 - len(title)) // 2
        print(f"\n{'═' * pad} {title} {'═' * pad}")
    else:
        print("═" * 50)


def check_model(model: str) -> bool:
    """Check if the model is available in Ollama."""
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        for m in available:
            if model in m:
                return True
        return False
    except Exception:
        return False


def build_llm(model: str):
    """Returns an LLM function for the given model."""
    def llm(prompt: str) -> str:
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Ollama error: {e}"
    return llm


# ─── sessions ─────────────────────────────────────────────────────────────────


def session_one(agent: Agent, llm):
    """Session 1 — introduce yourself and save to memory."""
    separator("Session 1 — Remember")

    conversations = [
        "اسمي عبدالله وأنا مطور Python",
        "أسكن في الرياض وأعمل في مشاريع الذكاء الاصطناعي",
        "مشروعي الحالي هو NafsAI لبناء ذاكرة عربية للـ LLM",
    ]

    for msg in conversations:
        print(f"\n أنت : {msg}")
        reply = agent.chat(msg, llm)
        print(f" AI  : {reply[:120]}...")

    separator("Memory Test — Same Session")

    questions = [
        "ما اسمي؟",
        "أين أسكن؟",
        "ماذا أعمل؟",
        "ما هو مشروعي الحالي؟",
    ]

    print()
    for q in questions:
        print(f" سؤال : {q}")
        answer = agent.chat(q, llm)
        print(f" جواب : {answer[:150]}")
        print()

    separator("Cache Test")
    print("\n نفس السؤال مرة ثانية: ما اسمي؟")
    cached_answer = agent.chat("ما اسمي؟", llm)
    print(f" الجواب (من الكاش) : {cached_answer[:150]}")

    stats = agent.cache.stats()
    print(f"\n الأسئلة في الكاش : {stats['cached']}")
    print(f" Hit Rate         : {stats['hit_rate']}")
    print(f" عدد رسائل الجلسة : {agent.session.count()}")


def session_two(agent: Agent, llm):
    """Session 2 — new instance, tests persistent memory."""
    separator("Session 2 — Recall After Restart")

    questions = [
        "هل تتذكرني؟ ما اسمي؟",
        "أين أسكن؟",
        "ما هو مشروعي؟",
        "ماذا تعرف عني؟",
    ]

    print()
    for q in questions:
        print(f" سؤال : {q}")
        answer = agent.chat(q, llm)
        print(f" جواب : {answer[:200]}")
        print()

    separator("Permanent Memory")
    memories = agent.recall("معلومات عن المستخدم", top_k=5)
    if memories:
        for i, mem in enumerate(memories, 1):
            print(f"\n [{i}] {mem[:120]}")
    else:
        print(" لا توجد ذكريات — شغّل Session 1 أولاً")

    separator("User Profile")
    profile = agent.memory.get_user_profile("معلومات المستخدم", top_k=5)
    if profile:
        for fact in profile:
            print(f" ✓ {fact}")
    else:
        print(" لا يوجد ملف شخصي بعد")


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NafsAI Demo with Ollama")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Show recommended models and exit",
    )
    args = parser.parse_args()

    # ── list models ──
    if args.list_models:
        print("\n Recommended models for NafsAI:\n")
        for model, desc in RECOMMENDED_MODELS.items():
            print(f"   ollama pull {model:<20} # {desc}")
        print()
        sys.exit(0)

    model = args.model

    separator()
    print(" NafsAI — Arabic Memory Layer")
    print(" The Arabic-First Memory Layer for Local AI")
    separator()

    # ── check model ──
    print(f"\n Checking model: {model}...")
    if not check_model(model):
        print(f"\n ✗ Model '{model}' not found in Ollama.")
        print(f"\n Run: ollama pull {model}")
        print("\n Or choose a recommended model:")
        for m, desc in RECOMMENDED_MODELS.items():
            print(f"   ollama pull {m:<20} # {desc}")
        print(f"\n Then run: python examples/ollama_demo.py --model {model}")
        sys.exit(1)

    print(f" ✓ Model ready: {model}\n")

    llm = build_llm(model)

    print(f"""
 This demo runs in two sessions:

 Session 1 — Introduce yourself → memory is saved
 Session 2 — New Agent instance → memory is recalled

 Model : {model}
 This proves NafsAI remembers across restarts.
    """)

    # ── Session 1 ──
    agent1 = Agent()
    session_one(agent1, llm)

    separator()
    print("\n Simulating restart — creating new Agent instance...\n")

    # ── Session 2 ──
    agent2 = Agent()
    session_two(agent2, llm)

    separator()
    print(f"""
 ✓ All conversations saved locally on your device
 ✓ No cloud — No API — No Docker
 ✓ Model used: {model}

 github.com/NafsAI/NafsAI
 pip install nafsai
    """)
    separator()


if __name__ == "__main__":
    main()
