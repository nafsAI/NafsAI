"""
NafsAI — Live Demo with Ollama
Tests memory persistence across sessions.

Installation:
    pip install ollama
    ollama pull llama3.2

Usage:
    python examples/ollama_demo.py
"""

import ollama
from nafsai import Agent


def separator(title: str = ""):
    if title:
        pad = (48 - len(title)) // 2
        print(f"\n{'═' * pad} {title} {'═' * pad}")
    else:
        print("═" * 50)


def llm(prompt: str) -> str:
    """Call llama3.2 via Ollama."""
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Ollama error: {e}"


def session_one(agent: Agent):
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


def session_two(agent: Agent):
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


def main():
    separator()
    print(" NafsAI — Arabic Memory Layer")
    print(" The Arabic-First Memory Layer for Local AI")
    separator()

    print("""
 This demo runs in two sessions:

 Session 1 — Introduce yourself → memory is saved
 Session 2 — New Agent instance → memory is recalled

 This proves NafsAI remembers across restarts.
    """)

    # ── Session 1 ──
    agent1 = Agent()
    session_one(agent1)

    separator()
    print("\n Simulating restart — creating new Agent instance...\n")

    # ── Session 2 ──
    agent2 = Agent()
    session_two(agent2)

    separator()
    print("""
 ✓ All conversations saved locally on your device
 ✓ No cloud — No API — No Docker

 github.com/NafsAI/NafsAI
 pip install nafsai
    """)
    separator()


if __name__ == "__main__":
    main()