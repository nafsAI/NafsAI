"""
Agent — NafsAI
Combines all components into a single interface.
"""
import logging
import time
from typing import Callable, Optional

from nafsai.cache import Cache
from nafsai.memory import Memory
from nafsai.normalizer import Normalizer
from nafsai.router import Router
from nafsai.session import Session

log = logging.getLogger("nafsai.agent")


class Agent:
    """
    NafsAI Agent — Arabic-First Memory Layer.

    Usage:
        agent = Agent()

        def my_llm(prompt: str) -> str:
            return your_llm_function(prompt)

        answer = agent.chat("What is my name?", my_llm)
    """

    SYSTEM_PROMPT_AR = (
        "أنت مساعد ذكي يتذكر المحادثات السابقة. "
        "استخدم الذاكرة والسياق المقدم للإجابة بدقة."
    )

    SYSTEM_PROMPT_EN = (
        "You are a smart assistant that remembers previous conversations. "
        "Use the provided memory and context to answer accurately."
    )

    def __init__(
        self,
        db_path: str = "./nafsai_memory.db",
        model_name: str = "intfloat/multilingual-e5-base",
        language: str = "ar",
        verbose: bool = True,
    ):
        if verbose:
            print("\n NafsAI — Initializing...")
            print("─" * 40)

        self.language = language
        self.system_prompt = (
            self.SYSTEM_PROMPT_AR
            if language == "ar"
            else self.SYSTEM_PROMPT_EN
        )

        self.normalizer = Normalizer()
        self.router     = Router(model_name=model_name)
        self.memory     = Memory(
            model_name=model_name,
            db_path=db_path,
            _shared_model=self.router.model,
        )
        self.cache   = Cache()
        self.session = Session()

        if verbose:
            print("─" * 40)
            memory_status = "Connected" if self.memory.available else "Unavailable"
            memory_mode = {
                "full":       "Full  (FTS5 + vectors)",
                "fts_only":   "Good  (FTS5 only)",
                "numpy_only": "Basic (numpy)",
            }.get(self.memory.mode, "Unknown")
            print(f" Memory  : {memory_status} — {memory_mode}")
            print(" Router  : Ready")
            print(" Cache   : Ready")
            print(f" Version : v{self._version()}")
            print("─" * 40)
            print(" NafsAI Ready\n")

        log.info("NafsAI ready ✓")

    def _version(self) -> str:
        try:
            from nafsai import __version__
            return __version__
        except Exception:
            return "0.1.0"

    def get_context(self, query: str) -> dict:
        """Gather all context for the LLM."""
        if not query or not query.strip():
            return {"error": "empty_query"}

        query = self.normalizer.normalize(query)

        if not query or not query.strip():
            return {"error": "empty_query"}

        route, confidence = self.router.route(query)

        cached = self.cache.get(query, route)
        if cached:
            return {
                "cached_answer": cached,
                "route":         route,
                "source":        "cache",
            }

        return {
            "route":           route,
            "confidence":      round(confidence, 3),
            "memories":        self.memory.search(query),
            "profile":         self.memory.get_user_profile(query),
            "session_context": self.session.get_context(last_n=3),
            "source":          "memory",
        }

    def build_prompt(self, query: str, ctx: dict) -> str:
        """Build a prompt ready for any LLM."""
        parts = [self.system_prompt, ""]

        if ctx.get("profile"):
            parts.append("User information:")
            for fact in ctx["profile"]:
                parts.append(f" - {fact}")
            parts.append("")

        if ctx.get("memories"):
            parts.append("From memory:")
            for mem in ctx["memories"]:
                parts.append(f" {mem}")
            parts.append("")

        if ctx.get("session_context"):
            parts.append("Current conversation:")
            parts.append(ctx["session_context"])
            parts.append("")

        parts.append(f"Question: {query}")
        return "\n".join(parts)

    def remember(
        self,
        query: str,
        answer: str,
        route: str = "general",
    ) -> dict:
        """Save a conversation to permanent memory."""
        query = self.normalizer.normalize(query)
        if not query:
            return {"error": "empty_query"}

        start = time.time()
        fact  = self.memory.extract_user_fact(query)
        if fact:
            self.memory.save_user_fact(fact)

        self.memory.save(query, answer, route)
        self.session.add(query, answer)
        self.cache.set(query, answer, route)

        return {
            "saved":          True,
            "fact_extracted": fact,
            "duration":       round(time.time() - start, 3),
        }

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """Search permanent memory."""
        query = self.normalizer.normalize(query)
        if not query:
            return []
        return self.memory.search(query, top_k=top_k)

    def chat(self, query: str, llm_fn: Callable) -> str:
        """
        Full pipeline: context → prompt → llm → remember.
        llm_fn: any function that takes str and returns str.
        """
        ctx = self.get_context(query)

        if cached := ctx.get("cached_answer"):
            return cached

        prompt = self.build_prompt(query, ctx)

        try:
            answer = llm_fn(prompt)
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            return "Sorry, an error occurred. Please try again."

        if not answer or not answer.strip():
            return "Sorry, I could not generate a response. Please try again."

        self.remember(query, answer, ctx.get("route", "general"))
        return answer

    def reset(self):
        """Clear session and cache."""
        self.session.clear()
        self.cache.clear()
        log.info("Session and cache cleared ✓")
        print("✓ Session and cache cleared")
