"""
Session — NafsAI
Manages current conversation context.
"""
from threading import Lock


class Session:
    """
    Stores the last N turns of the current session.
    Cleared on reset or program exit.
    """

    def __init__(self, max_turns: int = 10):
        self._session: list = []
        self._lock = Lock()
        self.max_turns = max_turns

    def add(self, query: str, answer: str):
        with self._lock:
            self._session.append({
                "q": query,
                "a": answer[:200],
            })
            if len(self._session) > self.max_turns:
                self._session.pop(0)

    def get_context(self, last_n: int = 3) -> str:
        with self._lock:
            if not self._session:
                return ""
            lines = []
            for turn in self._session[-last_n:]:
                lines.append(f"المستخدم: {turn['q']}")
                lines.append(f"المساعد: {turn['a']}")
            return "\n".join(lines)

    def get_last_topic(self) -> str:
        with self._lock:
            for turn in reversed(self._session):
                if len(turn["q"].split()) >= 4:
                    return turn["q"]
            return ""

    def count(self) -> int:
        with self._lock:
            return len(self._session)

    def clear(self):
        with self._lock:
            self._session.clear()