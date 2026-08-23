"""
Cache — NafsAI
Smart TTL-based cache per question type.
"""
import hashlib
import re
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional


class Cache:
    """
    Simple and fast cache:
    - Different TTL per question type
    - Ignores empty or error responses
    - Thread-safe
    """

    BAD_RESPONSES = {
        "عذراً، لم أتمكن من الإجابة. حاول مرة أخرى.",
        "الخدمة غير متاحة حالياً.",
        "انتهت مهلة الاستجابة.",
        "حدث خطأ غير متوقع.",
    }

    TTL_RULES = [
        (
            ["news", "price", "now", "today", "اليوم", "الآن", "سعر", "أخبار"],
            timedelta(minutes=5),
        ),
        (
            ["معادلة", "احسب", "calculate", "equation", "math"],
            timedelta(hours=24),
        ),
        (
            ["برمجة", "كود", "code", "function", "debug"],
            timedelta(hours=4),
        ),
    ]

    DEFAULT_TTL = timedelta(minutes=30)

    def __init__(self):
        self._cache: dict = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def _ttl(self, query: str, route: str) -> timedelta:
        q = query.lower()
        for keywords, ttl in self.TTL_RULES:
            if any(k in q for k in keywords):
                return ttl
        if route == "reasoning":
            return timedelta(hours=24)
        if route == "coding":
            return timedelta(hours=4)
        return self.DEFAULT_TTL

    def _normalize_for_key(self, text: str) -> str:
        text = text.strip().lower()
        for original, unified in [
            ("أ", "ا"), ("إ", "ا"), ("آ", "ا"),
            ("ة", "ه"), ("ى", "ي"),
        ]:
            text = text.replace(original, unified)
        text = re.sub(r"[؟!?،,.\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _key(self, query: str, route: str = "general") -> str:
        normalized = self._normalize_for_key(query)
        combined = f"{normalized}|{route}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, query: str, route: str = "general") -> Optional[str]:
        key = self._key(query, route)
        ttl = self._ttl(query, route)
        with self._lock:
            if key in self._cache:
                result, ts = self._cache[key]
                if datetime.now() - ts < ttl:
                    self.hits += 1
                    return result
                del self._cache[key]
        self.misses += 1
        return None

    def set(self, query: str, result: str, route: str = "general"):
        if not result or not result.strip():
            return
        if result.strip() in self.BAD_RESPONSES:
            return
        with self._lock:
            self._cache[self._key(query, route)] = (result, datetime.now())

    def clear(self):
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / total * 100:.1f}%" if total else "0%",
            "cached": len(self._cache),
        }