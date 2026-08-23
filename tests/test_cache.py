"""
test_cache.py — NafsAI
Tests for the Cache class.
"""
import pytest

from nafsai import Cache


class TestCache:

    def test_set_and_get(self, cache):
        cache.set("ما اسمك؟", "أنا NafsAI", route="general")
        result = cache.get("ما اسمك؟", route="general")
        assert result == "أنا NafsAI"

    def test_cache_miss(self, cache):
        result = cache.get("سؤال غير موجود", route="general")
        assert result is None

    def test_hit_rate_tracking(self, cache):
        cache.set("سؤال", "إجابة", route="general")
        cache.get("سؤال", route="general")        # hit
        cache.get("سؤال آخر", route="general")    # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_route_isolation(self, cache):
        """Same question, different route = separate cache entry."""
        cache.set("اشرح الكود", "إجابة برمجة", route="coding")
        result = cache.get("اشرح الكود", route="general")
        assert result is None

    def test_bad_response_not_cached(self, cache):
        cache.set("سؤال", "عذراً، لم أتمكن من الإجابة. حاول مرة أخرى.", route="general")
        assert cache.get("سؤال", route="general") is None

    def test_empty_response_not_cached(self, cache):
        cache.set("سؤال", "", route="general")
        assert cache.get("سؤال", route="general") is None

    def test_clear(self, cache):
        cache.set("سؤال", "إجابة", route="general")
        cache.clear()
        assert cache.get("سؤال", route="general") is None

    def test_stats_structure(self, cache):
        stats = cache.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "cached" in stats

    def test_arabic_normalization_same_key(self, cache):
        """Hamza variants in the same question must resolve to the same key."""
        cache.set("أين تسكن؟", "في الرياض", route="general")
        result = cache.get("اين تسكن؟", route="general")
        assert result == "في الرياض"