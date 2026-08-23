"""
test_router.py — NafsAI
Tests for the Router class.
"""
import pytest

from nafsai import Router


class TestRouter:

    def test_coding_keyword(self, router):
        route, conf = router.route("def factorial(n):")
        assert route == "coding"
        assert conf >= 0.9

    def test_coding_arabic(self, router):
        route, conf = router.route("اكتب دالة Python")
        assert route == "coding"

    def test_reasoning_arabic(self, router):
        route, conf = router.route("احسب مجموع الأعداد")
        assert route == "reasoning"

    def test_reasoning_english(self, router):
        route, conf = router.route("calculate the probability")
        assert route == "reasoning"

    def test_returns_tuple(self, router):
        result = router.route("ما هو الذكاء الاصطناعي؟")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_confidence_range(self, router):
        _, conf = router.route("أي سؤال عشوائي")
        assert 0.0 <= conf <= 1.0

    def test_valid_categories(self, router):
        valid = {"coding", "reasoning", "general"}
        for q in ["print('hello')", "احسب 2+2", "ما الطقس؟"]:
            route, _ = router.route(q)
            assert route in valid