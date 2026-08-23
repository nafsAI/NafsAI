"""
test_session.py — NafsAI
Tests for the Session class.
"""
import pytest

from nafsai import Session


class TestSession:

    def test_add_and_context(self, session):
        session.add("كيف حالك؟", "بخير شكراً")
        ctx = session.get_context()
        assert "كيف حالك؟" in ctx
        assert "بخير شكراً" in ctx

    def test_max_turns_limit(self, session):
        for i in range(10):
            session.add(f"سؤال {i}", f"إجابة {i}")
        assert session.count() == 5  # max_turns=5 in conftest

    def test_last_n_context(self, session):
        session.add("سؤال قديم", "إجابة قديمة")
        session.add("سؤال حديث", "إجابة حديثة")
        ctx = session.get_context(last_n=1)
        assert "سؤال حديث" in ctx
        assert "سؤال قديم" not in ctx

    def test_empty_context(self, session):
        assert session.get_context() == ""

    def test_clear(self, session):
        session.add("سؤال", "إجابة")
        session.clear()
        assert session.count() == 0
        assert session.get_context() == ""

    def test_get_last_topic(self, session):
        session.add("سؤال قصير", "إجابة")
        session.add("ما هو الذكاء الاصطناعي وكيف يعمل؟", "إجابة مفصلة")
        topic = session.get_last_topic()
        assert "الذكاء الاصطناعي" in topic

    def test_answer_truncated(self, session):
        long_answer = "إ" * 500
        session.add("سؤال", long_answer)
        ctx = session.get_context()
        # answer is stored with a max of 200 characters
        assert len(ctx) < 600

    def test_fstring_format(self, session):
        """Verifies that f-strings in get_context() produce correct output."""
        session.add("ما اسمك؟", "أنا NafsAI")
        ctx = session.get_context()
        assert "المستخدم: ما اسمك؟" in ctx
        assert "المساعد: أنا NafsAI" in ctx