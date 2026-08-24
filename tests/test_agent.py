"""
test_agent.py — NafsAI
Tests for the Agent class.
"""
import pytest
from unittest.mock import MagicMock, patch

from nafsai import Agent


@pytest.fixture
def agent():
    with patch("nafsai.agent.Memory") as MockMemory, \
         patch("nafsai.agent.Router") as MockRouter, \
         patch("nafsai.agent.Cache") as MockCache, \
         patch("nafsai.agent.Session") as MockSession:

        mock_memory = MagicMock()
        mock_memory.available = True
        mock_memory.search.return_value = []
        mock_memory.get_user_profile.return_value = []
        mock_memory.extract_user_fact.return_value = None
        MockMemory.return_value = mock_memory

        mock_router = MagicMock()
        mock_router.route.return_value = ("general", 0.85)
        mock_router.model = MagicMock()
        MockRouter.return_value = mock_router

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        MockCache.return_value = mock_cache

        mock_session = MagicMock()
        mock_session.get_context.return_value = ""
        MockSession.return_value = mock_session

        a = Agent()
        yield a


class TestAgent:

    def test_chat_basic(self, agent):
        llm = MagicMock(return_value="إجابة من النموذج")
        result = agent.chat("ما اسمك؟", llm)
        assert result == "إجابة من النموذج"

    def test_chat_uses_cache(self, agent):
        agent.cache.get.return_value = "إجابة مخزنة"
        llm = MagicMock()
        result = agent.chat("سؤال مكرر", llm)
        assert result == "إجابة مخزنة"
        llm.assert_not_called()

    def test_chat_llm_error(self, agent):
        llm = MagicMock(side_effect=Exception("LLM crashed"))
        result = agent.chat("سؤال", llm)
        assert "Sorry" in result

    def test_chat_empty_llm_response(self, agent):
        llm = MagicMock(return_value="")
        result = agent.chat("سؤال", llm)
        assert "Sorry" in result

    def test_chat_whitespace_llm_response(self, agent):
        llm = MagicMock(return_value="   ")
        result = agent.chat("سؤال", llm)
        assert "Sorry" in result

    def test_get_context_empty_query(self, agent):
        ctx = agent.get_context("")
        assert ctx.get("error") == "empty_query"

    def test_get_context_whitespace_query(self, agent):
        ctx = agent.get_context("   ")
        assert ctx.get("error") == "empty_query"

    def test_get_context_structure(self, agent):
        ctx = agent.get_context("ما هو Python؟")
        assert "route" in ctx
        assert "source" in ctx

    def test_get_context_from_cache(self, agent):
        agent.cache.get.return_value = "إجابة مخزنة"
        ctx = agent.get_context("سؤال مكرر")
        assert ctx["source"] == "cache"
        assert ctx["cached_answer"] == "إجابة مخزنة"

    def test_get_context_from_memory(self, agent):
        agent.cache.get.return_value = None
        ctx = agent.get_context("سؤال جديد")
        assert ctx["source"] == "memory"
        assert "memories" in ctx
        assert "profile" in ctx
        assert "session_context" in ctx

    def test_build_prompt_contains_question(self, agent):
        ctx = {
            "route": "general",
            "memories": [],
            "profile": [],
            "session_context": "",
        }
        prompt = agent.build_prompt("ما اسمك؟", ctx)
        assert "ما اسمك؟" in prompt
        assert agent.system_prompt in prompt

    def test_build_prompt_with_profile(self, agent):
        ctx = {
            "route": "general",
            "memories": [],
            "profile": ["اسم المستخدم هو: معتز"],
            "session_context": "",
        }
        prompt = agent.build_prompt("تذكر اسمي", ctx)
        assert "معتز" in prompt
        assert "User information:" in prompt

    def test_build_prompt_with_memories(self, agent):
        ctx = {
            "route": "general",
            "memories": ["Q: سؤال سابق\nA: إجابة سابقة"],
            "profile": [],
            "session_context": "",
        }
        prompt = agent.build_prompt("سؤال جديد", ctx)
        assert "From memory:" in prompt
        assert "سؤال سابق" in prompt

    def test_build_prompt_with_session(self, agent):
        ctx = {
            "route": "general",
            "memories": [],
            "profile": [],
            "session_context": "المستخدم: مرحبا\nالمساعد: أهلاً",
        }
        prompt = agent.build_prompt("كيف حالك؟", ctx)
        assert "Current conversation:" in prompt
        assert "مرحبا" in prompt

    def test_remember_returns_dict(self, agent):
        result = agent.remember("سؤال", "إجابة")
        assert isinstance(result, dict)
        assert "saved" in result
        assert result["saved"] is True

    def test_remember_empty_query(self, agent):
        result = agent.remember("", "إجابة")
        assert result.get("error") == "empty_query"

    def test_remember_extracts_fact(self, agent):
        agent.memory.extract_user_fact.return_value = "اسم المستخدم هو: أحمد"
        result = agent.remember("اسمي أحمد", "تم الحفظ")
        assert result["fact_extracted"] == "اسم المستخدم هو: أحمد"
        agent.memory.save_user_fact.assert_called_once()

    def test_remember_no_fact(self, agent):
        agent.memory.extract_user_fact.return_value = None
        result = agent.remember("ما هو Python؟", "لغة برمجة")
        assert result["fact_extracted"] is None
        agent.memory.save_user_fact.assert_not_called()

    def test_recall_empty_query(self, agent):
        result = agent.recall("")
        assert result == []

    def test_recall_calls_memory_search(self, agent):
        agent.memory.search.return_value = ["نتيجة 1", "نتيجة 2"]
        result = agent.recall("سؤال للبحث", top_k=2)
        assert len(result) == 2
        agent.memory.search.assert_called_once()

    def test_reset_clears_session_and_cache(self, agent):
        agent.reset()
        agent.session.clear.assert_called_once()
        agent.cache.clear.assert_called_once()

    def test_version_returns_string(self, agent):
        version = agent._version()
        assert isinstance(version, str)
        assert "." in version
