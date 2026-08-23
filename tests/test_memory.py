"""
test_memory.py — NafsAI
Tests for Memory — SQLite + FTS5 + sqlite-vec.
Covers all three modes: full, fts_only, numpy_only.
"""
import pytest
from nafsai import Memory


class TestMemoryBasic:
    """Basic tests that work across all memory modes."""

    def test_memory_initializes(self, memory):
        assert memory.available is True
        assert memory.mode in ("full", "fts_only", "numpy_only")

    def test_mode_is_valid(self, memory):
        assert memory.mode in ("full", "fts_only", "numpy_only")

    def test_save_and_search_returns_results(self, memory):
        memory.save("ما هو Python؟", "لغة برمجة عالية المستوى")
        results = memory.search("Python")
        assert isinstance(results, list)

    def test_search_empty_query_returns_empty(self, memory):
        assert memory.search("") == []
        assert memory.search("   ") == []

    def test_save_empty_query_does_not_crash(self, memory):
        memory.save("", "إجابة")
        memory.save("سؤال", "")

    def test_get_history_returns_list(self, memory):
        memory.save("سؤال", "إجابة")
        history = memory.get_history()
        assert isinstance(history, list)

    def test_get_history_structure(self, memory):
        memory.save("سؤال تاريخ", "إجابة تاريخ")
        history = memory.get_history(limit=1)
        if history:
            assert "query" in history[0]
            assert "answer" in history[0]
            assert "timestamp" in history[0]

    def test_get_history_limit(self, memory):
        for i in range(5):
            memory.save(f"سؤال {i}", f"إجابة {i}")
        history = memory.get_history(limit=3)
        assert len(history) <= 3

    def test_cleanup_does_not_crash(self, memory):
        memory.save("سؤال", "إجابة")
        memory.cleanup(days_threshold=0)


class TestUserFacts:
    """User profile extraction and storage tests."""

    def test_extract_arabic_name(self, memory):
        fact = memory.extract_user_fact("اسمي أحمد")
        assert fact is not None
        assert "احمد" in fact.lower() or "أحمد" in fact

    def test_extract_english_name(self, memory):
        fact = memory.extract_user_fact("my name is Ahmed")
        assert fact is not None
        assert "Ahmed" in fact

    def test_extract_location(self, memory):
        fact = memory.extract_user_fact("اسكن في جدة")
        assert fact is not None
        assert "جده" in fact or "جدة" in fact

    def test_extract_job(self, memory):
        fact = memory.extract_user_fact("اعمل في التقنية")
        assert fact is not None
        assert "التقنيه" in fact or "التقنية" in fact

    def test_extract_no_match_returns_none(self, memory):
        fact = memory.extract_user_fact("ما هو الطقس اليوم؟")
        assert fact is None

    def test_save_and_get_user_fact(self, memory):
        memory.save_user_fact("اسم المستخدم هو: عبدالله")
        profile = memory.get_user_profile("اسم المستخدم")
        assert isinstance(profile, list)

    def test_get_user_profile_empty_query(self, memory):
        result = memory.get_user_profile("")
        assert result == []

    def test_save_empty_fact_does_not_crash(self, memory):
        memory.save_user_fact("")


class TestNormalization:
    """Internal Arabic normalization tests."""

    def test_normalize_arabic_numbers(self, memory):
        result = memory._normalize("عمري ٢٥ سنة")
        assert "25" in result
        assert "٢٥" not in result

    def test_normalize_hamza(self, memory):
        result = memory._normalize("أحمد")
        assert result == "احمد"

    def test_normalize_taa_marbouta(self, memory):
        result = memory._normalize("مدرسة")
        assert result == "مدرسه"

    def test_normalize_diacritics(self, memory):
        result = memory._normalize("مُحَمَّد")
        assert result == "محمد"

    def test_normalize_empty(self, memory):
        assert memory._normalize("") == ""

    def test_normalize_spaces(self, memory):
        result = memory._normalize("مرحبا   كيف   حالك")
        assert result == "مرحبا كيف حالك"


class TestImportanceAndAging:
    """Importance scoring and aging decay tests."""

    def test_calc_importance_keywords(self, memory):
        score = memory._calc_importance("تذكر هذا مهم", "معلومة مهمة جداً")
        assert score > 0.5

    def test_calc_importance_range(self, memory):
        score = memory._calc_importance("سؤال", "إجابة")
        assert 0.0 <= score <= 1.0

    def test_calc_importance_long_answer(self, memory):
        long = "كلمة " * 100
        score = memory._calc_importance("سؤال", long)
        assert 0.0 <= score <= 1.0

    def test_calc_aging_recent(self, memory):
        from datetime import datetime
        now = datetime.now().isoformat()
        score = memory._calc_aging(now, importance=1.0)
        assert score > 0.9

    def test_calc_aging_bad_timestamp(self, memory):
        score = memory._calc_aging("invalid_timestamp", importance=0.5)
        assert score == 0.3

    def test_calc_aging_range(self, memory):
        from datetime import datetime
        now = datetime.now().isoformat()
        score = memory._calc_aging(now, importance=0.5)
        assert 0.0 <= score <= 1.0


class TestContentId:
    """Content-derived ID tests."""

    def test_same_content_same_id(self, memory):
        id1 = memory._content_id("سؤال", "إجابة")
        id2 = memory._content_id("سؤال", "إجابة")
        assert id1 == id2

    def test_different_content_different_id(self, memory):
        id1 = memory._content_id("سؤال 1")
        id2 = memory._content_id("سؤال 2")
        assert id1 != id2

    def test_id_is_valid_uuid(self, memory):
        import uuid
        result = memory._content_id("test")
        uuid.UUID(result)  # raises if not a valid UUID


class TestFallbackModes:
    """Dynamic fallback mode tests."""

    def test_fts_only_mode_search(self, memory_no_vec):
        memory_no_vec.save("اسمي عبدالله", "تم الحفظ")
        results = memory_no_vec.search("عبدالله")
        assert isinstance(results, list)

    def test_numpy_only_mode_search(self, memory_numpy_only):
        memory_numpy_only.save("اسمي محمد", "تم الحفظ")
        results = memory_numpy_only.search("محمد")
        assert isinstance(results, list)

    def test_fts_only_user_profile(self, memory_no_vec):
        memory_no_vec.save_user_fact("اسم المستخدم هو: فهد")
        profile = memory_no_vec.get_user_profile("اسم")
        assert isinstance(profile, list)

    def test_numpy_only_user_profile(self, memory_numpy_only):
        memory_numpy_only.save_user_fact("اسم المستخدم هو: سارة")
        profile = memory_numpy_only.get_user_profile("اسم")
        assert isinstance(profile, list)

    def test_all_modes_return_same_type(self, memory, memory_no_vec, memory_numpy_only):
        """All modes must return list — unified API."""
        for mem in [memory, memory_no_vec, memory_numpy_only]:
            mem.save("سؤال اختبار", "إجابة اختبار")
            result = mem.search("اختبار")
            assert isinstance(result, list)

    def test_search_returns_empty_not_error(self, memory):
        """No exception raised when no data exists."""
        result = memory.search("سؤال لم يُحفظ أبداً")
        assert isinstance(result, list)

    def test_save_multiple_and_search(self, memory):
        """Saves multiple conversations and searches across them."""
        data = [
            ("ما هو Python؟", "لغة برمجة"),
            ("ما هو FastAPI؟", "إطار عمل سريع"),
            ("كيف أتعلم البرمجة؟", "ابدأ بالأساسيات"),
        ]
        for q, a in data:
            memory.save(q, a)
        results = memory.search("Python")
        assert isinstance(results, list)


class TestPersistence:
    """Persistence tests using a real file."""

    def test_data_persists_across_instances(self, mock_model, tmp_path):
        """Data survives after closing an instance."""
        db_path = str(tmp_path / "test_memory.db")

        mem1 = Memory(db_path=db_path, _shared_model=mock_model)
        mem1.save("اسمي عبدالله", "تم الحفظ")

        mem2 = Memory(db_path=db_path, _shared_model=mock_model)
        history = mem2.get_history(limit=10)

        assert len(history) >= 1
        assert any("عبدالله" in h["query"] for h in history)

    def test_user_facts_persist(self, mock_model, tmp_path):
        """User profile survives after closing an instance."""
        db_path = str(tmp_path / "test_facts.db")

        mem1 = Memory(db_path=db_path, _shared_model=mock_model)
        mem1.save_user_fact("اسم المستخدم هو: فيصل")

        mem2 = Memory(db_path=db_path, _shared_model=mock_model)
        profile = mem2.get_user_profile("اسم")
        assert isinstance(profile, list)