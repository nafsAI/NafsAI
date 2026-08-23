"""
test_normalizer.py — NafsAI
Tests for the Normalizer class.
"""
import pytest

from nafsai import Normalizer


class TestNormalizer:

    def test_arabic_numbers(self, normalizer):
        result = normalizer.normalize("عمري ٢٥ سنة")
        assert "25" in result
        assert "٢٥" not in result

    def test_hamza_unification(self, normalizer):
        assert normalizer.normalize("أحمد") == "احمد"
        assert normalizer.normalize("إبراهيم") == "ابراهيم"
        assert normalizer.normalize("آمل") == "امل"

    def test_taa_marbouta(self, normalizer):
        assert normalizer.normalize("مدرسة") == "مدرسه"

    def test_alef_maqsura(self, normalizer):
        assert normalizer.normalize("يحيى") == "يحيي"

    def test_diacritics_removal(self, normalizer):
        result = normalizer.normalize("مُحَمَّد")
        assert result == "محمد"

    def test_math_symbols(self, normalizer):
        result = normalizer.normalize("٢ × ٣")
        assert "*" in result
        assert "×" not in result

    def test_extra_spaces(self, normalizer):
        result = normalizer.normalize("مرحبا   كيف   حالك")
        assert result == "مرحبا كيف حالك"

    def test_empty_string(self, normalizer):
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == "   "

    def test_mixed_arabic_english(self, normalizer):
        result = normalizer.normalize("أحب Python جداً")
        assert "Python" in result
        assert "احب" in result