"""
Normalizer — NafsAI
Cleans and unifies Arabic text before processing.
"""
import re


class Normalizer:
    """
    Normalizes Arabic text:
    - Converts Arabic numerals to Western
    - Unifies Hamza forms and Taa Marbouta
    - Removes diacritics (Tashkeel)
    - Cleans extra spaces and repeated symbols
    """

    ARABIC_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    ARABIC_CHARS = [
        ("أ", "ا"),
        ("إ", "ا"),
        ("آ", "ا"),
        ("ة", "ه"),
        ("ى", "ي"),
    ]

    DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

    def normalize(self, text: str) -> str:
        """Normalize Arabic text for search and processing."""
        if not text or not text.strip():
            return text

        text = text.strip()

        # Arabic numerals → Western
        text = text.translate(self.ARABIC_NUMS)

        # Remove diacritics
        text = self.DIACRITICS.sub("", text)

        # Unify characters
        for original, unified in self.ARABIC_CHARS:
            text = text.replace(original, unified)

        # Math symbols
        text = (
            text.replace("×", " * ")
            .replace("÷", " / ")
            .replace("−", "-")
        )

        # Clean repeated spaces and symbols
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r",{2,}", ",", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[.]{3,}", "...", text)

        return text.strip()