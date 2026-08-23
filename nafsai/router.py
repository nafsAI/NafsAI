"""
Router — NafsAI
Smart question routing: coding / reasoning / general
"""
import logging
import re
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("nafsai.router")


class Router:
    """
    Classifies each question into a category:
    - coding    : programming and technical questions
    - reasoning : calculations, logic, and analysis
    - general   : everything else
    """

    PROFILES = {
        "coding": [
            "اكتب كود برمجي",
            "اكتب دالة بايثون",
            "write a function",
            "أصلح هذا الكود",
            "debug this error",
            "implement an algorithm",
            "SQL query database",
            "REST API endpoint",
            "Docker container setup",
        ],
        "reasoning": [
            "احسب ناتج هذه المعادلة",
            "أثبت أن هذه النظرية صحيحة",
            "قارن بين خيارين",
            "لماذا تدور الأرض",
            "solve this equation",
            "calculate the probability",
            "compare and contrast",
        ],
        "general": [
            "ما هي عاصمة هذا البلد",
            "اشرح ما هو الذكاء الاصطناعي",
            "اكتب قصيدة",
            "ترجم هذه الجملة",
            "what is the capital",
            "explain machine learning",
            "write a short poem",
        ],
    }

    KEYWORD_RULES = [
        (
            r"\b(def |class |import |print\(|return |factorial|fibonacci)\b",
            "coding",
        ),
        (
            r"\b(اكتب|write)\b.{0,30}\b(function|دالة|script|كود)\b",
            "coding",
        ),
        (
            r"^(قارن|ما\s+الفرق|compare|what.s\s+the\s+difference)",
            "reasoning",
        ),
        (
            r"^(لماذا|ما\s+سبب|why\s+does)",
            "reasoning",
        ),
        (
            r"\b(احسب|حل|solve|calculate|prove|أثبت)\b",
            "reasoning",
        ),
    ]

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        _shared_model: Optional[SentenceTransformer] = None,
    ):
        if _shared_model:
            self.model = _shared_model
        else:
            log.info("Loading router model...")
            self.model = SentenceTransformer(model_name)

        self._build_centroids()
        log.info("Router ready ✓")

    def _build_centroids(self):
        self.centroids = {}
        for name, sentences in self.PROFILES.items():
            prefixed = [f"query: {s}" for s in sentences]
            vecs = self.model.encode(
                prefixed,
                batch_size=32,
                show_progress_bar=False,
            )
            self.centroids[name] = vecs.mean(axis=0)

    def route(self, query: str) -> Tuple[str, float]:
        """Returns (category, confidence)."""
        for pattern, category in self.KEYWORD_RULES:
            if re.search(pattern, query, re.IGNORECASE):
                return category, 0.92

        vec = self.model.encode(f"query: {query}")

        scores = {}
        for name, c in self.centroids.items():
            dot = float(np.dot(vec, c))
            norm = float(np.linalg.norm(vec)) * float(np.linalg.norm(c)) + 1e-9
            scores[name] = dot / norm

        best = max(scores, key=scores.__getitem__)
        conf = scores[best]
        return best, conf