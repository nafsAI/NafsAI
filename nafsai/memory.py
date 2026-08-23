"""
Memory — NafsAI
Permanent local memory: SQLite + FTS5 + sqlite-vec with smart fallback.
"""
import hashlib
import logging
import math
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("nafsai.memory")


class Memory:
    """
    Permanent memory — single local file.

    Auto-detected modes:
        full       → SQLite + FTS5 + sqlite-vec  (best)
        fts_only   → SQLite + FTS5               (good)
        numpy_only → numpy cosine similarity     (always works)
    """

    USER_FACT_PATTERNS = [
        (r"اسمي\s+(\w+)",                          "اسم المستخدم هو: {}"),
        (r"my name is\s+(\w+)",                    "اسم المستخدم هو: {}"),
        (r"انا\s+(مطور|مبرمج|طالب|مهندس|دكتور|معلم)\s*(\w*)", "المستخدم هو: {}"),
        (r"I(?:'m| am) (?:a |an )?(\w+)",          "المستخدم هو: {}"),
        (r"اعمل\s+(في|كـ):?\s*(\w+)",              "المستخدم يعمل في: {}"),
        (r"I work (?:at|in|as) (\w+)",             "المستخدم يعمل في: {}"),
        (r"اسكن\s+(في):?\s*(\w+)",                 "المستخدم يسكن في: {}"),
        (r"I live (?:in|at) (\w+)",                "المستخدم يسكن في: {}"),
    ]

    _ARABIC_NUMS  = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    _ARABIC_CHARS = [
        ("أ", "ا"), ("إ", "ا"), ("آ", "ا"),
        ("ة", "ه"), ("ى", "ي"),
    ]
    _DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

    def _normalize(self, text: str) -> str:
        """Internal Arabic normalization for FTS5 — standalone."""
        if not text or not text.strip():
            return text
        text = text.strip()
        text = text.translate(self._ARABIC_NUMS)
        text = self._DIACRITICS.sub("", text)
        for orig, unified in self._ARABIC_CHARS:
            text = text.replace(orig, unified)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ─── init ─────────────────────────────────────────────────────────────────

    def __init__(
        self,
        db_path: str = "./nafsai_memory.db",
        model_name: str = "intfloat/multilingual-e5-base",
        _shared_model: Optional[SentenceTransformer] = None,
    ):
        if _shared_model is None:
            print("NafsAI — Loading model (once only)...")
            print("First run may take 30-60 seconds...")

        self.model = _shared_model or SentenceTransformer(model_name)

        if _shared_model is None:
            print("Model ready\n")

        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")

        self.mode      = self._detect_mode()
        self.available = True
        self._ensure_schema()

        icons = {"full": "✓", "fts_only": "◑", "numpy_only": "○"}
        names = {
            "full":       "Full memory  (FTS5 + vectors)",
            "fts_only":   "Good memory  (FTS5 only)",
            "numpy_only": "Basic memory (numpy)",
        }
        print(f" {icons[self.mode]} {names[self.mode]}")
        log.info(f"Memory ready — mode: {self.mode}")

    # ─── mode detection ───────────────────────────────────────────────────────

    def _detect_mode(self) -> str:
        try:
            import sqlite_vec
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS "
                "_vec_probe USING vec0(v FLOAT[4])"
            )
            self.db.execute(
                "INSERT INTO _vec_probe(rowid, v) VALUES (1, '[1,2,3,4]')"
            )
            self.db.execute("DELETE FROM _vec_probe WHERE rowid = 1")
            self.db.execute("DROP TABLE IF EXISTS _vec_probe")
            self.db.commit()
            return "full"
        except Exception as e:
            log.debug(f"sqlite-vec not available: {e}")
            self.db.execute("DROP TABLE IF EXISTS _vec_probe")
            self.db.commit()

        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS "
                "_fts_probe USING fts5(t)"
            )
            self.db.execute("INSERT INTO _fts_probe(t) VALUES ('test')")
            self.db.execute("DROP TABLE IF EXISTS _fts_probe")
            self.db.commit()
            return "fts_only"
        except Exception as e:
            log.debug(f"FTS5 not available: {e}")
            self.db.execute("DROP TABLE IF EXISTS _fts_probe")
            self.db.commit()
            return "numpy_only"

    # ─── schema ───────────────────────────────────────────────────────────────

    def _ensure_schema(self):
        """Creates all required tables for the detected mode."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                query      TEXT NOT NULL,
                answer     TEXT NOT NULL,
                route      TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                timestamp  TEXT NOT NULL,
                date       TEXT,
                time       TEXT
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id         TEXT PRIMARY KEY,
                fact       TEXT NOT NULL,
                importance REAL DEFAULT 0.95,
                timestamp  TEXT NOT NULL
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id        TEXT PRIMARY KEY,
                vec_blob  BLOB NOT NULL,
                ref_table TEXT NOT NULL
            )
        """)

        if self.mode in ("full", "fts_only"):
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(id UNINDEXED, query, answer)
            """)
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS profiles_fts
                USING fts5(id UNINDEXED, fact)
            """)

        if self.mode == "full":
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_vec
                USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[768])
            """)
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS profiles_vec
                USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[768])
            """)

        self.db.commit()

    # ─── save ─────────────────────────────────────────────────────────────────

    def save(
        self,
        query: str,
        answer: str,
        route: str = "general",
    ):
        """Save a conversation to permanent memory."""
        if not query or not answer:
            return
        try:
            now        = datetime.now()
            importance = self._calc_importance(query, answer)
            record_id  = self._content_id(query, answer[:200])
            q_norm     = self._normalize(query)
            a_norm     = self._normalize(answer)

            self.db.execute("""
                INSERT OR REPLACE INTO conversations
                (id, query, answer, route, importance, timestamp, date, time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id, query, answer, route, importance,
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
            ))

            if self.mode in ("full", "fts_only"):
                try:
                    self.db.execute(
                        "INSERT OR REPLACE INTO conversations_fts"
                        "(id, query, answer) VALUES (?, ?, ?)",
                        (record_id, q_norm, a_norm)
                    )
                except Exception as e:
                    log.warning(f"FTS5 insert failed: {e}")

            vec = self.model.encode(
                f"query: {q_norm}\npassage: {a_norm[:500]}"
            ).astype(np.float32)

            if self.mode == "full":
                try:
                    self.db.execute(
                        "INSERT OR REPLACE INTO conversations_vec"
                        "(id, embedding) VALUES (?, ?)",
                        (record_id, vec.tobytes())
                    )
                except Exception as e:
                    log.warning(f"vec0 insert failed, fallback to embeddings: {e}")
                    self.db.execute(
                        "INSERT OR REPLACE INTO embeddings"
                        "(id, vec_blob, ref_table) VALUES (?, ?, ?)",
                        (record_id, vec.tobytes(), "conversations")
                    )
            else:
                self.db.execute(
                    "INSERT OR REPLACE INTO embeddings"
                    "(id, vec_blob, ref_table) VALUES (?, ?, ?)",
                    (record_id, vec.tobytes(), "conversations")
                )

            self.db.commit()
            log.info(f"Saved — mode: {self.mode} | id: {record_id[:8]}")

        except Exception as e:
            log.warning(f"Memory save failed: {e}")

    def save_user_fact(self, fact: str, importance: float = 0.95):
        """Save a user fact to the profile."""
        if not fact:
            return
        try:
            fact_id   = self._content_id(fact)
            fact_norm = self._normalize(fact)
            now       = datetime.now().isoformat()

            self.db.execute("""
                INSERT OR REPLACE INTO user_profiles
                (id, fact, importance, timestamp)
                VALUES (?, ?, ?, ?)
            """, (fact_id, fact, importance, now))

            if self.mode in ("full", "fts_only"):
                try:
                    self.db.execute(
                        "INSERT OR REPLACE INTO profiles_fts"
                        "(id, fact) VALUES (?, ?)",
                        (fact_id, fact_norm)
                    )
                except Exception as e:
                    log.warning(f"FTS5 profile insert failed: {e}")

            vec = self.model.encode(
                f"query: {fact_norm}"
            ).astype(np.float32)

            if self.mode == "full":
                try:
                    self.db.execute(
                        "INSERT OR REPLACE INTO profiles_vec"
                        "(id, embedding) VALUES (?, ?)",
                        (fact_id, vec.tobytes())
                    )
                except Exception as e:
                    log.warning(f"vec0 profile insert failed, fallback: {e}")
                    self.db.execute(
                        "INSERT OR REPLACE INTO embeddings"
                        "(id, vec_blob, ref_table) VALUES (?, ?, ?)",
                        (fact_id, vec.tobytes(), "user_profiles")
                    )
            else:
                self.db.execute(
                    "INSERT OR REPLACE INTO embeddings"
                    "(id, vec_blob, ref_table) VALUES (?, ?, ?)",
                    (fact_id, vec.tobytes(), "user_profiles")
                )

            self.db.commit()
            log.info(f"Saved user fact: {fact}")

        except Exception as e:
            log.warning(f"save_user_fact failed: {e}")

    # ─── search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search permanent memory."""
        if not query:
            return []

        q_norm = self._normalize(query)

        if self.mode == "full":
            results = self._hybrid_search(q_norm, top_k)
            if results:
                return results
            results = self._fts_search(q_norm, top_k)
            if results:
                return results
            return self._numpy_search(q_norm, top_k, table="conversations")

        elif self.mode == "fts_only":
            results = self._fts_search(q_norm, top_k)
            if results:
                return results
            return self._numpy_search(q_norm, top_k, table="conversations")

        else:
            return self._numpy_search(q_norm, top_k, table="conversations")

    def _hybrid_search(self, query: str, top_k: int) -> list[str]:
        """FTS5 + vec0 fused with RRF scoring."""
        try:
            vec = self.model.encode(
                f"query: {query}"
            ).astype(np.float32).tobytes()

            rows = self.db.execute("""
                WITH fts AS (
                    SELECT id,
                           ROW_NUMBER() OVER (ORDER BY rank) AS rn
                    FROM conversations_fts
                    WHERE conversations_fts MATCH ?
                    LIMIT ?
                ),
                vec AS (
                    SELECT id,
                           ROW_NUMBER() OVER (ORDER BY distance) AS rn
                    FROM conversations_vec
                    WHERE embedding MATCH ?
                    AND k = ?
                ),
                fused AS (
                    SELECT COALESCE(f.id, v.id) AS id,
                           COALESCE(1.0/(60+f.rn), 0) +
                           COALESCE(1.0/(60+v.rn), 0) AS score
                    FROM fts f
                    FULL OUTER JOIN vec v ON f.id = v.id
                )
                SELECT c.query, c.answer, c.timestamp, c.importance
                FROM fused
                JOIN conversations c ON c.id = fused.id
                ORDER BY fused.score DESC
                LIMIT ?
            """, (query, top_k * 3, vec, top_k * 3, top_k)).fetchall()

            return self._format_results(rows)

        except Exception as e:
            log.warning(f"Hybrid search failed: {e}")
            return []

    def _fts_search(self, query: str, top_k: int) -> list[str]:
        """FTS5 text search only."""
        try:
            rows = self.db.execute("""
                SELECT c.query, c.answer, c.timestamp, c.importance
                FROM conversations_fts f
                JOIN conversations c ON c.id = f.id
                WHERE conversations_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, top_k)).fetchall()
            return self._format_results(rows)
        except Exception as e:
            log.warning(f"FTS5 search failed: {e}")
            return []

    def _numpy_search(
        self,
        query: str,
        top_k: int,
        table: str = "conversations",
    ) -> list[str]:
        """Cosine similarity search using numpy."""
        try:
            q_vec = self.model.encode(
                f"query: {query}"
            ).astype(np.float32)

            rows = self.db.execute("""
                SELECT e.id, e.vec_blob
                FROM embeddings e
                WHERE e.ref_table = ?
                ORDER BY rowid DESC
                LIMIT 2000
            """, (table,)).fetchall()

            if not rows:
                return []

            ids    = [r[0] for r in rows]
            blobs  = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
            matrix = np.stack(blobs)
            norms  = (
                np.linalg.norm(matrix, axis=1) *
                np.linalg.norm(q_vec) + 1e-9
            )
            scores  = matrix @ q_vec / norms
            top_idx = np.argsort(scores)[::-1][:top_k]
            top_ids = [ids[i] for i in top_idx if scores[i] > 0.3]

            if not top_ids:
                return []

            placeholders = ",".join("?" * len(top_ids))

            if table == "conversations":
                rows_data = self.db.execute(
                    f"SELECT query, answer, timestamp, importance "
                    f"FROM conversations WHERE id IN ({placeholders})",
                    top_ids
                ).fetchall()
                return self._format_results(rows_data)
            else:
                rows_data = self.db.execute(
                    f"SELECT fact FROM user_profiles "
                    f"WHERE id IN ({placeholders})",
                    top_ids
                ).fetchall()
                return [r[0] for r in rows_data]

        except Exception as e:
            log.warning(f"numpy search failed: {e}")
            return []

    # ─── user profile ─────────────────────────────────────────────────────────

    def get_user_profile(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve user facts with dynamic fallback."""
        if not query:
            return []

        q_norm = self._normalize(query)

        if self.mode == "full":
            try:
                vec = self.model.encode(
                    f"query: {q_norm}"
                ).astype(np.float32).tobytes()

                rows = self.db.execute("""
                    SELECT p.fact
                    FROM profiles_vec v
                    JOIN user_profiles p ON p.id = v.id
                    WHERE embedding MATCH ?
                    AND k = ?
                    ORDER BY distance
                    LIMIT ?
                """, (vec, top_k, top_k)).fetchall()

                if rows:
                    return [r[0] for r in rows]

            except Exception as e:
                log.warning(f"profiles_vec search failed: {e}")

        if self.mode in ("full", "fts_only"):
            try:
                rows = self.db.execute("""
                    SELECT p.fact
                    FROM profiles_fts f
                    JOIN user_profiles p ON p.id = f.id
                    WHERE profiles_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (q_norm, top_k)).fetchall()

                if rows:
                    return [r[0] for r in rows]

            except Exception as e:
                log.warning(f"profiles_fts search failed: {e}")

        return self._numpy_search(q_norm, top_k, table="user_profiles")

    # ─── extract ──────────────────────────────────────────────────────────────

    def extract_user_fact(self, query: str) -> Optional[str]:
        """Extract a user fact from a message using regex patterns."""
        for pattern, template in self.USER_FACT_PATTERNS:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                value = m.group(1).strip() if m.lastindex else ""
                return template.format(value)
        return None

    # ─── history & cleanup ────────────────────────────────────────────────────

    def get_history(self, limit: int = 20) -> list[dict]:
        """Retrieve recent conversation history."""
        try:
            rows = self.db.execute("""
                SELECT query, answer, route, timestamp, date, time
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [
                {
                    "query":     r[0],
                    "answer":    r[1],
                    "route":     r[2],
                    "timestamp": r[3],
                    "date":      r[4],
                    "time":      r[5],
                }
                for r in rows
            ]
        except Exception as e:
            log.warning(f"get_history failed: {e}")
            return []

    def cleanup(self, days_threshold: int = 30):
        """Delete conversations older than N days."""
        try:
            cutoff = (
                datetime.now() - timedelta(days=days_threshold)
            ).isoformat()

            self.db.execute(
                "DELETE FROM conversations WHERE timestamp < ?", (cutoff,)
            )
            self.db.execute("""
                DELETE FROM embeddings
                WHERE ref_table = 'conversations'
                AND id NOT IN (SELECT id FROM conversations)
            """)
            self.db.execute("""
                DELETE FROM embeddings
                WHERE ref_table = 'user_profiles'
                AND id NOT IN (SELECT id FROM user_profiles)
            """)
            self.db.commit()
            log.info(f"Cleaned memories older than {days_threshold} days")

        except Exception as e:
            log.warning(f"cleanup failed: {e}")

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _format_results(self, rows) -> list[str]:
        """Sort results by aging score."""
        scored = []
        for row in rows:
            q, a, ts, imp = row[0], row[1], row[2], float(row[3])
            scored.append({
                "text":  f"Q: {q}\nA: {a}",
                "score": self._calc_aging(ts, imp),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [r["text"] for r in scored]

    def _content_id(self, *parts: str) -> str:
        combined = "|".join(parts)
        return str(uuid.UUID(hashlib.md5(combined.encode()).hexdigest()))

    def _calc_importance(self, query: str, answer: str) -> float:
        keywords = [
            "احفظ", "تذكر", "مهم", "خطأ",
            "error", "critical", "remember", "important",
        ]
        score = 0.5
        text  = (query + " " + answer).lower()
        score += min(sum(0.1 for k in keywords if k in text), 0.3)
        n = len(answer)
        if 100 <= n <= 500:
            score += 0.15
        elif n > 500:
            score += 0.05
        return min(score, 1.0)

    def _calc_aging(
        self, timestamp_str: str, importance: float = 0.5
    ) -> float:
        try:
            age_hrs = max(
                (
                    datetime.now() -
                    datetime.fromisoformat(timestamp_str)
                ).total_seconds() / 3600,
                0,
            )
            return round(
                importance * math.exp(-0.693 * age_hrs / 72), 4
            )
        except Exception:
            return 0.3