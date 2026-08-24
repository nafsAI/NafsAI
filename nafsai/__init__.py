"""
NafsAI — The Arabic-First Memory Layer for Local AI
"""
import sys

if sys.version_info < (3, 10):
    raise RuntimeError(
        "\n"
        "NafsAI requires Python 3.10 or newer.\n"
        f"Your current version: {sys.version_info.major}.{sys.version_info.minor}\n"
        "Download the latest version from: https://python.org\n"
    )

from nafsai.memory import Memory
from nafsai.router import Router
from nafsai.normalizer import Normalizer
from nafsai.cache import Cache
from nafsai.session import Session
from nafsai.agent import Agent

__version__ = "0.1.2"
__author__  = "NafsAI"

__all__ = [
    "Memory",
    "Router",
    "Normalizer",
    "Cache",
    "Session",
    "Agent",
]
