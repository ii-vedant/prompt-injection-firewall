"""Central configuration.

Why a config module? Every other file imports settings from here instead of
calling os.getenv() directly. One place to change, one place to audit, and
secrets never get hard-coded in source files.
"""
import os
from pathlib import Path

# python-dotenv is optional: if it's missing we just read real environment variables.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


class Settings:
    # --- LLM ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    # "mock" = deterministic, offline, no API key. "gemini" = real model.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")

    # --- Infra (used from Phase 2 onwards) ---
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/firewall"
    )


settings = Settings()
