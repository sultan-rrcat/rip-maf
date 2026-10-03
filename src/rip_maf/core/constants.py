"""Code-level constants shared across the backend.

Rule of thumb: if a value could change per-deployment, it belongs in Settings
(app/core/config.py). If it is an invariant fact about the code/library
(role names, SDK action strings, provider name prefixes), it belongs HERE.
"""
from __future__ import annotations

# --- Gemini provider facts (google.genai SDK) ---
GEMINI_ACTION_GENERATE_CONTENT = "generateContent"   # model action marking chat-capable
GEMINI_MODEL_ID_PREFIX = "models/"                   # models.list() names are "models/<id>"

# --- Provider message-role mapping (OpenAI-style list -> Gemini roles) ---
ROLE_USER = "user"
ROLE_MODEL = "model"

# --- Agent delegation confidence values ---
CONFIDENCE_SUCCESS = 0.95   # high-confidence agent success
CONFIDENCE_LOW = 0.0        # failure / missing input
