"""Unit tests for BGE path resolution (no torch/model load)."""

import os
import sys
from pathlib import Path

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def test_relative_resolves_to_repo_root_absolute():
    from rip_maf.core.config import Settings

    s = Settings(
        bge_m3_model_path="backend/models/bge-m3",
        bge_reranker_v2_m3="backend/models/reranker/bge_reranker_v2_m3",
    )
    repo_root = Path(BACKEND_DIR)
    assert Path(s.bge_m3_model_path).is_absolute()
    assert Path(s.bge_reranker_v2_m3).is_absolute()
    assert Path(s.bge_m3_model_path) == (repo_root / "backend/models/bge-m3").resolve()
    assert Path(s.bge_reranker_v2_m3) == (
        repo_root / "backend/models/reranker/bge_reranker_v2_m3"
    ).resolve()


def test_absolute_unchanged():
    from rip_maf.core.config import Settings

    s = Settings(
        bge_m3_model_path="/app/backend/models/bge-m3",
        bge_reranker_v2_m3="D:/models/reranker",
    )
    assert s.bge_m3_model_path == "/app/backend/models/bge-m3"
    # Windows normalizes D:/ -> D:\; compare via Path semantics.
    assert Path(s.bge_reranker_v2_m3) == Path("D:/models/reranker")


def test_legacy_alias_honored():
    from rip_maf.core.config import Settings

    s = Settings(_env_prefix="", _case_sensitive=False)
    # Aliases resolve through the same validator when set via env-style init.
    s2 = Settings.model_validate(
        {
            "bge_model_dir": "backend/models/bge-m3",
            "reranker_model_dir": "/tmp/reranker",
        }
    )
    assert Path(s2.bge_m3_model_path).is_absolute()
    assert s2.bge_reranker_v2_m3 == "/tmp/reranker"
    assert isinstance(s.bge_m3_model_path, str)
