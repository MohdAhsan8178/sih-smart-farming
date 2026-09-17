"""
Test payload contract synchronization (J1.4).
Validates that docs/PAYLOAD_CONTRACT.md stays synchronized with code constants and schema outputs.
"""
from pathlib import Path
import pytest
from configs.classes import CLASS_NAMES
from configs.classes_model_b import CLASS_NAMES as MODEL_B_CLASSES
from configs.reliability import MODEL_A_CROSS_SOURCE_RELIABILITY
from edge.rules_engine import TEMPLATES
from core.trap_segmentation import TRAP_ETL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_payload_contract_contains_all_classes():
    contract_file = REPO_ROOT / "docs" / "PAYLOAD_CONTRACT.md"
    assert contract_file.exists()
    content = contract_file.read_text(encoding="utf-8")

    # 1. Model A classes
    for c in CLASS_NAMES:
        assert c in content, f"Class {c} missing from PAYLOAD_CONTRACT.md"

    # 2. Model B classes
    for mb in MODEL_B_CLASSES:
        assert mb in content, f"Model B class {mb} missing from PAYLOAD_CONTRACT.md"

    # 3. Action templates
    for tid in TEMPLATES.keys():
        assert tid in content, f"Template ID {tid} missing from PAYLOAD_CONTRACT.md"

    # 4. Reliability tiers
    for tier in {"TESTED_ROBUST", "TESTED_WEAK", "TESTED_FAILED", "UNTESTED"}:
        assert tier in content, f"Reliability tier {tier} missing from PAYLOAD_CONTRACT.md"

    # 5. Verification status frozen enum
    for status in {"VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED"}:
        assert status in content, f"Verification status {status} missing from PAYLOAD_CONTRACT.md"

    # 6. Established ETL comparison statuses
    for etl_s in {"BELOW_ETL", "AT_ETL", "ABOVE_ETL"}:
        assert etl_s in content, f"ETL status {etl_s} missing from PAYLOAD_CONTRACT.md"
