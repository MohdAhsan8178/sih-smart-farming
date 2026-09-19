#!/usr/bin/env python3
"""
Generate and synchronize docs/PAYLOAD_CONTRACT.md and docs/APP_TEAM_CHANGES.md from authoritative code constants.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.classes import CLASS_NAMES
from configs.classes_model_b import CLASS_NAMES as MODEL_B_CLASSES
from edge.rules_engine import TEMPLATES


def generate_payload_contract() -> str:
    template_ids = sorted(list(TEMPLATES.keys()))
    class_names = list(CLASS_NAMES)
    model_b_classes = list(MODEL_B_CLASSES) + ["UNCERTAIN_NON_TARGET"]

    contract_path = REPO_ROOT / "docs" / "PAYLOAD_CONTRACT.md"
    app_changes_path = REPO_ROOT / "docs" / "APP_TEAM_CHANGES.md"
    assert contract_path.exists(), f"Contract file {contract_path} does not exist"
    assert app_changes_path.exists(), f"App changes file {app_changes_path} does not exist"

    content = contract_path.read_text(encoding="utf-8")
    app_content = app_changes_path.read_text(encoding="utf-8")

    # Verify that class_names, model_b_classes, and template_ids are present in content
    for c in class_names:
        assert c in content, f"Missing class {c} in contract"
    for mb in model_b_classes:
        assert mb in content, f"Missing Model B class {mb} in contract"
    for tid in template_ids:
        assert tid in content, f"Missing template ID {tid} in contract"

    # Verify key table fields are present in both documents
    fields = re.findall(r"\|\s*\`([a-zA-Z0-9_\.]+)\`\s*\|", content)
    assert len(fields) > 50, f"Expected > 50 fields, extracted {len(fields)}"

    for f in fields:
        assert f in app_content, f"Field {f} missing from APP_TEAM_CHANGES.md"

    return content


if __name__ == "__main__":
    generate_payload_contract()
    print("PAYLOAD_CONTRACT verified successfully against code constants.")
