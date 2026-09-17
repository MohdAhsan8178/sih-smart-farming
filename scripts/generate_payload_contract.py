#!/usr/bin/env python3
"""
Generate and synchronize docs/PAYLOAD_CONTRACT.md from authoritative code constants.
"""
from pathlib import Path
from configs.classes import CLASS_NAMES
from configs.classes_model_b import CLASS_NAMES as MODEL_B_CLASSES
from edge.rules_engine import TEMPLATES

REPO_ROOT = Path(__file__).resolve().parent.parent

def generate_payload_contract() -> str:
    template_ids = sorted(list(TEMPLATES.keys()))
    class_names = list(CLASS_NAMES)
    model_b_classes = list(MODEL_B_CLASSES) + ["UNCERTAIN_NON_TARGET"]

    contract_path = REPO_ROOT / "docs" / "PAYLOAD_CONTRACT.md"
    assert contract_path.exists(), f"Contract file {contract_path} does not exist"
    content = contract_path.read_text(encoding="utf-8")
    
    # Verify that class_names, model_b_classes, and template_ids are present in content
    for c in class_names:
        assert c in content, f"Missing class {c} in contract"
    for mb in model_b_classes:
        assert mb in content, f"Missing Model B class {mb} in contract"
    for tid in template_ids:
        assert tid in content, f"Missing template ID {tid} in contract"

    return content

if __name__ == "__main__":
    generate_payload_contract()
    print("PAYLOAD_CONTRACT verified successfully against code constants.")
