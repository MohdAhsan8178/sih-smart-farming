"""
tests/test_rules_engine.py — Unit tests for edge/rules_engine.py (Step 27).

Asserts:
  - Multi-crop degradation produces ACT_MULTICROP_INVESTIGATE with null crop.
  - Ambiguous/no-consensus state produces ACT_RESCAN_AMBIGUOUS.
  - Healthy confirmation produces ACT_MAINTAIN_ROUTINE.
  - Confirmed diseases emit cited treatment actions (rice blight, blast, sugarcane red rot, wheat rust).
  - Unsourced diseases honestly emit ACT_TREAT_RICE_OTHER_DISEASE or ACT_EXT_OFFICER_CONSULT.
  - Hardware-blocked params (cwsi, soil_moisture_pct) format gracefully when null.
  - Invariant: generated_by is ALWAYS "template" (never "placeholder" or "rules_engine").
  - Every master template definition carries a documented citation and provenance tag.
"""

from pathlib import Path
import pytest
from typing import Dict, Any, List

from edge.rules_engine import TEMPLATES, CLASS_TO_TEMPLATE, evaluate_rules


def test_every_template_has_verifiable_citation_and_provenance():
    """Asserts honest 3/16/0/2 provenance classification across all 21 templates."""
    valid_provenance_tags = {
        "verified-operational",
        "verified-calculation",
        "web-verified",
        "recalled-unverified",
        "unsourced",
    }
    counts = {
        "verified-operational": 0,
        "verified-calculation": 0,
        "web-verified": 0,
        "recalled-unverified": 0,
        "unsourced": 0,
    }
    for tmpl_id, defn in TEMPLATES.items():
        assert "template_id" in defn
        assert "action" in defn and len(defn["action"]) > 10
        assert "rationale" in defn and len(defn["rationale"]) > 10
        assert "citation" in defn and len(defn["citation"]) > 5, (
            f"Template {tmpl_id} lacks a citation!"
        )
        prov = defn.get("provenance")
        assert prov in valid_provenance_tags, (
            f"Template {tmpl_id} has invalid provenance: {prov}"
        )
        counts[prov] += 1

    # Exact honest breakdown
    assert counts["verified-operational"] == 2  # RESCAN_AMBIGUOUS, MULTICROP_INVESTIGATE
    assert counts["verified-calculation"] == 1  # IRRIGATE_WATER_DEFICIT (FAO-56 in edge/irrigation_model.py)
    assert counts["web-verified"] == 11         # 11 agronomic templates verified against CIB&RC / PPQS stored in docs/sources/
    assert counts["recalled-unverified"] == 5   # 5 unverified templates (4 sugarcane + 1 rice brown spot)
    assert counts["unsourced"] == 2             # EXT_OFFICER_CONSULT, TREAT_RICE_OTHER_DISEASE



def test_recalled_unverified_templates_emit_mandatory_warning_in_rationale():
    """Any template marked recalled-unverified must carry explicit warning in its emitted rationale."""
    for tmpl_id, defn in TEMPLATES.items():
        if defn.get("provenance") == "recalled-unverified":
            from edge.rules_engine import _build_action
            act = _build_action(defn, {})
            assert "RECALLED-UNVERIFIED" in act["rationale"], (
                f"Template {tmpl_id} is recalled-unverified but lacks warning in rationale!"
            )
            assert "requires human agronomist verification" in act["rationale"]


def test_multicrop_degradation_produces_multicrop_investigate():
    """When detections span multiple crops without supermajority, emit ACT_MULTICROP_INVESTIGATE."""
    actions = evaluate_rules(
        crop=None,
        state="UNCERTAIN",
        reason="MULTIPLE_CROPS_DETECTED",
        detections=[
            {"class": "rice__blast", "confidence": 0.95},
            {"class": "sugarcane__red_rot", "confidence": 0.94},
        ],
    )
    assert len(actions) == 1
    act = actions[0]
    assert act["template_id"] == "ACT_MULTICROP_INVESTIGATE"
    assert act["params"]["crop"] is None
    assert set(act["params"]["detected_crops"]) == {"rice", "sugarcane"}
    assert act["generated_by"] == "template"
    assert act["confidence"] == "low"
    assert "Citation:" in act["rationale"]


def test_unconfirmed_consensus_produces_rescan_ambiguous():
    """When state is UNCERTAIN, emit ACT_RESCAN_AMBIGUOUS with specific reason."""
    actions = evaluate_rules(
        crop="rice",
        state="UNCERTAIN",
        reason="UNCONFIRMED_DETECTIONS",
        detections=[{"class": "rice__blast", "confidence": 0.88}],
    )
    assert len(actions) == 1
    act = actions[0]
    assert act["template_id"] == "ACT_RESCAN_AMBIGUOUS"
    assert act["params"]["crop"] == "rice"
    assert act["params"]["reason"] == "UNCONFIRMED_DETECTIONS"
    assert act["generated_by"] == "template"
    assert act["confidence"] == "low"


def test_healthy_state_produces_maintain_routine():
    """When state is HEALTHY, emit ACT_MAINTAIN_ROUTINE advising no chemical spray."""
    actions = evaluate_rules(
        crop="rice",
        state="HEALTHY",
        detections=[{"class": "rice__normal", "confidence": 0.98}],
    )
    assert len(actions) == 1
    act = actions[0]
    assert act["template_id"] == "ACT_MAINTAIN_ROUTINE"
    assert act["params"]["crop"] == "rice"
    assert act["generated_by"] == "template"
    assert act["confidence"] == "high"


def test_confirmed_rice_blight_produces_icar_streptocycline_action():
    """Confirmed bacterial leaf blight emits ACT_TREAT_RICE_BLIGHT with ICAR-IIRR citation."""
    actions = evaluate_rules(
        crop="rice",
        state="DISEASE",
        detections=[
            {"class": "rice__bacterial_leaf_blight", "confidence": 0.96},
            {"class": "rice__bacterial_leaf_blight", "confidence": 0.94},
        ],
    )
    assert len(actions) >= 1
    act = actions[0]
    assert act["template_id"] == "ACT_TREAT_RICE_BLIGHT"
    assert act["params"]["crop"] == "rice"
    assert act["params"]["disease"] == "bacterial_leaf_blight"
    assert "Streptocycline" in act["action"]
    assert "Copper Oxychloride" in act["action"]
    assert "ICAR-IIRR" in act["rationale"]
    assert act["generated_by"] == "template"


def test_confirmed_sugarcane_red_rot_advises_roguing_and_rejects_foliar_spray():
    """Confirmed sugarcane red rot explicitly notes foliar chemical spraying is ineffective."""
    actions = evaluate_rules(
        crop="sugarcane",
        state="DISEASE",
        detections=[{"class": "sugarcane__red_rot", "confidence": 0.97}],
    )
    assert len(actions) >= 1
    act = actions[0]
    assert act["template_id"] == "ACT_TREAT_SUGARCANE_RED_ROT"
    assert "INEFFECTIVE" in act["action"]
    assert "ICAR-SBI" in act["rationale"]
    assert act["generated_by"] == "template"


def test_confirmed_wheat_yellow_rust_produces_propiconazole_action():
    """Confirmed yellow rust emits ACT_TREAT_WHEAT_YELLOW_RUST with ICAR-IIWBR citation."""
    actions = evaluate_rules(
        crop="wheat",
        state="DISEASE",
        detections=[{"class": "wheat__yellow_rust", "confidence": 0.98}],
    )
    assert len(actions) >= 1
    act = actions[0]
    assert act["template_id"] == "ACT_TREAT_WHEAT_YELLOW_RUST"
    assert "Propiconazole" in act["action"]
    assert "ICAR-IIWBR" in act["rationale"]
    assert act["generated_by"] == "template"


def test_unsourced_disease_honestly_emits_unsourced_consult():
    """Rice downy mildew emits ACT_TREAT_RICE_OTHER_DISEASE with explicit unsourced marker."""
    actions = evaluate_rules(
        crop="rice",
        state="DISEASE",
        detections=[{"class": "rice__downy_mildew", "confidence": 0.91}],
    )
    assert len(actions) >= 1
    act = actions[0]
    assert act["template_id"] == "ACT_TREAT_RICE_OTHER_DISEASE"
    assert "Explicitly Unsourced" in act["rationale"]
    assert act["generated_by"] == "template"


def test_secondary_irrigation_recommendation_handles_null_sensors_gracefully():
    """Secondary irrigation action renders correctly even when cwsi and soil_moisture_pct are null."""
    actions = evaluate_rules(
        crop="rice",
        state="HEALTHY",
        detections=[{"class": "rice__normal", "confidence": 0.98}],
        sensors={
            "etc_mm_day": 5.2,
            "cwsi": None,  # Hardware-blocked
            "soil_moisture_pct": None,  # Hardware-blocked
        },
    )
    assert len(actions) == 2
    irrig_act = actions[1]
    assert irrig_act["template_id"] == "ACT_IRRIGATE_WATER_DEFICIT"
    assert irrig_act["params"]["etc_mm_day"] == 5.2
    assert irrig_act["params"]["cwsi"] is None
    assert irrig_act["params"]["soil_moisture_pct"] is None
    assert "5.2" in irrig_act["action"]
    assert irrig_act["generated_by"] == "template"


def test_contract_standing_invariants_across_all_actions():
    """Standing guard: ALL actions emitted must conform strictly to ans_for_vitthal.md §7 F4 wire contract."""
    test_cases = [
        ("rice", "DISEASE", "rice__blast", 0.95),
        ("sugarcane", "DISEASE", "sugarcane__smut", 0.92),
        ("wheat", "DISEASE", "wheat__brown_rust", 0.89),
        ("rice", "HEALTHY", "rice__normal", 0.98),
        (None, "UNCERTAIN", "not_crop", 0.60),
    ]
    for crop, state, cname, conf in test_cases:
        acts = evaluate_rules(
            crop=crop,
            state=state,
            detections=[{"class": cname, "confidence": conf}],
        )
        for act in acts:
            assert act["generated_by"] == "template", "Violation: generated_by must always be 'template'"
            assert act["advisory_only"] is True
            assert act["source"] == "derived"
            assert isinstance(act["rank"], int) and act["rank"] >= 1
            assert isinstance(act["template_id"], str) and len(act["template_id"]) > 0
            assert isinstance(act["action"], str) and len(act["action"]) > 0
            assert isinstance(act["rationale"], str) and len(act["rationale"]) > 0
            assert isinstance(act["params"], dict)
            assert act["confidence"] in ("high", "medium", "low")
            # Structural verification_status invariants
            assert act["verification_status"] in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED")
            assert act["params"]["verification_status"] == act["verification_status"]


def test_standing_guard_chemical_doses_never_verified_without_repo_source():
    """
    Standing guard:
    1. Every template in TEMPLATES must declare verification_status in
       ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED").
    2. Any template emitting a chemical dose or formulation in its action string
       MUST NOT carry 'VERIFIED' (VERIFIED is strictly reserved for repo code/calculations).
    3. Any template marked 'WEB_VERIFIED' MUST carry a retrievable primary source URL
       (from an authoritative gov.in or official research domain) and a specific document reference.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent

    # Chemical indicator tokens: units of concentration, formulation types, or agrochemicals
    chem_indicators = [
        r"\bg/L\b", r"\bml/L\b", r"\bppm\b", r"\bWP\b", r"\bEC\b", r"\bSC\b", r"\bSG\b", r"\bGR\b", r"\bSP\b",
        r"\bStreptocycline\b", r"\bCopper Oxychloride\b", r"\bTricyclazole\b", r"\bIsoprothiolane\b",
        r"\bMancozeb\b", r"\bCarbendazim\b", r"\bThiamethoxam\b", r"\bDinotefuran\b",
        r"\bChlorantraniliprole\b", r"\bCartap\b", r"\bFlubendiamide\b", r"\bChlorpyriphos\b",
        r"\bQuinalphos\b", r"\bTriadimefon\b", r"\bPropiconazole\b", r"\bTebuconazole\b"
    ]
    pattern = re.compile("|".join(chem_indicators), re.IGNORECASE)

    for tmpl_id, defn in TEMPLATES.items():
        v_status = defn.get("verification_status")
        assert v_status in ("VERIFIED", "WEB_VERIFIED", "RECALLED_UNVERIFIED", "UNSOURCED"), (
            f"Template {tmpl_id} lacks a valid verification_status: {v_status}"
        )

        action_text = defn.get("action", "")
        has_chemical_dose = bool(pattern.search(action_text))

        if has_chemical_dose:
            # Must NOT be VERIFIED (VERIFIED is reserved for repo internal algorithms/code)
            assert v_status != "VERIFIED", (
                f"VIOLATION: Template {tmpl_id} prescribes chemical doses ('{action_text[:60]}...') "
                f"but is marked VERIFIED. Doses cannot be marked VERIFIED."
            )
            # Must be WEB_VERIFIED or RECALLED_UNVERIFIED
            assert v_status in ("WEB_VERIFIED", "RECALLED_UNVERIFIED")

        if v_status == "WEB_VERIFIED":
            # Must carry URL, document_reference, AND a local offline source file in docs/sources/
            assert "url" in defn and defn["url"] is not None, (
                f"Template {tmpl_id} is WEB_VERIFIED but missing 'url' key!"
            )
            assert defn["url"].startswith("http://") or defn["url"].startswith("https://"), (
                f"Template {tmpl_id} URL {defn['url']} is not a valid HTTP/HTTPS URL!"
            )
            assert any(domain in defn["url"] for domain in ["gov.in", "icar"]), (
                f"Template {tmpl_id} URL {defn['url']} is not from an approved primary source domain!"
            )
            assert "document_reference" in defn and len(defn["document_reference"]) > 10, (
                f"Template {tmpl_id} is WEB_VERIFIED but missing valid 'document_reference'!"
            )
            assert "offline_source_file" in defn and defn["offline_source_file"] is not None, (
                f"Template {tmpl_id} is WEB_VERIFIED but lacks 'offline_source_file'! "
                f"Every WEB_VERIFIED template must be backed by a fetched document stored in docs/sources/."
            )
            offline_path = repo_root / defn["offline_source_file"]
            assert offline_path.is_file(), (
                f"Template {tmpl_id} offline source file '{defn['offline_source_file']}' does not exist on disk!"
            )


def test_template_provenance_counts():
    """Assert exact counts: 3 VERIFIED, 11 WEB_VERIFIED, 5 RECALLED_UNVERIFIED, 2 UNSOURCED."""
    counts = {"VERIFIED": 0, "WEB_VERIFIED": 0, "RECALLED_UNVERIFIED": 0, "UNSOURCED": 0}
    for defn in TEMPLATES.values():
        st = defn.get("verification_status")
        counts[st] = counts.get(st, 0) + 1
    assert counts["VERIFIED"] == 3
    assert counts["WEB_VERIFIED"] == 11, f"Expected 11 WEB_VERIFIED, got {counts['WEB_VERIFIED']}"
    assert counts["RECALLED_UNVERIFIED"] == 5, f"Expected 5 RECALLED_UNVERIFIED, got {counts['RECALLED_UNVERIFIED']}"
    assert counts["UNSOURCED"] == 2


def test_web_verified_actions_emitted_correctly():
    """Validates that all 11 WEB_VERIFIED templates emit clean rationale, valid offline source, and correct status."""
    web_verified_count = 0
    for tmpl_id, defn in TEMPLATES.items():
        if defn.get("verification_status") == "WEB_VERIFIED":
            web_verified_count += 1
            from edge.rules_engine import _build_action
            act = _build_action(defn, {})
            assert act["verification_status"] == "WEB_VERIFIED"
            assert act["params"]["verification_status"] == "WEB_VERIFIED"
            assert "RECALLED-UNVERIFIED" not in act["rationale"]
            assert "Citation:" in act["rationale"]
            assert act["offline_source_file"] is not None
    assert web_verified_count == 11


def test_all_template_ids_valid_format_and_no_typo_collisions():
    """
    Guard:
    1. Every template_id in TEMPLATES matches ^ACT_[A-Z_]+$.
    2. Every template defn['template_id'] equals its dictionary key.
    3. Every mapped template in CLASS_TO_TEMPLATE exists in TEMPLATES.
    4. No two template IDs differ only by typo-distance (Levenshtein distance <= 2).
    """
    import re
    id_pattern = re.compile(r"^ACT_[A-Z_]+$")

    tids = list(TEMPLATES.keys())
    assert len(tids) == 21, "Expected exactly 21 templates in TEMPLATES"

    for tid in tids:
        assert id_pattern.match(tid), "Template ID %s does not match ^ACT_[A-Z_]+$" % tid
        defn = TEMPLATES[tid]
        assert defn.get("template_id") == tid, "Key %s mismatch with defn['template_id'] %s" % (
            tid, defn.get("template_id")
        )

    for cname, tid in CLASS_TO_TEMPLATE.items():
        assert tid in TEMPLATES, "CLASS_TO_TEMPLATE maps '%s' to missing template '%s'" % (cname, tid)

    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev[j + 1] + 1
                deletions = curr[j] + 1
                substitutions = prev[j] + (c1 != c2)
                curr.append(min(insertions, deletions, substitutions))
            prev = curr
        return prev[-1]

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            dist = levenshtein(tids[i], tids[j])
            assert dist > 2, (
                "Typo collision detected between '%s' and '%s' (edit distance %d <= 2)"
                % (tids[i], tids[j], dist)
            )


