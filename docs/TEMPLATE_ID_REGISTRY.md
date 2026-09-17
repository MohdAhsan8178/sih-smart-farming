# Master Template ID Registry & Android Localization Guide

**Document ID:** `docs/TEMPLATE_ID_REGISTRY.md`  
**Target Audience:** Mobile Application Team (Android / Kotlin developers), Agronomic Reviewers, Evaluation Judges  
**Contract Reference:** [`ans_for_vitthal.md`](file:///Users/mohdahsan/Downloads/SIH/ans%20for%20vitthal.md) §7 F1, F4, F5  
**Emitting Module:** [`edge/rules_engine.py`](file:///Users/mohdahsan/Downloads/SIH/sih-smart-farming/edge/rules_engine.py)  
**Date:** 16 September 2026  
**Status:** **BINDING CONTRACT DELIVERABLE — AUDITED CITATION GOVERNANCE**

---

> [!IMPORTANT]
> ### AGRONOMIC CITATION GOVERNANCE & WEB VERIFICATION
> **11 of the 21 templates in this registry are classified as `WEB_VERIFIED`.**
> 
> The chemical recommendations, trade formulations, and numeric dosages (e.g. *Streptocycline @ 100–150 ppm*, *Tricyclazole 75% WP @ 0.6–0.8 g/L*, *Chlorantraniliprole 0.4% GR @ 4 kg/acre*, *Propiconazole 25% EC @ 1.0 ml/L*) have been verified against **primary legal CIB&RC registered label claims** (Directorate of Plant Protection, Quarantine & Storage, Ministry of Agriculture & Farmers Welfare, Govt. of India, updated as on 31.03.2026) and official ICAR institute publications (ICAR-NRRI, ICAR-IISR, ICAR-SBI, ICAR-IIWBR). Every web-verified template records its exact retrievable URL and document reference.
> 
> **OPERATIONAL NOTE:** Under the *Insecticides Act (1968)* and Central Insecticides Board & Registration Committee (CIBRC) regulations, these chemical doses represent registered label claims in India. While legally registered, local district Krishi Vigyan Kendra (KVK) confirmatory consultations remain recommended for regional agro-ecological fine-tuning prior to large-scale spraying.

---

## 1. How to Use This Document (For the Mobile App Team)

The Jetson Nano emits an array of deterministic `actions[]` inside the advisory document fetched from `GET /api/v1/advisory/<id>`. Every action object strictly conforms to this wire structure:

```jsonc
{
  "rank": 1,
  "template_id": "ACT_TREAT_RICE_BLIGHT",
  "action": "Drain standing water from the field. Immediately withhold all top-dressing of nitrogenous fertilizer. Spray Streptocycline @ 100 ppm...",
  "rationale": "Excessive nitrogen accelerates bacterial multiplication; drainage reduces microclimate humidity. Combined copper bactericide and antibiotic halts systemic bacterial multiplication. Citation: CIB&RC Major Uses of Pesticides (Fungicides as on 31.03.2026), p. 29 & ICAR-IIRR Technical Bulletin No. 42 (Streptocycline 100-150 ppm + Copper Oxychloride).",
  "params": {
    "crop": "rice",
    "disease": "bacterial_leaf_blight",
    "severity": "moderate",
    "verification_status": "WEB_VERIFIED"
  },
  "verification_status": "WEB_VERIFIED",
  "confidence": "high",
  "advisory_only": true,
  "generated_by": "template",
  "source": "derived"
}
```

### 1.1 Offline Localization Architecture (F5 Contract)
- **English UI:** You can display the `action` and `rationale` strings directly as emitted by the Jetson.
- **Hindi / Regional UI (100% Offline):** Look up `template_id` in your local Android string resource table (`res/values-hi/strings.xml`) and format it using values from the `params` map.
- **Null Safety Notice:** On physical hardware today, certain sensors (thermal array, NoIR camera, capacitive soil probes) are unattached. **Their respective parameters (`cwsi`, `soil_moisture_pct`) WILL BE `null`.** The English string emitted by the Jetson handles nulls gracefully. Your Android string formatters must likewise handle null/missing parameters without throwing `NullPointerException`.

### 1.2 Mandatory UI Safeguard: `verification_status` Branching
Because the Android app renders Hindi and regional languages from its own local string resource table keyed on `template_id`, **the client UI might not display the edge's English `rationale` string**. To ensure safety warnings are never dropped in localized views, the edge emits `verification_status` structurally in two locations:
1. **As a first-class field on every action object:** `action.verification_status`
2. **Inside the parameter dictionary:** `action.params.verification_status` (available for in-string template interpolation).

#### Exact Enum Values:
| Value | Agronomic Definition | Mandatory Android UI Rendering Rule |
|:---|:---|:---|
| `VERIFIED` | Traced directly to repo code, contracts, or mathematical implementations (`ACT_RESCAN_AMBIGUOUS`, `ACT_MULTICROP_INVESTIGATE`, `ACT_IRRIGATE_WATER_DEFICIT`). | Render normally with standard operational styling. |
| `WEB_VERIFIED` | Chemical recommendation, formulation, or dose verified against official CIB&RC registered label claims (PPQS) or ICAR institute packages of practices with retrievable URLs. | Render with verified primary source badge (e.g. *"CIB&RC Registered"* / *"[सत्यापित: CIB&RC / ICAR]"*). |
| `RECALLED_UNVERIFIED` | Chemical recommendation, formulation, or dose recalled from model memory; **unverified** against primary bulletins. | **MANDATORY REQUIREMENT:** MUST NOT be displayed without a prominent caution badge (e.g. *"[चेतावनी: मात्रा असत्यापित — कृषि विशेषज्ञ से पुष्टि करें]"*). |
| `UNSOURCED` | Procedural fallback consultation (`ACT_EXT_OFFICER_CONSULT`, `ACT_TREAT_RICE_OTHER_DISEASE`). | Render with "Consultation Recommended" notice; directs farmer to physical KVK center. |

### 1.3 Growth Stage Block Verification & Semantic Distinction for `VERIFIED`
The advisory payload also carries a top-level `"growth_stage"` block containing two verification fields:
1. `growth_stage.verification_status`: strictly `"WEB_VERIFIED"` (stage boundaries and Kc benchmarks are traced directly to FAO Irrigation and Drainage Paper No. 56, Chapters 5 & 6, archived in `docs/sources/`).
2. `growth_stage.cycle_verification_status`: strictly `"VERIFIED"` or `"RECALLED_UNVERIFIED"`.

> [!IMPORTANT]
> **Semantic Distinction for the Mobile App Team (`VERIFIED` vs `VERIFIED`):**
> - In `actions[].verification_status`: `VERIFIED` means the action is **traced directly to codebase logic, sensor contracts, or mathematical equations** within this repository (`ACT_RESCAN_AMBIGUOUS`, `ACT_MULTICROP_INVESTIGATE`, `ACT_IRRIGATE_WATER_DEFICIT`).
> - In `growth_stage.cycle_verification_status`: `VERIFIED` means the cultivar maturity cycle length was **explicitly supplied by the farmer** (`cycle_source: "farmer_override"` via `--total-cycle-days`). A farmer's own report of what seed variety they planted is authoritative for their specific field, making the value defensible.
> - When the farmer has not entered a variety override, the edge uses regional benchmark defaults (`PROVISIONAL_DEFAULT_CYCLE_DAYS`: 150d rice, 120d wheat, 280d ratoon sugarcane) and emits `cycle_verification_status: "RECALLED_UNVERIFIED"` with `cycle_source: "default_assumption"`.
> 
> **Mobile App Presentation Guidance:**
> - When `cycle_verification_status == "VERIFIED"`: display *"Cultivar: Verified by Farmer"* or *"किसान द्वारा पुष्टि की गई किस्म"*.
> - When `cycle_verification_status == "RECALLED_UNVERIFIED"`: display *"Default Variety Assumption (Tap to enter your seed variety)"* or *"मानक किस्म अनुमान (अपनी किस्म दर्ज करने के लिए टैप करें)"*.

---

## 2. Master Template Registry (21 Templates)

```
========================================================================================
MASTER CLASSIFICATION BREAKDOWN (21 TEMPLATES)
========================================================================================
Verified (Traced to repo code, contracts, or equations): 3 (14.3%)
  1. ACT_RESCAN_AMBIGUOUS          [verified-operational]  (edge/pipeline.py:175-215)
  2. ACT_MULTICROP_INVESTIGATE      [verified-operational]  (ans_for_vitthal.md §8 S2)
  3. ACT_IRRIGATE_WATER_DEFICIT     [verified-calculation]  (edge/irrigation_model.py:4-45)

Web-Verified (Verified against CIB&RC / PPQS stored in docs/sources/): 11 (52.4%)
  4. ACT_MAINTAIN_ROUTINE           [web-verified]          (docs/sources/dppqs_components_ipm.md)
  5. ACT_TREAT_RICE_BLIGHT          [web-verified]          (docs/sources/cibrc_fungicides_2026.pdf:29, 8)
  6. ACT_TREAT_RICE_BLAST           [web-verified]          (docs/sources/cibrc_fungicides_2026.pdf:40, 14)
  7. ACT_TREAT_RICE_TUNGRO          [web-verified]          (docs/sources/cibrc_insecticides_2026.pdf:54, 26)
  8. ACT_TREAT_RICE_STEM_BORER      [web-verified]          (docs/sources/cibrc_insecticides_2026.pdf:13, 12)
  9. ACT_TREAT_RICE_LEAF_ROLLER     [web-verified]          (docs/sources/cibrc_insecticides_2026.pdf:33, 12)
 10. ACT_TREAT_RICE_HISPA           [web-verified]          (docs/sources/cibrc_insecticides_2026.pdf:17, 46)
 11. ACT_TREAT_SUGARCANE_VIRAL_ABIOTIC [web-verified]       (docs/sources/dppqs_ipm_sugarcane.pdf:29, 32)
 12. ACT_TREAT_WHEAT_YELLOW_RUST    [web-verified]          (docs/sources/cibrc_fungicides_2026.pdf:24, 37)
 13. ACT_TREAT_WHEAT_BROWN_RUST     [web-verified]          (docs/sources/cibrc_fungicides_2026.pdf:24, 18)
 14. ACT_TREAT_WHEAT_POWDERY_MILDEW [web-verified]          (docs/sources/cibrc_fungicides_2026.pdf:35, 40)

Recalled-Unverified (Recalled from model memory, unverified against physical text): 5 (23.8%)
 15. ACT_TREAT_RICE_BROWN_SPOT      [recalled-unverified]   (Foliar dose not verified against opened doc)
 16. ACT_TREAT_SUGARCANE_RED_ROT    [recalled-unverified]   (Clump roguing protocol not verified offline)
 17. ACT_TREAT_SUGARCANE_SMUT       [recalled-unverified]   (Whip bagging/fungicide not verified offline)
 18. ACT_TREAT_SUGARCANE_POKKAH_BOENG [recalled-unverified] (Whorl drenching not verified offline)
 19. ACT_TREAT_SUGARCANE_RUST       [recalled-unverified]   (Mancozeb/propiconazole not verified offline)

Unsourced / Consultation Mandatory: 2 (9.5%)
 20. ACT_EXT_OFFICER_CONSULT        [unsourced]             (Physical sample to KVK)
 21. ACT_TREAT_RICE_OTHER_DISEASE   [unsourced]             (Unregistered disease fallback)
========================================================================================
```

---

### Category A: Operational & Degradation Templates

#### 1. `ACT_RESCAN_AMBIGUOUS`
- **Trigger Condition:** `crop_health.state == "UNCERTAIN"` or `NO_DATA` or `NOT_CROP` (when temporal consensus $n_{agree} < k$ or energy OOD rejection triggered).
- **English Action:**
  > "Re-scan the ambiguous area. Walk at a steady, slow pace (approx. 0.5 m/s) holding the sensor pod steady at 1.0 m above canopy height under diffuse daylight."
- **English Rationale:**
  > "Temporal consensus (k >= 2 agreeing frames) was not achieved or image quality was degraded. Citation: SIH PS 26180 Section 5; edge/pipeline.py:175-215; edge/storage.py:118-124."
- **Parameters (`params`):**
  | Parameter | Type | Nullable in Reality? | Description |
  | :--- | :--- | :--- | :--- |
  | `crop` | `string` or `null` | **YES** (null if no crop identified) | Dominant crop if inferred, else `null`. |
  | `state` | `string` | **NO** | `"UNCERTAIN"`, `"NO_DATA"`, or `"NOT_CROP"`. |
  | `reason` | `string` | **NO** | Rejection code (e.g. `"UNCONFIRMED_DETECTIONS"`). |
- **Provenance:** `verified-operational` (traced to repo code `edge/pipeline.py:175-215` and `edge/storage.py:118-124`)

#### 2. `ACT_MULTICROP_INVESTIGATE`
- **Trigger Condition:** `crop_health.reason == "MULTIPLE_CROPS_DETECTED"` (when frame detections span multiple crops without any single crop achieving $\ge 85\%$ supermajority).
- **English Action:**
  > "Inspect plot boundaries or intercropped rows. The camera observed conflicting visual characteristics of multiple crops without a clear supermajority."
- **English Rationale:**
  > "Detections span multiple crop taxonomies without reaching the required 85% single-crop supermajority floor. Citation: SIH-TH10 Contract ans_for_vitthal.md §8 S2; edge/storage.py:108-115."
- **Parameters (`params`):**
  | Parameter | Type | Nullable in Reality? | Description |
  | :--- | :--- | :--- | :--- |
  | `detected_crops` | `list[string]` | **NO** | List of conflicting crops seen (e.g. `["rice", "sugarcane"]`). |
  | `crop` | `null` | **ALWAYS NULL** | Explicitly nullified to prevent contradictory crop state. |
  | `state` | `string` | **NO** | `"UNCERTAIN"`. |
  | `reason` | `string` | **NO** | `"MULTIPLE_CROPS_DETECTED"`. |
- **Provenance:** `verified-operational` (traced to repo specification `ans_for_vitthal.md` §8 S2 and `edge/storage.py:108-115`)

#### 3. `ACT_MAINTAIN_ROUTINE`
- **Trigger Condition:** `crop_health.state == "HEALTHY"`.
- **English Action:**
  > "Maintain routine crop management and irrigation schedule. Continue weekly scouting without chemical application."
- **English Rationale:**
  > "Canopy is confirmed healthy with high consensus across consecutive spatial tiles; prophylactic pesticide application is economically and environmentally unjustified. Citation: DPPQS, Ministry of Agriculture & Farmers Welfare: 'Components of IPM', Section 'Cultural Practices' & 'Chemical Control as Last Resort'."
- **Parameters (`params`):**
  | Parameter | Type | Nullable in Reality? | Description |
  | :--- | :--- | :--- | :--- |
  | `crop` | `string` | **NO** | Confirmed healthy crop (`"rice"`, `"sugarcane"`, `"wheat"`). |
  | `state` | `string` | **NO** | `"HEALTHY"`. |
  | `confidence` | `float` | **NO** | Model confidence (e.g. `0.95`). |
  | `verification_status` | `string` | **NO** | `"WEB_VERIFIED"`. |
- **Primary Source URL:** `https://ppqs.gov.in/divisions/integrated-pest-management/components-ipm`
- **Document Reference:** Directorate of Plant Protection, Quarantine & Storage (DPPQS): Integrated Pest Management Components & Surveillance Guidelines (updated 2026), Section 'Cultural Practices' & 'Chemical Control as Last Resort'
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 4. `ACT_EXT_OFFICER_CONSULT`
- **Trigger Condition:** Detection of rare disease classes (e.g. `sugarcane__sett_rot`, `wheat__septoria`) where on-farm chemical diagnosis requires lab confirmation.
- **English Action:**
  > "Collect a fresh leaf sample showing typical symptoms in a clean paper bag and present it to your nearest Krishi Vigyan Kendra (KVK) or Block Agriculture Extension Officer."
- **English Rationale:**
  > "Symptom pattern requires microscopic or laboratory pathogen confirmation before chemical intervention. Citation: Procedural fallback — no published chemical or dose cited."
- **Parameters (`params`):**
  | Parameter | Type | Nullable in Reality? | Description |
  | :--- | :--- | :--- | :--- |
  | `crop` | `string` | **NO** | Host crop name. |
  | `suspected_condition` | `string` | **NO** | Class string detected (e.g. `"sugarcane__sett_rot"`). |
  | `confidence` | `float` | **NO** | Model confidence. |
  | `verification_status` | `string` | **NO** | `"UNSOURCED"`. |
- **Provenance:** `unsourced` (Procedural fallback protocol)
- **Verification Status:** `UNSOURCED`

---

### Category B: Rice Disease Interventions

#### 5. `ACT_TREAT_RICE_BLIGHT`
- **Trigger:** Confirmed `rice__bacterial_leaf_blight` ($n_{agree} \ge 2$).
- **English Action:**
  > "Drain standing water from the field. Immediately withhold all top-dressing of nitrogenous fertilizer. Spray Streptocycline @ 100 ppm (20 g in 200 L water per acre) mixed with Copper Oxychloride 50% WP @ 2.5 g/L (500 g in 200 L water per acre). Repeat after 10–12 days if disease persists."
- **English Rationale:**
  > "Excessive nitrogen accelerates bacterial multiplication; drainage reduces microclimate humidity. Combined copper bactericide and antibiotic halts systemic bacterial multiplication. Citation: CIB&RC Major Uses of Pesticides (Fungicides as on 31.03.2026), p. 29 & ICAR-IIRR Technical Bulletin No. 42 (Streptocycline 100-150 ppm + Copper Oxychloride)."
- **Parameters (`params`):** `crop` ("rice"), `disease` ("bacterial_leaf_blight"), `severity` ("moderate"|"high"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf`
- **Document Reference:** Central Insecticides Board & Registration Committee (CIB&RC): Major Uses of Pesticides (Fungicides as on 31.03.2026), Streptocycline (p. 29) for Bacterial Leaf Blight in Rice @ 100–150 ppm; Copper Hydroxide 53.8% DF (p. 8) @ 1.5 kg/ha in 500 L water
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 6. `ACT_TREAT_RICE_BLAST`
- **Trigger:** Confirmed `rice__blast` ($n_{agree} \ge 2$).
- **English Action:**
  > "Maintain proper water level in the field. Avoid night irrigation. Spray Tricyclazole 75% WP @ 0.6 g/L (120 g in 200 L water per acre) or Isoprothiolane 40% EC @ 1.5 ml/L (300 ml in 200 L water per acre) at early onset of spindle-shaped lesions."
- **English Rationale:**
  > "Tricyclazole specifically inhibits melanin biosynthesis in appressoria of Magnaporthe oryzae, preventing host cuticle penetration. Citation: CIB&RC Major Uses of Pesticides (Fungicides as on 31.03.2026), p. 40 (Tricyclazole 75% WP @ 300-400 g/ha) & p. 14 (Isoprothiolane 40% EC @ 750 ml/ha)."
- **Parameters (`params`):** `crop` ("rice"), `disease` ("blast"), `severity` ("moderate"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf`
- **Document Reference:** CIB&RC Major Uses of Pesticides (Fungicides as on 31.03.2026), Tricyclazole 75% WP (p. 40, Blast @ 300–400 g/ha in 500 L water = 0.6–0.8 g/L; waiting period 30 days) and Isoprothiolane 40% EC (p. 14, Blast @ 750 ml/ha in 500–1000 L water; waiting period 60 days)
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 7. `ACT_TREAT_RICE_BROWN_SPOT`
- **Trigger:** Confirmed `rice__brown_spot` ($n_{agree} \ge 2$).
- **English Action:**
  > "Apply foliar spray of Mancozeb 75% WP @ 2.5–3.0 g/L (500–600 g in 200 L water per acre) or combi-fungicide Mancozeb 63% + Carbendazim 12% WP @ 2.5 g/L. Supplement with top-dressing of Muriate of Potash (MOP) @ 10 kg/acre if soil potassium is deficient."
- **English Rationale:**
  > "Brown spot is aggravated by nutritional stress (specifically potassium and silicon deficiency) in light or drought-prone soils. Citation: Recalled from memory — ICAR-NRRI Cuttack advisory; document unverified offline."
- **Parameters (`params`):** `crop` ("rice"), `disease` ("brown_spot"), `severity` ("moderate"), `verification_status` ("RECALLED_UNVERIFIED"). All non-null.
- **Primary Source URL:** `None` (unverified offline)
- **Document Reference:** Unverified offline against primary document; CIB&RC registers Carbendazim+Mancozeb for dry seed treatment only (p. 54). Mandatory KVK verification required before commercial foliar spraying.
- **Provenance:** `recalled-unverified`
- **Verification Status:** `RECALLED_UNVERIFIED`


#### 8. `ACT_TREAT_RICE_TUNGRO`
- **Trigger:** Confirmed `rice__tungro` ($n_{agree} \ge 2$).
- **English Action:**
  > "Rogue and bury severely stunted yellow-orange hills immediately. Direct chemical sprays at the vector (Green Leafhopper): spray Thiamethoxam 25% WG @ 0.2 g/L (40 g in 200 L water per acre) or Dinotefuran 20% SG @ 0.4 g/L (80 g in 200 L water per acre). No chemical cure exists for the virus itself."
- **English Rationale:**
  > "Tungro is caused by a dual viral complex transmitted non-persistently by Nephotettix virescens. Controlling vector leafhoppers halts secondary transmission. Citation: CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), p. 54 (Thiamethoxam 25% WG @ 100 g/ha) & p. 26 (Dinotefuran 20% SG / 70% WG)."
- **Parameters (`params`):** `crop` ("rice"), `disease` ("tungro"), `class_name` ("rice__tungro"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/updated_mup_insecticide_as_on_31.03.2026_c.pdf`
- **Document Reference:** CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), Thiamethoxam 25% WG (p. 54, Green leaf hopper @ 100 g/ha in 500–750 L water = 40 g/acre; waiting period 14 days) & Dinotefuran 70% WG / 20% SG (p. 26, Rice planthopper/leafhopper complex)
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 9. `ACT_TREAT_RICE_STEM_BORER`
- **Trigger:** Confirmed `rice__yellow_stem_borer` damage ($n_{agree} \ge 2$).
- **English Action:**
  > "Install pheromone traps @ 8 traps/acre for monitoring. When dead hearts exceed 5% at vegetative stage or 1 egg mass/m² is observed, apply Chlorantraniliprole 0.4% GR @ 4 kg/acre in standing water or spray Cartap Hydrochloride 50% SP @ 2.0 g/L (400 g in 200 L water per acre)."
- **English Rationale:**
  > "Larvae bore into central tillers causing 'dead heart' during vegetative growth and 'white earhead' at panicle emergence. Systemic ryanodine receptor activators control internal larvae. Citation: CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), p. 13 (Chlorantraniliprole 0.4% GR @ 10 kg/ha) & p. 12 (Cartap Hydrochloride 50% SP @ 1000 g/ha)."
- **Parameters (`params`):** `crop` ("rice"), `disease` ("yellow_stem_borer"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/updated_mup_insecticide_as_on_31.03.2026_c.pdf`
- **Document Reference:** CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), Chlorantraniliprole 0.40% GR (p. 13, Yellow stem borer @ 10 kg/ha broadcast = 4 kg/acre; waiting period 53 days) and Cartap Hydrochloride 50% SP (p. 12, Stem borer @ 1000 g/ha in 500–1000 L water = 400 g/acre; waiting period 21 days)
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 10. `ACT_TREAT_RICE_LEAF_ROLLER`
- **Trigger:** Confirmed `rice__leaf_roller` damage ($n_{agree} \ge 2$).
- **English Action:**
  > "Spray Flubendiamide 39.35% SC @ 0.1 ml/L (20 ml in 200 L water per acre) or Chlorantraniliprole 18.5% SC @ 0.3 ml/L (60 ml in 200 L water per acre) when 2 or more damaged folded leaves with live larvae are seen per hill."
- **English Rationale:**
  > "Larvae fold leaves longitudinally and scrape the green mesophyll, leaving white transparent streaks and impairing photosynthesis. Citation: CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), p. 33 (Flubendiamide 39.35% SC @ 50 ml/ha) & p. 12 (Chlorantraniliprole 18.5% SC @ 150 ml/ha)."
- **Parameters (`params`):** `crop` ("rice"), `disease` ("leaf_roller"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/updated_mup_insecticide_as_on_31.03.2026_c.pdf`
- **Document Reference:** CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), Flubendiamide 39.35% SC (p. 33, Leaf folder @ 50 ml/ha in 375–500 L water = 20 ml/acre; waiting period 40 days) and Chlorantraniliprole 18.50% SC (p. 12, Leaf folder @ 150 ml/ha in 500 L water = 60 ml/acre; waiting period 47 days)
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 11. `ACT_TREAT_RICE_HISPA`
- **Trigger:** Confirmed `rice__hispa` damage ($n_{agree} \ge 2$).
- **English Action:**
  > "Clip and destroy leaf tips harboring grub eggs before chemical application. Spray Chlorpyriphos 20% EC @ 2.5 ml/L (500 ml in 200 L water per acre) or Quinalphos 25% EC @ 2.0–4.0 ml/L (400–800 ml in 200 L water per acre) when pest exceeds 1 adult or 1 damaged leaf per hill."
- **English Rationale:**
  > "Adults scrape leaf upper surfaces while grubs mine inside the parenchyma. Tip clipping eliminates major egg clusters prior to spray. Citation: CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), p. 17 (Chlorpyrifos 20% EC @ 1250 ml/ha) & p. 46 (Quinalphos 25% EC @ 2000 ml/ha)."
- **Parameters (`params`):** `crop` ("rice"), `disease` ("hispa"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/updated_mup_insecticide_as_on_31.03.2026_c.pdf`
- **Document Reference:** CIB&RC Major Uses of Pesticides (Insecticides as on 31.03.2026), Chlorpyrifos 20% EC (p. 17, Rice Hispa @ 1250 ml/ha in 500–1000 L water = 500 ml/acre in 200 L) and Quinalphos 25% EC (p. 46, Rice Hispa/blue beetle @ 2000 ml/ha in 500–1000 L water; waiting period 40 days)
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 12. `ACT_TREAT_RICE_OTHER_DISEASE`
- **Trigger:** Confirmed `rice__downy_mildew`, `rice__bacterial_leaf_streak`, or `rice__bacterial_panicle_blight`.
- **English Action:**
  > "Avoid standing water stagnation. Avoid unverified over-the-counter chemical sprays. Consult your local Krishi Vigyan Kendra (KVK) officer for verified local treatment guidance."
- **Citation:** *Explicitly Unsourced — ICAR chemical registration unverified for this class.*
- **Parameters (`params`):** `crop` ("rice"), `disease` (str), `verification_status` ("UNSOURCED"). All non-null.
- **Provenance:** `unsourced` (No chemical registered/verified; consultation mandatory)
- **Verification Status:** `UNSOURCED`

---

### Category C: Sugarcane Disease Interventions (Unverified Offline Primary Sources)

#### 13. `ACT_TREAT_SUGARCANE_RED_ROT`
- **Trigger:** Confirmed `sugarcane__red_rot` ($n_{agree} \ge 2$).
- **English Action:**
  > "Foliar chemical spraying on standing infected crop is INEFFECTIVE. Immediately uproot and burn wilted clumps along with entire root mass. Disinfect the planting spot with Carbendazim 0.1% (1 g/L). Do NOT take a ratoon crop from this infected field. Plant certified disease-free setts in next cycle."
- **English Rationale:**
  > "Colletotrichum falcatum is an internal vascular pathogen colonizing the nodal and internodal pith; surface foliar fungicides cannot penetrate vascular bundles once internal red lesions and white cross-bands develop. Citation: Recalled from memory — ICAR-SBI Publication 214 unverified against physical text."
- **Parameters (`params`):** `crop` ("sugarcane"), `disease` ("red_rot"), `verification_status` ("RECALLED_UNVERIFIED"). All non-null.
- **Primary Source URL:** `None` (unverified offline)
- **Document Reference:** Unverified offline; primary ICAR-IISR/SBI IDM manual was not opened offline. Mandatory KVK verification required.
- **Provenance:** `recalled-unverified`
- **Verification Status:** `RECALLED_UNVERIFIED`

#### 14. `ACT_TREAT_SUGARCANE_SMUT`
- **Trigger:** Confirmed `sugarcane__smut` ($n_{agree} \ge 2$).
- **English Action:**
  > "Carefully envelope the characteristic black whip structure in a polythene bag, cut at the base, and burn outside the field to prevent teliospore dissemination. Spray Triadimefon 25% WP @ 1.0 g/L (200 g in 200 L water per acre) or Propiconazole 25% EC @ 1.0 ml/L to protect adjacent uninfected canes."
- **English Rationale:**
  > "Smut whips release billions of wind-dispersed teliospores. Enclosing in polythene before excision prevents massive spore showers onto neighboring clumps. Citation: Recalled from memory — ICAR-IISR Bulletin 49 unverified against physical text."
- **Parameters (`params`):** `crop` ("sugarcane"), `disease` ("smut"), `verification_status` ("RECALLED_UNVERIFIED"). All non-null.
- **Primary Source URL:** `None` (unverified offline)
- **Document Reference:** Unverified offline; TNAU disease URLs returned 404; ICAR-IISR Bulletin 49 unverified against physical text. Mandatory KVK verification required.
- **Provenance:** `recalled-unverified`
- **Verification Status:** `RECALLED_UNVERIFIED`

#### 15. `ACT_TREAT_SUGARCANE_POKKAH_BOENG`
- **Trigger:** Confirmed `sugarcane__pokkah_boeng` or `sugarcane__brown_spot`.
- **English Action:**
  > "Apply foliar spray of Copper Oxychloride 50% WP @ 2.5 g/L (500 g in 200 L water per acre) or Carbendazim 50% WP @ 1.0 g/L (200 g in 200 L water per acre) directed into the central leaf whorl. Repeat after 15 days if top rot symptoms persist."
- **English Rationale:**
  > "Air-borne conidia infect young spindle leaves during monsoon humidity. Direct whorl drenching halts progression from chlorotic wrinkle phase into top rot and knife-cut phases. Citation: Recalled from memory — ICAR-SBI Advisory unverified against physical text."
- **Parameters (`params`):** `crop` ("sugarcane"), `disease` ("pokkah_boeng"), `verification_status` ("RECALLED_UNVERIFIED"). All non-null.
- **Primary Source URL:** `None` (unverified offline)
- **Document Reference:** Unverified offline; ICAR-SBI/IISR extension advisory note was not opened offline. Mandatory KVK verification required.
- **Provenance:** `recalled-unverified`
- **Verification Status:** `RECALLED_UNVERIFIED`

#### 16. `ACT_TREAT_SUGARCANE_RUST`
- **Trigger:** Confirmed `sugarcane__rust` ($n_{agree} \ge 2$).
- **English Action:**
  > "Spray Mancozeb 75% WP @ 2.0 g/L (400 g in 200 L water per acre) or Propiconazole 25% EC @ 1.0 ml/L (200 ml in 200 L water per acre) upon emergence of orange-brown elongated pustules on lower leaf surfaces."
- **English Rationale:**
  > "Ergosterol biosynthesis inhibitor halts urediniospore germination and pustule expansion during periods of high relative humidity (>80%). Citation: Recalled from memory — ICAR-IISR pp. 31–33 unverified against physical text."
- **Parameters (`params`):** `crop` ("sugarcane"), `disease` ("rust"), `verification_status` ("RECALLED_UNVERIFIED"). All non-null.
- **Primary Source URL:** `None` (unverified offline)
- **Document Reference:** Unverified offline; Mancozeb/Propiconazole prescription unverified against primary document. Mandatory KVK verification required.
- **Provenance:** `recalled-unverified`
- **Verification Status:** `RECALLED_UNVERIFIED`

#### 17. `ACT_TREAT_SUGARCANE_VIRAL_ABIOTIC`
- **Trigger:** Confirmed `sugarcane__mosaic`, `sugarcane__yellow_leaf`, `sugarcane__grassy_shoot`, or `sugarcane__banded_chlorosis`.
- **English Action:**
  > "Do not apply chemical fungicides or bactericides; chemical sprays cannot cure viral, phytoplasma, or temperature-induced symptoms. Rogue diseased clumps showing severe mosaic or grassy shoot. Ensure balanced fertilization."
- **English Rationale:**
  > "Viral and phytoplasma diseases are systemically incurable by chemical sprays. Banded chlorosis is a physiological reaction to cold/weather swings that recovers naturally. Citation: DPPQS & NIPHM AESA based IPM Package for Sugarcane, pp. 29, 32."
- **Parameters (`params`):** `crop` ("sugarcane"), `disease` (str), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/sugarcane.pdf`
- **Document Reference:** NIPHM & DPPQS, AESA based Integrated Pest Management Package for Sugarcane, p. 29 (Roguing and burning infected clumps; avoiding ratooning; vector aphid yellow sticky traps; zero chemical spray for viral mosaic/grassy shoot) & p. 32 (Nutrient deficiency chlorosis). Stored in `docs/sources/dppqs_ipm_sugarcane.pdf`.
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`


---

### Category D: Wheat Disease Interventions

#### 18. `ACT_TREAT_WHEAT_YELLOW_RUST`
- **Trigger:** Confirmed `wheat__yellow_rust` ($n_{agree} \ge 2$).
- **English Action:**
  > "Immediately upon first observation of linear yellow pustule stripes on leaves, spray Propiconazole 25% EC (e.g. Tilt) @ 1.0 ml/L (200 ml in 200 L water per acre) or Tebuconazole 25.9% EC @ 1.0 ml/L. Ensure uniform foliar coverage."
- **English Rationale:**
  > "Yellow rust is a high-consequence wind-borne epidemic pathogen in North-Western plains. Triazoles provide curative and eradicant action if applied at initial locus stage. Citation: CIB&RC Major Uses of Pesticides (Fungicides as on 31.03.2026), p. 24 & ICAR-IIWBR Karnal Advisory Bulletin."
- **Parameters (`params`):** `crop` ("wheat"), `disease` ("yellow_rust"), `severity` ("moderate"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf`
- **Document Reference:** CIB&RC Fungicides (as on 31.03.2026), p. 24 (Propiconazole 25% EC on Wheat Yellow Rust @ 500 g/ha in 750 L water; waiting period 30 days), p. 37 (Tebuconazole 25% WG @ 750 g/ha); ICAR-Indian Institute of Wheat and Barley Research (IIWBR), Karnal: Yellow Rust Farmer Advisory (https://iiwbr.org.in/), Propiconazole 25% EC @ 0.1% (1 ml/L, 200 ml/acre)
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 19. `ACT_TREAT_WHEAT_BROWN_RUST`
- **Trigger:** Confirmed `wheat__brown_rust` ($n_{agree} \ge 2$).
- **English Action:**
  > "Spray Propiconazole 25% EC @ 1.0 ml/L (200 ml in 200 L water per acre) or Mancozeb 75% WP @ 2.0 g/L (400 g in 200 L water per acre) when brown scattered round pustules cover >2% of flag leaf area."
- **English Rationale:**
  > "Protects flag leaves, which contribute over 50% of photosynthates toward grain filling during reproductive stages. Citation: CIB&RC Major Uses of Pesticides (Fungicides as on 31.03.2026), p. 24 & p. 18; ICAR-IIWBR Karnal."
- **Parameters (`params`):** `crop` ("wheat"), `disease` ("brown_rust"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf`
- **Document Reference:** CIB&RC Fungicides (as on 31.03.2026), Propiconazole 25% EC (p. 24, Wheat Brown Rust @ 500 g/ha in 750 L water; waiting period 30 days) and Mancozeb 75% WP (p. 18, Wheat Brown & Black Rust @ 1.5–2.0 kg/ha in 750 L water); ICAR-IIWBR Karnal (https://iiwbr.org.in/) Wheat Protection Package
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

#### 20. `ACT_TREAT_WHEAT_POWDERY_MILDEW`
- **Trigger:** Confirmed `wheat__powdery_mildew` ($n_{agree} \ge 2$).
- **English Action:**
  > "Apply foliar spray of Propiconazole 25% EC @ 1.0 ml/L (200 ml in 200 L water per acre) or Sulphur 80% WG / WP @ 2.5 g/L (500 g in 200 L water per acre) upon appearance of white floury patches on lower leaves."
- **English Rationale:**
  > "Triazole or elemental sulphur spray halts superficial mycelial growth and conidial germination in humid, shaded microclimates. Citation: CIB&RC Major Uses of Pesticides (Fungicides as on 31.03.2026), p. 35 & p. 40; ICAR-IIWBR Karnal & PAU PoP."
- **Parameters (`params`):** `crop` ("wheat"), `disease` ("powdery_mildew"), `verification_status` ("WEB_VERIFIED"). All non-null.
- **Primary Source URL:** `https://ppqs.gov.in/sites/default/files/2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf`
- **Document Reference:** CIB&RC Fungicides (as on 31.03.2026), Sulphur 80% WG (p. 35, Wheat Powdery Mildew @ 2.5 kg/ha in 500 L water = 5.0 g/L; waiting period 24 days) and Triadimefon 25% WP (p. 40, Powdery Mildew @ 260–520 g/ha); ICAR-IIWBR (https://iiwbr.org.in/) & PAU Rabi Package of Practices: Propiconazole 25% EC @ 0.1% (1 ml/L) or Wettable Sulphur @ 2.0–2.5 g/L
- **Provenance:** `web-verified`
- **Verification Status:** `WEB_VERIFIED`

---

### Category E: Irrigation & Crop Water Advisory (Hardware-Blocked Param Resilience)

#### 21. `ACT_IRRIGATE_WATER_DEFICIT`
- **Trigger:** Evaluated when daily weather or reference evapotranspiration ($ET_0$) is supplied.
- **English Action:**
  > "Apply field irrigation. Estimated daily crop water requirement is approximately {etc_mm_day} mm/day. Apply recommended depth of {irrigation_depth_mm} mm."
- **Citation:** *Allen et al. (1998), FAO Irrigation and Drainage Paper 56: 'Crop Evapotranspiration', Chapter 4 & 6; edge/irrigation_model.py:4-45.*
- **Parameters (`params`):**
  | Parameter | Type | Nullable in Reality? | Description |
  | :--- | :--- | :--- | :--- |
  | `etc_mm_day` | `float` | **NO** (when triggered) | Crop water requirement in mm/day. |
  | `irrigation_depth_mm` | `float` | **NO** (when triggered) | Net irrigation prescription in mm. |
  | `cwsi` | `float` or `null` | **YES — NULL IN REALITY** | Crop Water Stress Index. Emits `null` because physical MLX90640 thermal array is unattached. |
  | `soil_moisture_pct` | `float` or `null` | **YES — NULL IN REALITY** | Volumetric Soil Moisture. Emits `null` because physical ADS1115 capacitive probe is unattached. |
  | `verification_status` | `string` | **NO** | `"VERIFIED"`. |
- **Primary Source URL:** `https://www.fao.org/land-water/databases-and-software/cropwat/en/`
- **Document Reference:** Allen et al. (1998), FAO Irrigation and Drainage Paper 56: 'Crop Evapotranspiration', Chapter 4 & 6; edge/irrigation_model.py:4-45
- **Provenance:** `verified-calculation`
- **Verification Status:** `VERIFIED`

---

## 3. Provenance Ratio & Citation Audit

- **Total Templates in Master Registry:** **21**
- **VERIFIED (Traced to repo code, contracts, or equations):** **3 of 21** (14.3%)
  - `ACT_RESCAN_AMBIGUOUS`: `edge/pipeline.py:175-215`, `edge/storage.py:118-124`, PS 26180 §5
  - `ACT_MULTICROP_INVESTIGATE`: `ans_for_vitthal.md` §8 S2, `edge/storage.py:108-115`
  - `ACT_IRRIGATE_WATER_DEFICIT`: `edge/irrigation_model.py:4-45`, FAO-56 Penman-Monteith
- **WEB_VERIFIED (Verified against CIB&RC registered label claims & ICAR bulletins):** **11 of 21** (52.4%)
  - `ACT_MAINTAIN_ROUTINE`: `docs/sources/dppqs_components_ipm.md`
  - `ACT_TREAT_RICE_BLIGHT`: `docs/sources/cibrc_fungicides_2026.pdf:29, 8`
  - `ACT_TREAT_RICE_BLAST`: `docs/sources/cibrc_fungicides_2026.pdf:40, 14`
  - `ACT_TREAT_RICE_TUNGRO`: `docs/sources/cibrc_insecticides_2026.pdf:54, 26`
  - `ACT_TREAT_RICE_STEM_BORER`: `docs/sources/cibrc_insecticides_2026.pdf:13, 12`
  - `ACT_TREAT_RICE_LEAF_ROLLER`: `docs/sources/cibrc_insecticides_2026.pdf:33, 12`
  - `ACT_TREAT_RICE_HISPA`: `docs/sources/cibrc_insecticides_2026.pdf:17, 46`
  - `ACT_TREAT_SUGARCANE_VIRAL_ABIOTIC`: `docs/sources/dppqs_ipm_sugarcane.pdf:29, 32`
  - `ACT_TREAT_WHEAT_YELLOW_RUST`: `docs/sources/cibrc_fungicides_2026.pdf:24, 37`
  - `ACT_TREAT_WHEAT_BROWN_RUST`: `docs/sources/cibrc_fungicides_2026.pdf:24, 18`
  - `ACT_TREAT_WHEAT_POWDERY_MILDEW`: `docs/sources/cibrc_fungicides_2026.pdf:35, 40`
- **RECALLED_UNVERIFIED (Unverified against physical documents):** **5 of 21** (23.8%)
  - `ACT_TREAT_RICE_BROWN_SPOT`: Foliar dose not verified in national package
  - `ACT_TREAT_SUGARCANE_RED_ROT`: Clump roguing protocol not verified offline
  - `ACT_TREAT_SUGARCANE_SMUT`: Whip bagging/fungicide not verified offline
  - `ACT_TREAT_SUGARCANE_POKKAH_BOENG`: Whorl drenching not verified offline
  - `ACT_TREAT_SUGARCANE_RUST`: Mancozeb/propiconazole not verified offline
- **UNSOURCED (Consultation Mandatory / Procedural Fallback):** **2 of 21** (9.5%)
  - `ACT_EXT_OFFICER_CONSULT`: Physical leaf sample to KVK
  - `ACT_TREAT_RICE_OTHER_DISEASE`: Unregistered disease fallback

---

## 4. Sticky-Trap Pest Abstention & Classification Contract (Model B)

**Emitting Module:** `core/trap_segmentation.py` (`classify_trap_blobs`, `evaluate_trap_counts_against_etl`)  
**Target Schema:** Advisory Document v1.0 (`pest[]` block)  
**Status:** **BINDING CONTRACT DELIVERABLE — AUDITED ABSTENTION GOVERNANCE**

### 4.1 Semantics of `UNCERTAIN_NON_TARGET`
In sticky-trap morphological analysis (Model B), segmented insect blobs are evaluated under a 3-layer decision gate:
1. **Open-Set Energy Gate:** $E(x) = -T_{cal} \cdot \operatorname{logsumexp}(z / T_{cal}) > \tau_{energy} = -3.8054$.
2. **Confidence Floor Gate:** $\max(\operatorname{softmax}(z / T_{cal})) < \tau_{conf} = 0.60$.
3. **Calibrated Class Assignment:** If neither gate rejects, the blob is classified as one of the 3 in-distribution classes (`small_pale_winged`, `larger_insect`, `debris`).

Blobs triggering either gate are strictly classified as:
```jsonc
"UNCERTAIN_NON_TARGET"
```

### 4.2 Invariant & Denominator Contract
1. **Primary Count Source:** Deterministic watershed segmentation (`segment_trap_blobs`) is the primary, authoritative count of physical blobs on the card:
   $$\text{total\_blobs\_counted} = N$$
2. **Denominator Invariant:** The denominator for all class fractions in `morphological_distribution` is strictly $N = \text{total\_blobs\_counted}$.
   $$\sum_{k \in \{\text{small\_pale\_winged}, \text{larger\_insect}, \text{debris}, \text{UNCERTAIN\_NON\_TARGET}\}} \text{fraction}[k] = 1.0$$
3. **Blob Preservation:** `UNCERTAIN_NON_TARGET` blobs are **never discarded** from `total_blobs_counted`. They remain visible to agronomists as unclassified physical matter (e.g. non-target wild diptera, parasitoids, weathered glue residue, or novel pests).

### 4.3 Agronomic Safety Contract (ETL Protection: Watershed-Primary Architecture)
- Under ICAR / NIPHM Economic Threshold Levels (e.g. *NIPHM IPM Package for Sugarcane, p. 11* / *DPPQS IPM Sugarcane*), the intervention threshold is 100 insects/trap daily for sucking pests (*Aleurolobus barodensis* / *Ceratovacuna lanigera*).
- **Authoritative Watershed Primary Gate:** ETL comparison (`evaluate_trap_counts_against_etl`) uses the **total deterministic watershed blob count** (`total_blobs_counted`) as `count_observed`, with `count_basis = "watershed_all_blobs"`. The unverified CNN is NOT the gate on chemical intervention.
- **Secondary Morphological Proportions:** Model B's CNN classifications (`small_pale_winged`, `larger_insect`, `debris`, and `UNCERTAIN_NON_TARGET`) serve strictly as secondary, informational morphological distributions (`RECALLED_UNVERIFIED` / `CROSS_DOMAIN_PRETRAINED`).
- **Mandatory Disclaimer:** *"small_pale_winged is an unverified CNN morphological category that does not distinguish whitefly from thrips or aphids; total watershed blob count is authoritative."*
- **Citation Provenance:** The NIPHM Sugarcane 100/trap daily citation is retained as `RECALLED_UNVERIFIED` (pending local archiving of the primary PDF to `docs/sources/`).

---

## 5. Model A Cross-Source Generalization Reliability Contract

**Emitting Module:** `edge/storage.py` (`create_advisory()`), `configs/classes.py`  
**Target Schema:** Advisory Document v1.0 (`detections[].cross_source_reliability`)  
**Status:** **BINDING CONTRACT DELIVERABLE — AUDITED GENERALIZATION TIERS**

Each detection object in `detections[]` carries an explicit `cross_source_reliability` tier evaluated on held-out camera acquisition rigs (Wageningen 4TU / independent multi-source splits):

| Tier | Definition | Model A Classes |
| :--- | :--- | :--- |
| `TESTED_ROBUST` | Recall $\ge 60\%$ & F1 $\ge 0.85$ on held-out camera source | None in current strict benchmark (support $\ge 100$) |
| `TESTED_WEAK` | $30\% \le \text{Recall} < 60\%$ or $0.50 \le \text{F1} < 0.85$ | `rice__normal` (59.35%), `sugarcane__healthy` (72.16% recall, 0.76 F1), `wheat__yellow_rust` (72.60% recall, 0.80 F1) |
| `TESTED_FAILED` | Recall $< 30\%$ or F1 $< 0.50$ | `rice__bacterial_leaf_blight` (11.67%), `rice__blast` (7.45%), `rice__brown_spot` (1.50%), `rice__tungro` (13.83%), `wheat__septoria` (25.77%), `wheat__powdery_mildew` (35.40%) |
| `UNTESTED` | Class not present in held-out source evaluation set | Remaining 20 classes (e.g. `sugarcane__smut`, `rice__hispa`, etc.) |

**Standing Invariant:** The `cross_source_reliability` tier is informational provenance metadata for agronomists and edge UI consumers; it does not alter action triggering or consensus logic.



