# Primary Agronomic Sources Manifest

This directory (`docs/sources/`) contains official regulatory and technical publications downloaded directly from Government of India and ICAR portals. These files serve as the ground-truth offline registry for all `WEB_VERIFIED` templates in `edge/rules_engine.py`.

---

## Stored Primary Documents

### 1. CIB&RC Major Uses of Pesticides (Fungicides)
- **Local File Path**: `docs/sources/cibrc_fungicides_2026.pdf`
- **File Size**: 1,413,647 bytes
- **SHA256**: `2662c8f376bfa831743b105c1386185357b86230e0646f7a09bc00b652b5809f`
- **Fetch Date**: 2026-09-16
- **Origin URL**: `https://ppqs.gov.in/sites/default/files/2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf`
- **Document Title**: *MAJOR USES OF PESTICIDES (Registered under the Insecticides Act, 1968) (UPTO - 31/03/2026) — FUNGICIDES*
- **Issuing Body**: Central Insecticide Board & Registration Committee (CIB&RC), Directorate of Plant Protection, Quarantine & Storage (DPPQS), Ministry of Agriculture & Farmers Welfare, Faridabad, Haryana.
- **Templates Verified**:
  - `ACT_TREAT_RICE_BLIGHT`: Streptocycline 100–150 ppm (p. 29) + Copper Hydroxide (p. 8)
  - `ACT_TREAT_RICE_BLAST`: Tricyclazole 75% WP @ 300–400 g/ha (p. 40) / Isoprothiolane 40% EC @ 750 ml/ha (p. 14)
  - `ACT_TREAT_WHEAT_YELLOW_RUST`: Propiconazole 25% EC @ 500 g/ha (p. 24) / Tebuconazole 25% WG @ 750 g/ha (p. 37)
  - `ACT_TREAT_WHEAT_BROWN_RUST`: Propiconazole 25% EC @ 500 g/ha (p. 24) / Mancozeb 75% WP @ 1.5–2.0 kg/ha (p. 18)
  - `ACT_TREAT_WHEAT_POWDERY_MILDEW`: Sulphur 80% WG @ 2.5 kg/ha (p. 35) / Triadimefon 25% WP @ 260–520 g/ha (p. 40)

### 2. CIB&RC Major Uses of Pesticides (Insecticides)
- **Local File Path**: `docs/sources/cibrc_insecticides_2026.pdf`
- **File Size**: 1,760,349 bytes
- **SHA256**: `20fcd282d572e953815b56e8a2b83027da667ec844cf7bdfabb043daec315838`
- **Fetch Date**: 2026-09-16
- **Origin URL**: `https://ppqs.gov.in/sites/default/files/updated_mup_insecticide_as_on_31.03.2026_c.pdf`
- **Document Title**: *MAJOR USES OF PESTICIDES (Registered under the Insecticides Act, 1968) (UPTO–31.03.2026) — INSECTICIDES*
- **Issuing Body**: Central Insecticide Board & Registration Committee (CIB&RC), Directorate of Plant Protection, Quarantine & Storage (DPPQS), Ministry of Agriculture & Farmers Welfare, Faridabad, Haryana.
- **Templates Verified**:
  - `ACT_TREAT_RICE_TUNGRO`: Thiamethoxam 25% WG @ 100 g/ha (p. 54)
  - `ACT_TREAT_RICE_STEM_BORER`: Chlorantraniliprole 0.4% GR @ 10 kg/ha (p. 13) / Cartap Hydrochloride 50% SP @ 1000 g/ha (p. 12)
  - `ACT_TREAT_RICE_LEAF_ROLLER`: Flubendiamide 39.35% SC @ 50 ml/ha = 20 ml/acre (p. 33) / Chlorantraniliprole 18.5% SC @ 150 ml/ha (p. 12)
  - `ACT_TREAT_RICE_HISPA`: Chlorpyrifos 20% EC @ 1250 ml/ha (p. 17) / Quinalphos 25% EC @ 2000 ml/ha (p. 46)

### 3. DPPQS Components of Integrated Pest Management
- **Local File Path**: `docs/sources/dppqs_components_ipm.md`
- **File Size**: 139,295 bytes
- **SHA256**: `4b12ac58ea0c434c1396ed8d08e63c86177f1e969aca88f19438abc59b6a165a`
- **Fetch Date**: 2026-09-16
- **Origin URL**: `https://ppqs.gov.in/divisions/integrated-pest-management/components-ipm`
- **Document Title**: *Components of IPM — Directorate of Plant Protection, Quarantine & Storage*
- **Templates Verified**:
  - `ACT_MAINTAIN_ROUTINE`: Cultural practices and economic thresholding; chemical control strictly as last resort.

### 4. DPPQS & NIPHM AESA based IPM Package for Sugarcane
- **Local File Path**: `docs/sources/dppqs_ipm_sugarcane.pdf`
- **File Size**: 2,300,020 bytes
- **SHA256**: `0c1bfe634185444da43ff217c9910eed6dd1a5a356437f81cd16c0668a672609`
- **Fetch Date**: 2026-09-16
- **Origin URL**: `https://ppqs.gov.in/sites/default/files/sugarcane.pdf`
- **Document Title**: *AESA based Integrated Pest Management Package for Sugarcane*
- **Issuing Body**: National Institute of Plant Health Management (NIPHM) working group & Directorate of Plant Protection, Quarantine & Storage (DPPQS), Ministry of Agriculture, Govt. of India.
- **Templates Verified**:
  - `ACT_TREAT_SUGARCANE_VIRAL_ABIOTIC`: p. 29 (cultural roguing and burning of infected clumps along with root system; no ratooning; yellow sticky traps for aphid vectors; zero chemical sprays recommended for viral mosaic/grassy shoot) & p. 32 (nutrient deficiency chlorosis).

### 5. DPPQS & NCIPM Integrated Pest Management Package for Rice
- **Local File Path**: `docs/sources/dppqs_ipm_rice.pdf`
- **File Size**: 1,591,897 bytes
- **SHA256**: `9d63c8d64e67ca08fc35a804df522f812e28ef479ebf9241eea42c20e6d7486f`
- **Fetch Date**: 2026-09-16
- **Origin URL**: `https://ppqs.gov.in/sites/default/files/rice.pdf`
- **Document Title**: *Integrated Pest Management Package for Rice*
- **Issuing Body**: National Centre for Integrated Pest Management (NCIPM / ICAR-NCIPM) & Directorate of Plant Protection, Quarantine & Storage (DPPQS), Faridabad.
- **Status / Findings**: Contains Rice Brown Spot ETL (`2-3 spots/leaf & 2-3 infected plants/m2`, p. 18) and seed treatment (`carbendazim 50% WP @ 2 g/kg seed`, p. 19). Does NOT contain a registered foliar chemical spray for brown spot (confirming `ACT_TREAT_RICE_BROWN_SPOT` foliar spray must remain `RECALLED_UNVERIFIED`).

### 6. FAO Irrigation and Drainage Paper No. 56 — Chapter 5 (Crop Growth Stages)
- **Local File Path**: `docs/sources/fao56_chapter5_growth_stages.md`
- **File Size**: 30,517 bytes
- **SHA256**: `99727831e039779f61ae072dd5a063962c6217b666135df0ab69fa21ae66f2e4`
- **Fetch Date**: 2026-09-16
- **Origin URL**: `https://www.fao.org/3/x0490e/x0490e0a.htm`
- **Document Title**: *Chapter 5 - Introduction to crop evapotranspiration (ETc), Section 'Crop growth stages'*
- **Issuing Body**: Food and Agriculture Organization of the United Nations (FAO), Rome, Italy.
- **Standards Verified**:
  - Initial Stage: Planting/emergence to ~10% ground cover ($f_c \le 0.10$).
  - Crop Development Stage: 10% ground cover to effective full cover ($0.10 < f_c < 0.70\text{--}0.80$).
  - Mid-Season Stage: Effective full cover ($f_c \ge 0.70$) to start of maturity/senescence.
  - Late Season Stage: Start of maturity to harvest or full senescence.

### 7. FAO Irrigation and Drainage Paper No. 56 — Chapter 6 (Table 11 Stage Lengths & Table 12 Kc)
- **Local File Path**: `docs/sources/fao56_chapter6_stage_lengths_table11.md`
- **File Size**: 92,148 bytes
- **SHA256**: `74139f66f53538954491b5e9aed03080b3cb391fbb09f0e5b21e733b242ff6a6`
- **Fetch Date**: 2026-09-16
- **Origin URL**: `https://www.fao.org/3/x0490e/x0490e0b.htm`
- **Document Title**: *Chapter 6 - ETc - Single crop coefficient (Kc), Table 11 & Table 12*
- **Issuing Body**: Food and Agriculture Organization of the United Nations (FAO), Rome, Italy.
- **Standards Verified**:
  - Table 11 Stage Durations in Days:
    - Rice (Tropics): Initial: 30d, Dev: 30d, Mid: 60d, Late: 30d (Total 150d)
    - Wheat (Central India): Initial: 15d, Dev: 25d, Mid: 50d, Late: 30d (Total 120d)
    - Sugarcane (Ratoon, Low Latitudes): Initial: 25d, Dev: 70d, Mid: 135d, Late: 50d (Total 280d); Virgin: 35/60/190/120 (Total 405d)
  - Table 12 Crop Coefficients:
    - Rice: $K_{c,\text{ini}} = 1.05$, $K_{c,\text{mid}} = 1.20$, $K_{c,\text{end}} = 0.90\text{--}0.60$
    - Wheat: $K_{c,\text{ini}} = 0.30\text{--}0.70$, $K_{c,\text{mid}} = 1.15$, $K_{c,\text{end}} = 0.25\text{--}0.40$
    - Sugarcane: $K_{c,\text{ini}} = 0.40$, $K_{c,\text{mid}} = 1.25$, $K_{c,\text{end}} = 0.75$

