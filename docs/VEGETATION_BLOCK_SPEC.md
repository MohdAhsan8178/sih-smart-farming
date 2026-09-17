# Vegetation Block Wire Contract Specification

**Document:** `docs/VEGETATION_BLOCK_SPEC.md`  
**Target Audience:** Mobile Application Team (Android / iOS UI & Data Engineers)  
**Contract Baseline:** `data_flow_architecture.md §7.3` reconciled with `ans for vitthal.md §B6, §B7`  
**Emitting Module:** `edge/storage.py` (`create_advisory()`)  
**Status:** **FROZEN API DELIVERABLE**  

---

## 1. Overview & Agronomic Design

The Handheld Nano Pod computes **uncalibrated RGB vegetation indices** directly from nadir RGB camera imagery captured while walking crop furrows.

### 1.1 Why Relative Interpretation?
Because uncalibrated RGB ratios depend on ambient sunlight spectrum, solar angle, sensor gain, and white balance, **no absolute universal health thresholds exist** for RGB VARI, ExG, TGI, or DGCI. Attempting to assert an absolute threshold (e.g. `VARI > 0.40 = HEALTHY`) without an on-board calibrated reflectance target is ungrounded.

Instead, the system computes the **within-scan relative distribution** of each index across all accepted frames in that session. Confounds (lighting, camera gain, sun angle) cancel out, allowing the farmer to identify cells that are noticeably lagging or greener relative to their own field on that specific day.

### 1.2 Mandatory Presentation Rule
> [!IMPORTANT]
> **Client Rendering Caveat**:
> The app MUST render the following caveat whenever relative vegetation bands are shown:
> *"Compares parts of your field against each other. It cannot tell you whether the whole field is healthy."*
> Do not conceal this in a hidden tooltip.

---

## 2. JSON Wire Schema

In every advisory emitted by `GET /api/v1/advisory/<id>`, the root object contains `"vegetation"`:

```jsonc
"vegetation": {
  "interpretation_mode": "relative",
  
  "canopy_cover": {
    "mean": 0.7338,
    "p10": 0.3040,
    "p50": 0.9444,
    "p90": 0.9880,
    "min_fraction_threshold": 0.15,
    "status": "OK",
    "threshold_source": "PROVISIONAL — uncalibrated ExG threshold = 20",
    "threshold_confirmed": false,
    "source": "measured"
  },

  "vari": {
    "mean": 0.1347,
    "p10": 0.1097,
    "p50": 0.1341,
    "p90": 0.1600,
    "field_median": 0.1341,
    "band": "TYPICAL",
    "band_basis": "within_scan_percentile",
    "threshold_source": "PROVISIONAL — within-scan relative, no published absolute band exists for uncalibrated RGB VARI",
    "threshold_confirmed": false,
    "source": "measured"
  },

  "exg": {
    "mean": 95.66,
    "threshold_source": "PROVISIONAL — uncalibrated ExG threshold = 20",
    "threshold_confirmed": false,
    "source": "measured"
  },

  "tgi": {
    "mean": 53.02,
    "threshold_source": "PROVISIONAL — uncalibrated RGB TGI regression",
    "threshold_confirmed": false,
    "source": "measured"
  },

  "dgci": {
    "mean": 0.4329,
    "out_of_domain_fraction": 0.0405,
    "threshold_source": "PROVISIONAL — narrowed foliage domain [60, 120] deg",
    "threshold_confirmed": false,
    "source": "measured"
  },

  "ndvi": null,
  "ndvi_status": "PENDING_HARDWARE_FINALIZATION",
  "ndvi_reason": "Optical path and calib_matrix in progress. Field reserved; not estimated."
}
```

---

## 3. Field Definitions & Value Constraints

| JSON Path | Type | Allowed Values / Range | Description |
|---|---|---|---|
| `interpretation_mode` | `string` | `"relative"` | Mode of statistical interpretation |
| `canopy_cover.mean` | `float` | `0.0` to `1.0` | Scan-wide mean fractional green canopy coverage |
| `canopy_cover.p10` | `float` | `0.0` to `1.0` | 10th percentile of frame canopy fractions |
| `canopy_cover.p50` | `float` | `0.0` to `1.0` | Median frame canopy fraction |
| `canopy_cover.p90` | `float` | `0.0` to `1.0` | 90th percentile of frame canopy fractions |
| `canopy_cover.min_fraction_threshold` | `float` | `0.15` | Minimum canopy fraction floor |
| `canopy_cover.status` | `string` | `"OK"`, `"INSUFFICIENT_CANOPY"` | Whether scan passed canopy fraction floor |
| `vari.mean` | `float` or `null` | Typically `-0.5` to `+0.5` | Visible Atmospherically Resistant Index mean |
| `vari.band` | `string` or `null` | See Table 4 | Relative within-scan decile band |
| `vari.field_median` | `float` or `null` | Typically `-0.5` to `+0.5` | Equal to `p50` |
| `exg.mean` | `float` or `null` | `0.0` to `255.0` | Excess Green ($2G - R - B$) |
| `tgi.mean` | `float` or `null` | `0.0` to `255.0` | Triangular Greenness Index ($G - 0.39R - 0.61B$) |
| `dgci.mean` | `float` or `null` | `0.0` to `1.0` | Dark Green Colour Index |
| `dgci.out_of_domain_fraction` | `float` | `0.0` to `1.0` | Proportion of pixels outside $[60^\circ, 120^\circ]$ hue |
| `ndvi` | `null` | `null` | Hardware reserved; always `null` in current build |
| `ndvi_status` | `string` | `"PENDING_HARDWARE_FINALIZATION"` | Reason for absence of NDVI |
| `threshold_confirmed` | `boolean` | `false` | Always `false` on all uncalibrated heuristics |
| `threshold_source` | `string` | Starts with `"PROVISIONAL"` | Heuristic citation provenance |
| `source` | `string` | `"measured"` | Provenance marker |

---

## 4. Relative Banding Table for VARI

When `vari.mean` is non-null, `vari.band` is assigned based on the scan's internal distribution:

| Band Key | Condition | UI Display String (English) | UI Display String (Hindi Guidance) | Farmer-Facing Interpretation |
|---|---|---|---|---|
| `LOWER_TAIL` | Below $10\text{th}$ percentile | "Noticeably below field average" | "खेत के औसत से काफी कम" | Noticeably less green than the rest of the field. Worth walking over to inspect. |
| `BELOW_TYPICAL` | $10\text{th}$ to $25\text{th}$ percentile | "Slightly below field average" | "खेत के औसत से थोड़ा कम" | Slightly below typical field vigor. |
| `TYPICAL` | $25\text{th}$ to $75\text{th}$ percentile | "Typical for this field" | "खेत के सामान्य स्तर के अनुसार" | In line with the rest of the field. |
| `ABOVE_TYPICAL` | Above $75\text{th}$ percentile | "Above field average" | "खेत के औसत से अधिक हरा" | Among the greener, denser parts of the field. |

---

## 5. Null Handling & Provisional Degradation Modes

The app must gracefully handle `null` numeric values without crashing or rendering `0.0` or blank text:

### Case 1: Insufficient Canopy Cover (`mean_canopy < 0.15`)
When the entire scan contains insufficient vegetation (e.g. freshly sown soil, bare fallow, or non-crop pavement):
- `canopy_cover.status`: `"INSUFFICIENT_CANOPY"`
- `vari.mean`, `exg.mean`, `tgi.mean`, `dgci.mean` are all `null`
- `reason`: `"INSUFFICIENT_CANOPY_FRACTION"`
- **Client Behavior**: Render *"Canopy cover too low to measure vegetation indices (<15%)."* Do NOT display zero.

### Case 2: Excessive Shade / Non-Foliage in DGCI (`out_of_domain_fraction > 0.30`)
When over 30% of canopy pixels have hue outside $[60^\circ, 120^\circ]$ (e.g. heavy shadow or blue-shifted cuticles):
- `dgci.mean`: `null`
- `dgci.reason`: `"OUT_OF_DOMAIN_FRACTION_EXCEEDED"`
- `dgci.out_of_domain_fraction`: $> 0.30$
- **Client Behavior**: Render *"DGCI withheld due to severe shade or non-green reflectance."*

### Case 3: Reserved NDVI
- `ndvi`: `null`
- `ndvi_status`: `"PENDING_HARDWARE_FINALIZATION"`
- **Client Behavior**: Show disabled or *"Awaiting Dual-Band Optical Hardware"*.
