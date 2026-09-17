"""Single source of truth for the Model B sticky-trap patch taxonomy.

Python 3.6 compatible.
"""
from typing import Dict, List

# The 3 strictly honest morphological categories
CLASS_NAMES: List[str] = [
    'small_pale_winged',  # Whitefly-like morphology (Trialeurodes, Bemisia, sugarcane whitefly adults)
    'larger_insect',      # Non-target insects (Macrolophus, Nesidiocoris, spittlebug bycatch, beetles, wild arthropods)
    'debris',             # Card adhesive imperfections, dust clumps, fibers, frass, glue bubbles, glare
]

NUM_CLASSES: int = len(CLASS_NAMES)
IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NAME_FROM_IDX: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

TARGET_CLASS_NAME: str = 'small_pale_winged'
TARGET_CLASS_IDX: int = IDX[TARGET_CLASS_NAME]

# Formal definitions of the two known unfixable data gaps
KNOWN_UNFIXABLE_GAPS: Dict[str, Dict[str, str]] = {
    "sugarcane_woolly_aphid_wax": {
        "taxon": "Ceratovacuna lanigera",
        "morphology": "White flocculent wax secretions and cottony filaments rather than discrete insect bodies",
        "status": "UNREPRESENTED_IN_PUBLIC_DATA",
        "expected_model_behavior": "Segmented as irregular blobs by watershed; likely classified as 'debris' or 'small_pale_winged'",
        "resolution_requirement": "Physical sticky-card captures from infested sugarcane fields with expert ground-truth annotation",
    },
    "soft_bodied_aphids_thrips": {
        "taxa": "Rhopalosiphum padi, Sitobion avenae, Anaphothrips obscurus",
        "morphology": "Soft-bodied, elongate, matte insects with cornicles or fringed wings",
        "status": "UNLABELLED_IN_PUBLIC_STICKY_TRAP_DATA",
        "expected_model_behavior": "Counted in total blob density and interval rate by watershed; unclassified or routed to 'larger_insect'/'debris' by CNN",
        "resolution_requirement": "Physical sticky-card captures from wheat/rice fields with species-level ground-truth annotation",
    },
}
