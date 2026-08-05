"""
Structure presets, in nanometers.

These are illustrative of publicly known industry scaling trends (6F^2 DRAM
cell architecture, FinFET fin/gate pitch scaling) -- NOT exact proprietary
fab specifications. Override any field via CLI/config for other nodes.
"""

DRAM_1X = {
    "kind": "dram",
    "feature_size_nm": 18,          # F
    "word_line_pitch_nm": 36,       # ~2F
    "word_line_width_nm": 18,       # ~F
    "bit_line_pitch_nm": 54,        # ~3F  (6F^2 folded-bitline cell = 2F x 3F)
    "bit_line_width_nm": 18,        # ~F
    "contact_diameter_nm": 18,      # ~F, storage-node landing pad
}

FINFET_10NM = {
    "kind": "finfet",
    "fin_pitch_nm": 34,
    "fin_width_nm": 8,
    "gate_pitch_nm": 64,            # contacted poly pitch (CPP)
    "gate_length_nm": 18,
    "contact_size_nm": 20,
}

PRESETS = {
    "dram_1x": DRAM_1X,
    "finfet_10nm": FINFET_10NM,
}


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return dict(PRESETS[name])
