#!/usr/bin/env python3
"""Extracts Listing A and Listing B from docs/HARDWARE_WIRING_GUIDE.md verbatim."""
from pathlib import Path

guide_path = Path("docs/HARDWARE_WIRING_GUIDE.md")
lines = guide_path.read_text(encoding="utf-8").splitlines()

a_start = None
a_end = None
for i, line in enumerate(lines):
    if "### Listing A:" in line:
        for j in range(i + 1, len(lines)):
            if lines[j].startswith("```"):
                a_start = j + 1
                break
    if a_start is not None and i > a_start and line == "```":
        a_end = i
        break

b_start = None
b_end = None
for i, line in enumerate(lines):
    if "### Listing B:" in line:
        for j in range(i + 1, len(lines)):
            if lines[j].startswith("```"):
                b_start = j + 1
                break
    if b_start is not None and i > b_start and line == "```":
        b_end = i
        break

print("Listing A:", a_start, "to", a_end, "First:", lines[a_start], "Last:", lines[a_end - 1])
print("Listing B:", b_start, "to", b_end, "First:", lines[b_start], "Last:", lines[b_end - 1])

listing_a = lines[a_start:a_end]
listing_b = lines[b_start:b_end]

Path("firmware/node_n01").mkdir(parents=True, exist_ok=True)
Path("firmware/esp32cam_trap").mkdir(parents=True, exist_ok=True)

Path("firmware/node_n01/node_n01.ino").write_text("\n".join(listing_a) + "\n", encoding="utf-8")
Path("firmware/esp32cam_trap/esp32cam_trap.ino").write_text("\n".join(listing_b) + "\n", encoding="utf-8")

print("Successfully wrote firmware/node_n01/node_n01.ino (", len(listing_a), "lines )")
print("Successfully wrote firmware/esp32cam_trap/esp32cam_trap.ino (", len(listing_b), "lines )")
