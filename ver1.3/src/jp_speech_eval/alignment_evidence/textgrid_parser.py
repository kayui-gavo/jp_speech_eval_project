from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .base import PhoneSegment


def parse_textgrid_phone_segments(path: str | Path, *, tier_names: tuple[str, ...] = ("phones", "phone", "segments")) -> List[PhoneSegment]:
    """Parse simple Praat TextGrid interval tiers into phone segments.

    This parser intentionally handles the common text TextGrid format used by
    forced aligners. It is not a full Praat parser, but it fails softly by
    returning an empty list when no interval tier can be read.
    """

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    tiers = re.split(r"item \[\d+\]:", text)
    chosen = ""
    for tier in tiers:
        name_match = re.search(r'name\s*=\s*"([^"]+)"', tier)
        if name_match and name_match.group(1).lower() in tier_names:
            chosen = tier
            break
    if not chosen and len(tiers) > 1:
        chosen = tiers[-1]
    out: List[PhoneSegment] = []
    for block in re.split(r"intervals \[\d+\]:", chosen):
        xmin = re.search(r"xmin\s*=\s*([0-9.]+)", block)
        xmax = re.search(r"xmax\s*=\s*([0-9.]+)", block)
        mark = re.search(r'text\s*=\s*"([^"]*)"', block)
        if not xmin or not xmax or mark is None:
            continue
        phone = mark.group(1).strip()
        if not phone or phone in {"sil", "sp", "pau", "<eps>"}:
            continue
        out.append(PhoneSegment(phone=phone, start=float(xmin.group(1)), end=float(xmax.group(1)), confidence=1.0))
    return out


def parse_lab_phone_segments(path: str | Path) -> List[PhoneSegment]:
    """Parse simple HTK/JVS style lab files: start end phone."""

    out: List[PhoneSegment] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        phone = parts[2].strip()
        if not phone or phone in {"sil", "sp", "pau", "<eps>"}:
            continue
        out.append(PhoneSegment(phone=phone, start=start, end=end, confidence=1.0))
    return out
