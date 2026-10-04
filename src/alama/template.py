"""Stage 4b — template: serialise minutiae to interchange formats.

- ISO/IEC 19794-2:2005 finger minutiae record (single view), readable by
  SourceAFIS (FingerprintCompatibility.importTemplate) and most commercial SDKs.
- NBIS .xyt text (x y theta [quality]) for BOZORTH3.

Templates are biometric data: write them only under data/ or a temp directory.
"""

from __future__ import annotations

import math
import struct

from alama.models import Minutia, MinutiaType, Template

_ISO_TYPE = {MinutiaType.OTHER: 0b00, MinutiaType.ENDING: 0b01, MinutiaType.BIFURCATION: 0b10}
_ISO_TYPE_REV = {v: k for k, v in _ISO_TYPE.items()}
_ISO_ANGLE_UNIT = 2 * math.pi / 256  # 1.40625 degrees


def to_iso19794_2(
    template: Template,
    finger_position: int = 0,  # 0 = unknown; 1..10 per ISO finger codes
    impression_type: int = 0,  # 0 = live-scan plain; 8 = contactless (later editions)
    finger_quality: int = 0,
) -> bytes:
    """Encode as an ISO/IEC 19794-2:2005 record (one finger view, no extended data)."""
    minutiae = template.minutiae[:255]
    res = round(template.dpi / 2.54)  # pixels per cm
    body = bytearray()
    body += struct.pack(
        ">BBBB", finger_position, impression_type & 0x0F, finger_quality, len(minutiae)
    )
    for m in minutiae:
        x = min(max(int(round(m.x)), 0), 0x3FFF)
        y = min(max(int(round(m.y)), 0), 0x3FFF)
        angle = int(round((m.angle % (2 * math.pi)) / _ISO_ANGLE_UNIT)) % 256
        body += struct.pack(
            ">HHBB", (_ISO_TYPE[m.kind] << 14) | x, y, angle, min(max(m.quality, 0), 100)
        )
    body += struct.pack(">H", 0)  # extended data block length

    header_len = 24
    header = (
        b"FMR\x00"
        + b" 20\x00"
        + struct.pack(
            ">IHHHHHBB",
            header_len + len(body),
            0,  # capture equipment compliance (4 bits) + id (12 bits): unspecified
            template.width,
            template.height,
            res,
            res,
            1,  # number of finger views
            0,  # reserved
        )
    )
    return header + bytes(body)


def from_iso19794_2(data: bytes) -> Template:
    """Decode the first finger view of an ISO/IEC 19794-2:2005 record."""
    if data[:4] != b"FMR\x00" or data[4:8] != b" 20\x00":
        raise ValueError("not an ISO/IEC 19794-2:2005 record")
    _, _, width, height, xres, _, _, _ = struct.unpack(">IHHHHHBB", data[8:24])
    _, _, _, count = struct.unpack(">BBBB", data[24:28])
    minutiae = []
    offset = 28
    for _ in range(count):
        tx, y, angle, quality = struct.unpack(">HHBB", data[offset : offset + 6])
        offset += 6
        minutiae.append(
            Minutia(
                x=float(tx & 0x3FFF),
                y=float(y & 0x3FFF),
                angle=angle * _ISO_ANGLE_UNIT,
                kind=_ISO_TYPE_REV.get(tx >> 14, MinutiaType.OTHER),
                quality=quality,
            )
        )
    return Template(minutiae, width=width, height=height, dpi=round(xres * 2.54))


def to_xyt(template: Template) -> str:
    """NBIS .xyt text: one minutia per line, 'x y theta_degrees quality'.

    Uses the ISO-style convention of alama (origin top-left, CCW degrees), i.e. the
    same convention as `mindtct -m1`. Only compare .xyt files from one extractor.
    """
    lines = [
        f"{round(m.x)} {round(m.y)} {round(math.degrees(m.angle)) % 360} {m.quality}"
        for m in template.minutiae
    ]
    return "\n".join(lines) + ("\n" if lines else "")
