import math

from alama.models import Minutia, MinutiaType, Template
from alama.template import from_iso19794_2, to_iso19794_2, to_xyt

TEMPLATE = Template(
    minutiae=[
        Minutia(10, 20, 0.0, MinutiaType.ENDING, 50),
        Minutia(123, 45, math.pi / 2, MinutiaType.BIFURCATION, 80),
        Minutia(300, 400, 3 * math.pi / 2 + 0.01, MinutiaType.ENDING, 0),
    ],
    width=416,
    height=512,
    dpi=500,
)


def test_iso_header_layout():
    data = to_iso19794_2(TEMPLATE)
    assert data[:8] == b"FMR\x00 20\x00"
    assert int.from_bytes(data[8:12], "big") == len(data)
    assert len(data) == 24 + 4 + 6 * 3 + 2
    assert int.from_bytes(data[14:16], "big") == 416
    assert int.from_bytes(data[16:18], "big") == 512
    assert int.from_bytes(data[18:20], "big") == 197  # 500 dpi in px/cm


def test_iso_roundtrip():
    back = from_iso19794_2(to_iso19794_2(TEMPLATE))
    assert (back.width, back.height, back.dpi) == (416, 512, 500)
    assert len(back) == len(TEMPLATE)
    for a, b in zip(TEMPLATE.minutiae, back.minutiae, strict=True):
        assert (a.x, a.y, a.kind, a.quality) == (b.x, b.y, b.kind, b.quality)
        diff = (a.angle - b.angle + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) <= math.radians(1.40625) / 2 + 1e-9


def test_xyt_lines():
    assert to_xyt(TEMPLATE).splitlines() == ["10 20 0 50", "123 45 90 80", "300 400 271 0"]


def test_empty_template():
    empty = Template([], width=10, height=10)
    assert to_xyt(empty) == ""
    assert len(from_iso19794_2(to_iso19794_2(empty))) == 0
