"""Alama: camera-based fingerprint verification — feasibility experiment.

Pipeline stages (one module each, each independently testable and swappable):
    capture -> segment -> enhance -> minutiae -> template -> match -> evaluate
"""

__version__ = "0.1.0"
