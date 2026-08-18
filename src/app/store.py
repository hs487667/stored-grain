"""Readings for the current session.

A session is one sitting at one tray of lots. It lives in memory and dies with
the process, which is the honest lifetime: a stored reading outliving the
models that produced it would be a reading nobody could interpret later.
"""

from __future__ import annotations

from src.pipeline import LotReading, RankedLot, rank_lots


class SessionStore:
    def __init__(self) -> None:
        self._readings: dict[str, LotReading] = {}

    def add(self, reading: LotReading) -> None:
        """Store a reading, replacing any earlier one for the same lot.

        Replacement rather than accumulation because re-photographing is how a
        bad shot gets fixed, and two readings of one lot would both reach the
        ranking as if they were two lots.
        """
        self._readings[reading.lot_id] = reading

    def get(self, lot_id: str) -> LotReading | None:
        return self._readings.get(lot_id)

    def remove(self, lot_id: str) -> bool:
        """Drop a lot. Returns whether it was there, so a caller can 404."""
        return self._readings.pop(lot_id, None) is not None

    def reset(self) -> None:
        self._readings.clear()

    def readings(self) -> list[LotReading]:
        return list(self._readings.values())

    def ranking(self) -> list[RankedLot]:
        if not self._readings:
            return []
        return rank_lots(self.readings())

    def __contains__(self, lot_id: object) -> bool:
        return lot_id in self._readings
