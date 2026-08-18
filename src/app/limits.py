"""What an open URL needs that a LAN address does not.

Every upload runs a segmenter and a classifier on this machine. That makes the
endpoint expensive in a way an ordinary form post is not, and it is the reason
these guards exist: the cost of a request is paid in inference, not bandwidth.
"""

from __future__ import annotations

import time
from collections import deque

#: Largest upload accepted, before decoding. A 200 MP phone photograph lands
#: around 40 MB; past this the caller is not photographing a tray.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

#: Largest decoded image accepted, in pixels. A compressed file says nothing
#: about what it expands to -- a few hundred kilobytes of PNG can decode to
#: gigabytes of pixels, which is a way to exhaust this process without ever
#: reaching the upload cap.
MAX_IMAGE_PIXELS = 120_000_000

#: Measurements per address per window. Generous for somebody photographing a
#: row of lots, quick to bite on a script.
UPLOAD_LIMIT = 12
UPLOAD_WINDOW_S = 60.0


class TooLarge(Exception):
    """The upload exceeded a size the endpoint accepts."""


async def read_capped(upload, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an upload, refusing it as soon as it passes the cap.

    Chunked rather than a single read so an oversized body is abandoned partway
    instead of being brought into memory in full and measured afterwards, which
    is the failure the cap exists to prevent.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise TooLarge(
                f"the photograph is larger than {limit // (1024 * 1024)} MB"
            )
        chunks.append(chunk)
    return b"".join(chunks)


class RateLimiter:
    """A sliding window of request times per caller.

    Sliding rather than fixed: a fixed window lets a caller spend a full
    allowance on either side of the boundary and take double the rate in an
    instant, which on this endpoint means double the inference.
    """

    def __init__(
        self,
        limit: int = UPLOAD_LIMIT,
        window_s: float = UPLOAD_WINDOW_S,
        clock=time.monotonic,
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._seen: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """Record a request. Returns seconds to wait if it is over the limit.

        A refused request is not recorded, so a caller who keeps hammering does
        not push their own window further out with every attempt.
        """
        now = self._clock()
        self._forget_idle(now)

        times = self._seen.setdefault(key, deque())
        while times and now - times[0] >= self._window_s:
            times.popleft()

        if len(times) >= self._limit:
            return self._window_s - (now - times[0])

        times.append(now)
        return None

    def _forget_idle(self, now: float) -> None:
        stale = [
            key
            for key, times in self._seen.items()
            if not times or now - times[-1] >= self._window_s
        ]
        for key in stale:
            del self._seen[key]

    def __len__(self) -> int:
        return len(self._seen)
