"""The HTTP surface: photograph in, reading out, lots ranked.

One process, one pipeline, one lock. There is a single set of weights and a
single MPS context, and a capstone demonstration has a single photographer, so
requests are serialised rather than made concurrent -- concurrency here would
add failure modes and buy nothing.

The pipeline is built lazily and once. Loading weights takes 1.2 seconds, which
is fine at startup and unusable per request, and deferring it means the tests
can inject a fake and never touch the models at all.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.app.images import NotAnImage, decode, to_working_scale
from src.app.overlay import draw
from src.app.sessions import Session, SessionRegistry
from src.pipeline import LotReading

STATIC = Path(__file__).parent / "static"

#: Name of the cookie carrying the session id.
COOKIE = "session_id"


def serialise(reading: LotReading) -> dict:
    """One reading as JSON, carrying its own limits.

    The limits travel with the number deliberately. A damage percentage without
    its resolvable gap, its mode, and whether it was corrected is a figure
    someone will quote out of context -- most likely in a review.
    """
    assessment = reading.assessment
    return {
        "lot_id": reading.lot_id,
        "kernels_counted": reading.kernels_counted,
        "damage_mass_pct": reading.damage_mass_pct,
        "damage_count_pct": reading.damage_count_pct,
        "raw_damage_mass_pct": reading.raw_damage_mass_pct,
        "calibrated": reading.calibration is not None,
        "biological_pct": reading.biological_pct,
        "class_counts": reading.class_counts,
        "temperature_c": reading.temperature_c,
        "moisture_pct_wb": reading.moisture_pct_wb,
        "resolvable_gap_pct": reading.resolvable_gap_pct,
        "degradation_rate": assessment.degradation_rate,
        "mode": assessment.mode,
        "days_to_threshold": assessment.days_to_threshold,
        "suppression_reasons": list(assessment.suppression_reasons),
        "ranges_ok": assessment.ranges.all_ok,
        "range_notes": list(assessment.ranges.notes),
    }


def _default_pipeline():
    from src.pipeline import Pipeline

    return Pipeline()


def create_app(
    pipeline_factory=_default_pipeline, registry: SessionRegistry | None = None
):
    app = FastAPI(title="Maize damage capture")
    registry = registry if registry is not None else SessionRegistry()

    state = {"pipeline": None}

    # One set of weights and one Metal context, so inference is serialised
    # across every visitor. Sessions isolate what people see, not what the
    # hardware can do at once.
    lock = threading.Lock()

    def pipeline():
        if state["pipeline"] is None:
            state["pipeline"] = pipeline_factory()
        return state["pipeline"]

    def visitor(request: Request, response: Response) -> Session:
        """The caller's session, re-issuing the cookie on every request.

        Re-issuing rather than setting it once keeps the cookie's lifetime in
        step with the session's own idle timeout, so a browser does not hold an
        id the server has already forgotten.
        """
        session_id, session = registry.get_or_create(request.cookies.get(COOKIE))
        response.set_cookie(
            COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
            max_age=int(registry.idle_timeout_s),
            path="/",
        )
        return session

    @app.post("/api/lots", status_code=201)
    async def add_lot(
        photo: UploadFile = File(...),
        lot_id: str = Form(...),
        temperature_c: float = Form(...),
        moisture_pct_wb: float = Form(...),
        session: Session = Depends(visitor),
    ):
        raw = await photo.read()
        try:
            image = to_working_scale(decode(raw))
        except NotAnImage as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        with lock, TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            image.save(path)
            try:
                reading, detections = pipeline().analyse(
                    path,
                    lot_id=lot_id,
                    temperature_c=temperature_c,
                    moisture_pct_wb=moisture_pct_wb,
                )
            except ValueError as exc:
                # A failed photograph never mutates the session.
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            buffer = io.BytesIO()
            draw(image, detections).save(buffer, format="PNG")
            session.overlays[lot_id] = buffer.getvalue()
            session.photographs[lot_id] = raw
            session.store.add(reading)

        return serialise(reading)

    @app.get("/api/lots")
    def list_lots(session: Session = Depends(visitor)):
        return {
            "lots": [serialise(r) for r in session.store.readings()],
            "ranking": [
                {
                    "lot_id": entry.reading.lot_id,
                    "rank": entry.rank,
                    "tied_with": list(entry.tied_with),
                    "degradation_rate": entry.reading.assessment.degradation_rate,
                    "damage_mass_pct": entry.reading.damage_mass_pct,
                    "resolvable_gap_pct": entry.reading.resolvable_gap_pct,
                }
                for entry in session.store.ranking()
            ],
        }

    @app.get("/api/lots/{lot_id}/overlay.png")
    def overlay(lot_id: str, session: Session = Depends(visitor)):
        if lot_id not in session.overlays:
            raise HTTPException(status_code=404, detail=f"no lot {lot_id!r}")
        return Response(content=session.overlays[lot_id], media_type="image/png")

    @app.delete("/api/lots/{lot_id}", status_code=204)
    def delete_lot(lot_id: str, session: Session = Depends(visitor)):
        if not session.store.remove(lot_id):
            raise HTTPException(status_code=404, detail=f"no lot {lot_id!r}")
        session.overlays.pop(lot_id, None)
        session.photographs.pop(lot_id, None)
        return Response(status_code=204)

    @app.post("/api/session/reset", status_code=204)
    def reset(session: Session = Depends(visitor)):
        session.store.reset()
        session.overlays.clear()
        session.photographs.clear()
        return Response(status_code=204)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
