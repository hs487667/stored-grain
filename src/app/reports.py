"""Downloadable evidence reports for one visitor's ranked lots."""

from __future__ import annotations

import csv
import io
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.pipeline import RankedLot


CSV_FIELDS = (
    "rank",
    "tied_with",
    "lot_id",
    "temperature_c",
    "moisture_pct_wb",
    "kernels_counted",
    "mechanical_damage_mass_pct",
    "mechanical_damage_count_pct",
    "biological_damage_pct",
    "resolvable_gap_pct",
    "degradation_rate",
    "mode",
    "days_to_threshold",
    "days_to_threshold_error_pct",
    "calibrated",
    "model_warnings",
)


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _warnings(entry: RankedLot) -> str:
    assessment = entry.reading.assessment
    messages = dict.fromkeys(
        (*assessment.suppression_reasons, *assessment.model_notes)
    )
    return "; ".join(messages)


def _csv_cell(value: str) -> str:
    return f"'{value}" if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def _rows(ranking: list[RankedLot]) -> list[dict[str, str | int]]:
    rows = []
    for entry in ranking:
        reading = entry.reading
        assessment = reading.assessment
        biological = "; ".join(
            f"{name}: {value:.2f}" for name, value in sorted(reading.biological_pct.items())
        )
        rows.append(
            {
                "rank": entry.rank,
                "tied_with": "; ".join(entry.tied_with),
                "lot_id": reading.lot_id,
                "temperature_c": _number(reading.temperature_c),
                "moisture_pct_wb": _number(reading.moisture_pct_wb),
                "kernels_counted": reading.kernels_counted,
                "mechanical_damage_mass_pct": _number(reading.damage_mass_pct),
                "mechanical_damage_count_pct": _number(reading.damage_count_pct),
                "biological_damage_pct": biological,
                "resolvable_gap_pct": _number(reading.resolvable_gap_pct),
                "degradation_rate": f"{assessment.degradation_rate:.6f}",
                "mode": assessment.mode,
                "days_to_threshold": _number(assessment.days_to_threshold),
                "days_to_threshold_error_pct": _number(
                    assessment.days_to_threshold_error_pct
                ),
                "calibrated": "yes" if reading.calibration is not None else "no",
                "model_warnings": _warnings(entry),
            }
        )
    return rows


def csv_report(ranking: list[RankedLot]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in _rows(ranking):
        row["lot_id"] = _csv_cell(str(row["lot_id"]))
        row["tied_with"] = _csv_cell(str(row["tied_with"]))
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _safe(text: object) -> str:
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


def pdf_report(ranking: list[RankedLot]) -> bytes:
    output = io.BytesIO()
    document = canvas.Canvas(output, pagesize=A4, pageCompression=0, invariant=1)
    width, height = A4
    margin = 42
    y = height - margin

    def line(text: object, *, bold: bool = False, size: int = 9) -> None:
        nonlocal y
        if y < margin + 18:
            document.showPage()
            y = height - margin
        document.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        document.drawString(margin, y, _safe(text))
        y -= size + 5

    line("Maize lot evidence report", bold=True, size=16)
    line("Lots ranked fastest-degrading first. Results are provisional.")
    y -= 8

    for row in _rows(ranking):
        line(f"Rank {row['rank']}  |  {row['lot_id']}", bold=True, size=12)
        if row["tied_with"]:
            line(f"Tied with: {row['tied_with']}")
        line(
            f"Temperature: {row['temperature_c']} C  |  "
            f"Moisture: {row['moisture_pct_wb']}% wet basis"
        )
        line(
            f"Mechanical damage: {row['mechanical_damage_mass_pct']}% by mass  |  "
            f"{row['mechanical_damage_count_pct']}% by count"
        )
        line(
            f"Kernels counted: {row['kernels_counted']}  |  "
            f"Resolvable gap: {row['resolvable_gap_pct']} percentage points"
        )
        line(f"Degradation rate: {row['degradation_rate']}  |  Mode: {row['mode']}")
        if row["days_to_threshold"]:
            line(
                f"Days to 0.5% dry-matter-loss threshold: {row['days_to_threshold']} "
                f"(+/- {row['days_to_threshold_error_pct']}%)"
            )
        else:
            line("Days to threshold: withheld")
        if row["biological_damage_pct"]:
            line(f"Biological observations: {row['biological_damage_pct']}")
        for warning_line in textwrap.wrap(str(row["model_warnings"]), width=96):
            line(f"Note: {warning_line}")
        y -= 10

    document.save()
    return output.getvalue()
