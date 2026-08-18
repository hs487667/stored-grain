"""Downloadable evidence reports for one visitor's ranked lots."""

from __future__ import annotations

import csv
import io

from reportlab.lib.colors import HexColor
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


def _ellipsise(text: object, limit: int) -> str:
    value = _safe(text)
    return value if len(value) <= limit else f"{value[:limit - 3]}..."


def pdf_report(ranking: list[RankedLot]) -> bytes:
    output = io.BytesIO()
    document = canvas.Canvas(output, pagesize=A4, pageCompression=0, invariant=1)
    width, height = A4
    margin = 36
    card_width = width - (margin * 2)
    card_height = 174
    card_gap = 14

    forest = HexColor("#173F32")
    maize = HexColor("#D5A62E")
    paper = HexColor("#F4F1E8")
    ink = HexColor("#17231E")
    muted = HexColor("#68736D")
    rule = HexColor("#D8DDD8")
    pale_green = HexColor("#E5EFE9")
    white = HexColor("#FFFFFF")

    document.setTitle("Maize lot evidence report")

    page_number = 0

    def metric(x: float, y: float, label: str, value: str) -> None:
        document.setFillColor(muted)
        document.setFont("Helvetica-Bold", 6.8)
        document.drawString(x, y, label.upper())
        document.setFillColor(ink)
        document.setFont("Helvetica-Bold", 10.5)
        document.drawString(x, y - 15, _safe(value))

    def start_page() -> float:
        nonlocal page_number
        page_number += 1
        document.setFillColor(paper)
        document.rect(0, 0, width, height, stroke=0, fill=1)

        document.setFillColor(forest)
        document.rect(0, height - 108, width, 108, stroke=0, fill=1)
        document.setFillColor(maize)
        document.rect(0, height - 111, width, 3, stroke=0, fill=1)

        document.setFillColor(white)
        document.setFont("Helvetica-Bold", 21)
        document.drawString(margin, height - 48, "Maize lot report")
        document.setFillColor(HexColor("#DCE8E1"))
        document.setFont("Helvetica", 9)
        document.drawString(margin, height - 68, "Mechanical damage and deterioration ranking")
        document.setFillColor(maize)
        document.setFont("Helvetica-Bold", 7)
        document.drawRightString(width - margin, height - 43, "SESSION EVIDENCE")

        summary_y = height - 155
        document.setFillColor(white)
        document.roundRect(margin, summary_y - 31, card_width, 46, 7, stroke=0, fill=1)
        metric(margin + 18, summary_y, "Lots analysed", str(len(ranking)))
        metric(margin + 180, summary_y, "Order", "Fastest degrading first")
        metric(margin + 390, summary_y, "Primary output", "Relative ranking")

        document.setFillColor(muted)
        document.setFont("Helvetica", 7)
        document.drawCentredString(width / 2, 22, f"Page {page_number}")
        return summary_y - 55

    def draw_card(row: dict[str, str | int], top: float) -> None:
        bottom = top - card_height
        document.setFillColor(white)
        document.roundRect(margin, bottom, card_width, card_height, 9, stroke=0, fill=1)

        badge_x = margin + 14
        badge_y = top - 46
        document.setFillColor(maize)
        document.roundRect(badge_x, badge_y, 42, 31, 6, stroke=0, fill=1)
        document.setFillColor(forest)
        document.setFont("Helvetica-Bold", 7)
        document.drawCentredString(badge_x + 21, badge_y + 19, "RANK")
        document.setFont("Helvetica-Bold", 12)
        document.drawCentredString(badge_x + 21, badge_y + 6, str(row["rank"]))

        document.setFillColor(ink)
        document.setFont("Helvetica-Bold", 14)
        document.drawString(margin + 68, top - 28, _ellipsise(row["lot_id"], 44))
        document.setFillColor(muted)
        document.setFont("Helvetica", 7.5)
        tie = f"Tied with {row['tied_with']}" if row["tied_with"] else "Independent rank"
        document.drawString(margin + 68, top - 43, _ellipsise(tie, 70))

        document.setStrokeColor(rule)
        document.setLineWidth(0.6)
        document.line(margin + 14, top - 57, margin + card_width - 14, top - 57)

        col_1 = margin + 18
        col_2 = margin + 184
        col_3 = margin + 350
        row_1 = top - 77
        row_2 = top - 117
        metric(col_1, row_1, "Temperature", f"{row['temperature_c']} C")
        metric(col_2, row_1, "Moisture", f"{row['moisture_pct_wb']}% wb")
        metric(col_3, row_1, "Kernels counted", str(row["kernels_counted"]))
        metric(
            col_1,
            row_2,
            "Mechanical damage",
            f"{row['mechanical_damage_mass_pct']}% mass",
        )
        metric(col_2, row_2, "Resolvable gap", f"{row['resolvable_gap_pct']} points")
        metric(col_3, row_2, "Degradation rate", str(row["degradation_rate"]))

        band_x = margin + 14
        band_y = bottom + 12
        band_width = card_width - 28
        document.setFillColor(pale_green)
        document.roundRect(band_x, band_y, band_width, 29, 5, stroke=0, fill=1)
        document.setFillColor(forest)
        document.setFont("Helvetica-Bold", 7)
        document.drawString(band_x + 12, band_y + 17, "DAYS TO 0.5% DML")
        document.setFont("Helvetica-Bold", 10)
        days = row["days_to_threshold"]
        day_value = f"{days} days" if days else "Withheld"
        document.drawString(band_x + 12, band_y + 6, day_value)
        document.setFont("Helvetica-Bold", 7)
        document.drawRightString(
            band_x + band_width - 12,
            band_y + 11,
            f"Mode: {row['mode']}",
        )

    y = start_page()
    for row in _rows(ranking):
        if y - card_height < 34:
            document.showPage()
            y = start_page()
        draw_card(row, y)
        y -= card_height + card_gap

    document.save()
    return output.getvalue()
