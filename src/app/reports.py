"""Downloadable evidence reports for one visitor's ranked lots."""

from __future__ import annotations

import csv
import io
from functools import lru_cache
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.pdfgen.textobject import bidiShapedText

from src.app.i18n import normalize_locale, translate, translate_message
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

FONTS = Path(__file__).parent / "fonts"
FONT_MAP = {
    "en": ("Helvetica", "Helvetica-Bold", False),
    "hi": ("NotoSansDevanagari", "NotoSansDevanagari-Bold", True),
    "te": ("NotoSansTelugu", "NotoSansTelugu-Bold", True),
}


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _warnings(entry: RankedLot, locale: str) -> str:
    assessment = entry.reading.assessment
    messages = dict.fromkeys(
        (*assessment.suppression_reasons, *assessment.model_notes)
    )
    return "; ".join(translate_message(locale, message) for message in messages)


def _csv_cell(value: str) -> str:
    return f"'{value}" if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def _rows(ranking: list[RankedLot], locale: str = "en") -> list[dict[str, str | int]]:
    rows = []
    for entry in ranking:
        reading = entry.reading
        assessment = reading.assessment
        biological = "; ".join(
            f"{translate(locale, f'class.{name}')}: {value:.2f}"
            for name, value in sorted(reading.biological_pct.items())
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
                "mode": translate(
                    locale, f"mode.{assessment.mode.replace('+absolute', '_absolute')}"
                ),
                "days_to_threshold": _number(assessment.days_to_threshold),
                "days_to_threshold_error_pct": _number(
                    assessment.days_to_threshold_error_pct
                ),
                "calibrated": translate(
                    locale, "value.yes" if reading.calibration is not None else "value.no"
                ),
                "model_warnings": _warnings(entry, locale),
            }
        )
    return rows


def csv_report(ranking: list[RankedLot], locale: str = "en") -> bytes:
    code = normalize_locale(locale)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([translate(code, f"csv.{field}") for field in CSV_FIELDS])
    for row in _rows(ranking, code):
        row["lot_id"] = _csv_cell(str(row["lot_id"]))
        row["tied_with"] = _csv_cell(str(row["tied_with"]))
        writer.writerow([row[field] for field in CSV_FIELDS])
    return output.getvalue().encode("utf-8-sig")


@lru_cache(maxsize=1)
def _register_fonts() -> None:
    for name, filename in (
        ("NotoSansDevanagari", "NotoSansDevanagari-Regular.ttf"),
        ("NotoSansDevanagari-Bold", "NotoSansDevanagari-Bold.ttf"),
        ("NotoSansTelugu", "NotoSansTelugu-Regular.ttf"),
        ("NotoSansTelugu-Bold", "NotoSansTelugu-Bold.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, FONTS / filename, shapable=True))


def _fonts(locale: str) -> tuple[str, str, bool]:
    code = normalize_locale(locale)
    if code != "en":
        _register_fonts()
    return FONT_MAP[code]


def _font_runs(text: object) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for character in str(text):
        codepoint = ord(character)
        script = (
            "hi"
            if 0x0900 <= codepoint <= 0x097F
            else "te"
            if 0x0C00 <= codepoint <= 0x0C7F
            else "en"
        )
        if runs and runs[-1][0] == script:
            runs[-1] = (script, runs[-1][1] + character)
        else:
            runs.append((script, character))
    return runs


def _ellipsise(text: object, limit: int) -> str:
    value = str(text)
    return value if len(value) <= limit else f"{value[:limit - 3]}..."


def pdf_report(ranking: list[RankedLot], locale: str = "en") -> bytes:
    code = normalize_locale(locale)
    regular_font, bold_font, shaping = _fonts(code)
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

    document.setTitle(translate(code, "pdf.title"))

    page_number = 0

    def draw_string(
        x: float, y: float, text: object, *, run_shaping: bool | None = None
    ) -> None:
        document.drawString(
            x,
            y,
            str(text),
            shaping=shaping if run_shaping is None else run_shaping,
        )

    def draw_mixed_string(
        x: float, y: float, text: object, *, font_size: float, bold: bool
    ) -> None:
        runs = _font_runs(text)
        if any(script != "en" for script, _ in runs):
            _register_fonts()
        text_object = document.beginText(x, y)
        for script, run in runs:
            run_regular, run_bold, run_shaping = FONT_MAP[script]
            font_name = run_bold if bold else run_regular
            text_object.setFont(font_name, font_size)
            shaped_run, _ = bidiShapedText(
                run,
                direction=None,
                fontName=font_name,
                fontSize=font_size,
                shaping=run_shaping,
            )
            text_object.textOut(shaped_run)
        document.drawText(text_object)

    def draw_right_string(x: float, y: float, text: object) -> None:
        document.drawRightString(x, y, str(text), shaping=shaping)

    def draw_centred_string(x: float, y: float, text: object) -> None:
        document.drawCentredString(x, y, str(text), shaping=shaping)

    def heading(text: str) -> str:
        return text.upper() if code == "en" else text

    def metric(
        x: float, y: float, label: str, value: str, *, latin_value: bool = False
    ) -> None:
        document.setFillColor(muted)
        document.setFont(bold_font, 6.8)
        draw_string(x, y, heading(label))
        document.setFillColor(ink)
        document.setFont("Helvetica-Bold" if latin_value else bold_font, 10.5)
        draw_string(x, y - 15, value)

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
        document.setFont(bold_font, 21)
        draw_string(margin, height - 48, translate(code, "pdf.title"))
        document.setFillColor(HexColor("#DCE8E1"))
        document.setFont(regular_font, 9)
        draw_string(margin, height - 68, translate(code, "pdf.subtitle"))
        document.setFillColor(maize)
        document.setFont(bold_font, 7)
        draw_right_string(
            width - margin,
            height - 43,
            heading(translate(code, "pdf.session_evidence")),
        )

        summary_y = height - 155
        document.setFillColor(white)
        document.roundRect(margin, summary_y - 31, card_width, 46, 7, stroke=0, fill=1)
        metric(
            margin + 18,
            summary_y,
            translate(code, "pdf.lots_analysed"),
            str(len(ranking)),
        )
        metric(
            margin + 180,
            summary_y,
            translate(code, "pdf.order"),
            translate(code, "pdf.fastest_first"),
        )
        metric(
            margin + 390,
            summary_y,
            translate(code, "pdf.primary_output"),
            translate(code, "pdf.relative_ranking"),
        )

        document.setFillColor(muted)
        document.setFont(regular_font, 7)
        draw_centred_string(
            width / 2, 22, f"{translate(code, 'pdf.page')} {page_number}"
        )
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
        document.setFont(bold_font, 7)
        draw_centred_string(
            badge_x + 21,
            badge_y + 19,
            heading(translate(code, "pdf.rank")),
        )
        document.setFont(bold_font, 12)
        draw_centred_string(badge_x + 21, badge_y + 6, row["rank"])

        document.setFillColor(ink)
        draw_mixed_string(
            margin + 68,
            top - 28,
            _ellipsise(row["lot_id"], 44),
            font_size=14,
            bold=True,
        )
        document.setFillColor(muted)
        document.setFont(regular_font, 7.5)
        if row["tied_with"]:
            tie_label = f"{translate(code, 'pdf.tied_with')} "
            draw_string(margin + 68, top - 43, tie_label)
            tie_x = margin + 68 + pdfmetrics.stringWidth(
                tie_label, regular_font, 7.5
            )
            draw_mixed_string(
                tie_x,
                top - 43,
                _ellipsise(row["tied_with"], 50),
                font_size=7.5,
                bold=False,
            )
        else:
            draw_string(
                margin + 68,
                top - 43,
                translate(code, "pdf.independent_rank"),
            )

        document.setStrokeColor(rule)
        document.setLineWidth(0.6)
        document.line(margin + 14, top - 57, margin + card_width - 14, top - 57)

        col_1 = margin + 18
        col_2 = margin + 184
        col_3 = margin + 350
        row_1 = top - 77
        row_2 = top - 117
        metric(
            col_1,
            row_1,
            translate(code, "pdf.temperature"),
            f"{row['temperature_c']} °C",
            latin_value=True,
        )
        metric(
            col_2,
            row_1,
            translate(code, "pdf.moisture"),
            f"{row['moisture_pct_wb']}% wb",
            latin_value=True,
        )
        metric(
            col_3,
            row_1,
            translate(code, "pdf.kernels_counted"),
            str(row["kernels_counted"]),
        )
        metric(
            col_1,
            row_2,
            translate(code, "pdf.mechanical_damage"),
            f"{row['mechanical_damage_mass_pct']}% {translate(code, 'pdf.mass')}",
        )
        metric(
            col_2,
            row_2,
            translate(code, "pdf.resolvable_gap"),
            f"{row['resolvable_gap_pct']} {translate(code, 'pdf.points')}",
        )
        metric(
            col_3,
            row_2,
            translate(code, "pdf.degradation_rate"),
            str(row["degradation_rate"]),
            latin_value=True,
        )

        band_x = margin + 14
        band_y = bottom + 12
        band_width = card_width - 28
        document.setFillColor(pale_green)
        document.roundRect(band_x, band_y, band_width, 29, 5, stroke=0, fill=1)
        document.setFillColor(forest)
        document.setFont(bold_font, 7)
        draw_string(
            band_x + 12,
            band_y + 17,
            heading(translate(code, "pdf.days_to_loss")),
        )
        document.setFont(bold_font, 10)
        days = row["days_to_threshold"]
        day_value = (
            f"{days} {translate(code, 'value.days')}"
            if days
            else translate(code, "value.withheld")
        )
        draw_string(band_x + 12, band_y + 6, day_value)
        document.setFont(bold_font, 7)
        draw_right_string(
            band_x + band_width - 12,
            band_y + 11,
            f"{translate(code, 'pdf.mode')}: {row['mode']}",
        )

    y = start_page()
    for row in _rows(ranking, code):
        if y - card_height < 34:
            document.showPage()
            y = start_page()
        draw_card(row, y)
        y -= card_height + card_gap

    document.save()
    return output.getvalue()
