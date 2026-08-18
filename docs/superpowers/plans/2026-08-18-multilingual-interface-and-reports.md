# Multilingual Interface and Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an English, Hindi, and Telugu language switcher that localizes the complete browser interface and the generated PDF and CSV reports.

**Architecture:** Three JSON locale catalogs under the static directory are the shared source of truth. A small Python module reads them for server reports and converts model prose into stable message descriptors; a dependency-free browser module reads the same files and translates static and generated UI. Report endpoints accept `?lang=en|hi|te`; ReportLab uses bundled Noto fonts and HarfBuzz shaping for Hindi and Telugu.

**Tech Stack:** Python 3.14, FastAPI 0.141.1, vanilla JavaScript, CSS, ReportLab 5.0.0, uharfbuzz 0.55.0, pypdf 6.14.2, pytest 9.1.1, Node 26 built-in test runner

**Spec:** `docs/superpowers/specs/2026-08-18-multilingual-interface-and-reports-design.md`

## Global Constraints

- Supported locale codes are exactly `en`, `hi`, and `te`.
- English is selected after every page load; do not use local storage, cookies, or server-side locale persistence.
- Translate all application-owned visible and accessible text, including generated errors, warnings, PDF labels, and CSV headers and textual values.
- Never translate user-entered lot names, numeric values, identifiers, units, model names, formulas, or literature citations.
- Existing results and rankings must re-render without rerunning analysis when language changes.
- PDF reports remain polished and note-free.
- CSV output must begin with a UTF-8 BOM and retain spreadsheet-formula neutralization.
- No live translation API, third-party runtime script, or third-party runtime font request.
- Missing or invalid locales and missing localized keys fall back to English.
- ReportLab Indic shaping is experimental; rendered Hindi and Telugu PDFs require visual verification in addition to tests.

---

### Task 1: Shared Locale Catalogs and Python Localization API

**Files:**
- Create: `src/app/i18n.py`
- Create: `src/app/static/locales/en.json`
- Create: `src/app/static/locales/hi.json`
- Create: `src/app/static/locales/te.json`
- Create: `tests/test_i18n.py`

**Interfaces:**
- Produces: `SUPPORTED_LOCALES: tuple[str, ...]`
- Produces: `normalize_locale(locale: str | None) -> str`
- Produces: `catalog(locale: str | None) -> dict[str, str]`
- Produces: `translate(locale: str | None, key: str, **values: object) -> str`
- Produces: `describe_message(message: str) -> dict[str, object]`
- Produces: `translate_message(locale: str | None, message: str) -> str`
- Locale files are flat JSON objects whose values are strings and whose interpolation placeholders use Python/JavaScript-compatible `{name}` syntax.

- [ ] **Step 1: Write failing catalog and fallback tests**

Create `tests/test_i18n.py`:

```python
import json
import re
from pathlib import Path

import pytest

from src.app.i18n import (
    SUPPORTED_LOCALES,
    catalog,
    describe_message,
    normalize_locale,
    translate,
    translate_message,
)


LOCALES = Path("src/app/static/locales")
PLACEHOLDERS = re.compile(r"\{([a-z_]+)\}")


def test_supported_locales_are_fixed_and_invalid_values_fall_back_to_english():
    assert SUPPORTED_LOCALES == ("en", "hi", "te")
    assert normalize_locale(None) == "en"
    assert normalize_locale("") == "en"
    assert normalize_locale("HI") == "hi"
    assert normalize_locale("te-IN") == "te"
    assert normalize_locale("fr") == "en"


def test_every_locale_has_the_same_keys_and_placeholders():
    catalogs = {
        locale: json.loads((LOCALES / f"{locale}.json").read_text(encoding="utf-8"))
        for locale in SUPPORTED_LOCALES
    }
    assert set(catalogs["hi"]) == set(catalogs["en"])
    assert set(catalogs["te"]) == set(catalogs["en"])
    for key, english in catalogs["en"].items():
        expected = set(PLACEHOLDERS.findall(english))
        assert set(PLACEHOLDERS.findall(catalogs["hi"][key])) == expected, key
        assert set(PLACEHOLDERS.findall(catalogs["te"][key])) == expected, key


def test_translation_interpolates_and_falls_back_to_english():
    assert translate("hi", "status.measured", lot="A-4") == "A-4 मापा गया।"
    assert translate("te", "status.measured", lot="A-4") == "A-4 కొలవబడింది."
    assert translate("fr", "status.measured", lot="A-4") == "Measured A-4."
    assert translate("hi", "missing.key") == "missing.key"


@pytest.mark.parametrize(
    ("message", "key"),
    [
        ("temperature 60 C is outside Steele's tested 5-40 C", "message.range.temperature"),
        ("damage 55% is outside the published 2-40%", "message.range.damage"),
        (
            "Days to threshold is a Steele (1967) estimate and carries his stated "
            "standard error of 7.5% at the 0.5% dry-matter-loss level.",
            "message.model.days_estimate",
        ),
    ],
)
def test_model_messages_become_stable_descriptors(message, key):
    assert describe_message(message)["key"] == key


def test_known_model_message_is_localized_and_unknown_message_is_preserved():
    translated = translate_message(
        "hi", "temperature 60 C is outside Steele's tested 5-40 C"
    )
    assert "60" in translated and "5-40" in translated
    assert "तापमान" in translated
    assert translate_message("te", "unrecognized model message") == "unrecognized model message"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_i18n.py -q
```

Expected: collection fails because `src.app.i18n` does not exist.

- [ ] **Step 3: Create complete English, Hindi, and Telugu catalogs**

Use flat semantic keys. Each file must contain this complete key inventory; do not leave English values in Hindi or Telugu except preserved proper names, units, and citations:

```text
meta.title, meta.description, skip.capture, language.aria
topbar.brand, topbar.prototype
boundary.label, boundary.text
hero.eyebrow, hero.title.line1, hero.title.line2, hero.description
premise.measurement.label, premise.measurement.value
premise.decision.label, premise.decision.value
premise.reference.label, premise.reference.value
flow.aria, flow.capture.title, flow.capture.detail
flow.measure.title, flow.measure.detail, flow.rank.title, flow.rank.detail
capture.index, capture.title, capture.description
field.lot, field.lot.placeholder, field.temperature, field.moisture
field.photo, field.photo.primary, field.photo.secondary, action.measure
result.index, result.title, result.measured
ranking.index, ranking.title, ranking.method, ranking.description
action.download_pdf, action.download_csv, action.clear_session
footer.description
error.not_image, error.no_kernels, error.lot_missing, error.server_request
error.photo_too_large, error.too_many, error.server_unavailable
error.rejected, error.bad_reply, error.network, error.unknown, error.language_load
validation.lot, validation.photo, validation.temperature, validation.moisture
status.measuring, status.measured, status.overlay_failed
status.removing, status.removed, status.clearing, status.cleared
confirm.clear_session
ranking.empty, ranking.rate, ranking.remove, ranking.tie
reading.overlay_alt, reading.legend.damage, reading.legend.sound
reading.damage_by_weight, reading.resolvable, reading.kernels_counted
reading.by_count, reading.temperature, reading.moisture
reading.days_to_loss, reading.mode, reading.biological
reading.none_reported, reading.uncorrected, reading.threshold_withheld
reading.outside_range, reading.range_unknown
mode.ranking, mode.ranking_absolute
value.yes, value.no, value.withheld, value.days
class.mould_suspect, class.insect_damaged, class.sprouted
class.heated, class.discoloured
message.range.temperature, message.range.moisture, message.range.damage
message.model.steele_only, message.model.days_estimate
pdf.title, pdf.subtitle, pdf.session_evidence, pdf.lots_analysed
pdf.order, pdf.fastest_first, pdf.primary_output, pdf.relative_ranking
pdf.page, pdf.rank, pdf.tied_with, pdf.independent_rank
pdf.temperature, pdf.moisture, pdf.kernels_counted
pdf.mechanical_damage, pdf.mass, pdf.resolvable_gap, pdf.points
pdf.degradation_rate, pdf.days_to_loss, pdf.mode
csv.rank, csv.tied_with, csv.lot_id, csv.temperature_c
csv.moisture_pct_wb, csv.kernels_counted
csv.mechanical_damage_mass_pct, csv.mechanical_damage_count_pct
csv.biological_damage_pct, csv.resolvable_gap_pct
csv.degradation_rate, csv.mode, csv.days_to_threshold
csv.days_to_threshold_error_pct, csv.calibrated, csv.model_warnings
```

Required exact parameter contracts:

```json
{
  "error.rejected": "The server rejected that request ({status}).",
  "status.measured": "Measured {lot}.",
  "status.overlay_failed": "Measured {lot}, but the outlined image did not load. Reload to see it.",
  "status.removing": "Removing {lot}…",
  "status.removed": "Removed {lot}.",
  "ranking.remove": "Remove {lot}",
  "ranking.tie": "Cannot be separated at this sample size — the damage gap is smaller than {gap} points.",
  "reading.resolvable": "±{gap} resolvable · {count} kernels counted",
  "reading.threshold_withheld": "Days to threshold withheld: {reasons}",
  "reading.outside_range": "Outside the published validity range: {reasons}",
  "message.range.temperature": "Temperature {value} C is outside Steele's tested {low}-{high} C.",
  "message.range.moisture": "Moisture {value}% w.b. ({dry}% d.b.) is outside the published {low}-{high}% wet basis.",
  "message.range.damage": "Damage {value}% is outside the published {low}-{high}%.",
  "message.model.days_estimate": "Days to threshold is a Steele (1967) estimate and carries his stated standard error of {error}% at the {threshold}% dry-matter-loss level."
}
```

Hindi and Telugu must preserve the same named placeholders. Use simple terms: `मक्का`/`మొక్కజొన్న` for maize, `नमी`/`తేమ` for moisture, `यांत्रिक क्षति`/`యాంత్రిక నష్టం` for mechanical damage, and `क्षरण दर`/`క్షీణత రేటు` for degradation rate. Keep `Steele 1967`, `Bern 2002`, `°C`, `%`, and `wb` unchanged.

- [ ] **Step 4: Implement the Python loader, formatter, and model-message descriptors**

Create `src/app/i18n.py` with cached JSON loading and explicit regexes for all five generated model-message forms:

```python
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


SUPPORTED_LOCALES = ("en", "hi", "te")
LOCALES = Path(__file__).parent / "static" / "locales"


def normalize_locale(locale: str | None) -> str:
    candidate = (locale or "en").lower().split("-", 1)[0]
    return candidate if candidate in SUPPORTED_LOCALES else "en"


@lru_cache(maxsize=3)
def catalog(locale: str | None) -> dict[str, str]:
    code = normalize_locale(locale)
    return json.loads((LOCALES / f"{code}.json").read_text(encoding="utf-8"))


def translate(locale: str | None, key: str, **values: object) -> str:
    english = catalog("en")
    template = catalog(locale).get(key, english.get(key, key))
    return template.format(**values)


MESSAGE_PATTERNS = (
    (
        re.compile(r"temperature (?P<value>[-0-9.]+) C is outside Steele's tested (?P<low>[-0-9.]+)-(?P<high>[-0-9.]+) C"),
        "message.range.temperature",
    ),
    (
        re.compile(r"moisture (?P<value>[-0-9.]+)% w\.b\. \((?P<dry>[-0-9.]+)% d\.b\.\) is outside the published (?P<low>[-0-9.]+)-(?P<high>[-0-9.]+)% wet basis"),
        "message.range.moisture",
    ),
    (
        re.compile(r"damage (?P<value>[-0-9.]+)% is outside the published (?P<low>[-0-9.]+)-(?P<high>[-0-9.]+)%"),
        "message.range.damage",
    ),
    (
        re.compile(r"Days to threshold is a Steele \(1967\) estimate and carries his stated standard error of (?P<error>[-0-9.]+)% at the (?P<threshold>[-0-9.]+)% dry-matter-loss level\."),
        "message.model.days_estimate",
    ),
)


def describe_message(message: str) -> dict[str, object]:
    for pattern, key in MESSAGE_PATTERNS:
        if match := pattern.fullmatch(message):
            return {"key": key, "values": match.groupdict(), "fallback": message}
    if message.startswith("Every figure here is Steele (1967):"):
        return {"key": "message.model.steele_only", "values": {}, "fallback": message}
    return {"key": "", "values": {}, "fallback": message}


def translate_message(locale: str | None, message: str) -> str:
    descriptor = describe_message(message)
    key = str(descriptor["key"])
    if not key:
        return message
    return translate(locale, key, **dict(descriptor["values"]))
```

- [ ] **Step 5: Run localization tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_i18n.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/i18n.py src/app/static/locales tests/test_i18n.py
git commit -m "feat: add shared language catalogs"
```

---

### Task 2: Localized API Messages and CSV Reports

**Files:**
- Modify: `src/app/server.py:25-225`
- Modify: `src/app/reports.py:15-92`
- Modify: `tests/test_server.py:181-334`

**Interfaces:**
- Consumes: `normalize_locale`, `describe_message`, `translate`, and `translate_message` from Task 1.
- Produces: `serialise(reading)` fields `suppression_messages`, `model_messages`, and `range_messages`, each a list of `{key, values, fallback}` objects while retaining existing English arrays for API compatibility.
- Changes: `csv_report(ranking: list[RankedLot], locale: str = "en") -> bytes`.
- Changes: `/api/report.csv?lang=<locale>` localizes headers and generated values.

- [ ] **Step 1: Write failing API descriptor and localized CSV tests**

Add to `tests/test_server.py`:

```python
@pytest.mark.parametrize(
    ("lang", "rank_header", "yes_value", "suffix"),
    [
        ("hi", "क्रम", "नहीं", "-hi.csv"),
        ("te", "ర్యాంక్", "కాదు", "-te.csv"),
    ],
)
def test_csv_report_uses_requested_language(client, lang, rank_header, yes_value, suffix):
    _upload(client, "lot_5.0")

    response = client.get(f"/api/report.csv?lang={lang}")
    text = response.content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert response.content.startswith(b"\xef\xbb\xbf")
    assert rank_header in rows[0]
    assert rows[0][rank_header] == "1"
    assert yes_value in rows[0].values()
    assert rows[0]["लॉट_नाम" if lang == "hi" else "లాట్_పేరు"] == "lot_5.0"
    assert suffix in response.headers["content-disposition"]


def test_invalid_csv_language_falls_back_to_existing_english_contract(client):
    _upload(client, "lot_5.0")
    response = client.get("/api/report.csv?lang=fr")
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
    assert rows[0]["rank"] == "1"
    assert 'filename="maize-lot-report.csv"' in response.headers["content-disposition"]


def test_reading_exposes_localizable_model_message_descriptors(client):
    body = _upload(client, "lot_5.0").json()
    assert body["model_messages"]
    assert all({"key", "values", "fallback"} <= set(item) for item in body["model_messages"])
```

Update existing CSV tests to decode with `utf-8-sig` before `csv.DictReader`; keep every existing ranking, evidence, session-isolation, and spreadsheet-injection assertion.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_server.py -q -k "csv or descriptor"
```

Expected: failures for missing descriptors, missing locale parameter behavior, untranslated headers, and missing BOM.

- [ ] **Step 3: Add stable message descriptors to serialized readings**

In `src/app/server.py`, import `describe_message` and add:

```python
        "suppression_messages": [describe_message(text) for text in assessment.suppression_reasons],
        "model_messages": [describe_message(text) for text in assessment.model_notes],
        "range_messages": [describe_message(text) for text in assessment.ranges.notes],
```

Do not remove `suppression_reasons`, `model_notes`, or `range_notes`.

- [ ] **Step 4: Localize CSV headers and generated text**

Refactor `src/app/reports.py` so internal row keys remain `CSV_FIELDS`, but `csv.writer` emits translated headers in field order. Pass locale through `_warnings`, `_rows`, and `csv_report`:

```python
def _warnings(entry: RankedLot, locale: str) -> str:
    assessment = entry.reading.assessment
    messages = dict.fromkeys((*assessment.suppression_reasons, *assessment.model_notes))
    return "; ".join(translate_message(locale, message) for message in messages)


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
```

Within `_rows`, localize only generated strings:

```python
"mode": translate(locale, f"mode.{assessment.mode.replace('+absolute', '_absolute')}")
"calibrated": translate(locale, "value.yes" if reading.calibration is not None else "value.no")
"model_warnings": _warnings(entry, locale)
```

Translate biological class display names with `class.<name>` but keep source class IDs unchanged elsewhere.

- [ ] **Step 5: Pass normalized locale through the CSV endpoint and filename**

In `src/app/server.py`:

```python
    @app.get("/api/report.csv")
    def download_csv(lang: str = "en", session: Session = Depends(visitor)):
        locale = normalize_locale(lang)
        suffix = "" if locale == "en" else f"-{locale}"
        return Response(
            content=csv_report(report_ranking(session), locale),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    f'attachment; filename="maize-lot-report{suffix}.csv"'
            },
        )
```

- [ ] **Step 6: Run focused and full server tests**

Run:

```bash
.venv/bin/pytest tests/test_i18n.py tests/test_server.py -q
```

Expected: all tests pass, including unchanged English report behavior.

- [ ] **Step 7: Commit**

```bash
git add src/app/server.py src/app/reports.py tests/test_server.py
git commit -m "feat: localize CSV reports"
```

---

### Task 3: Unicode Hindi and Telugu PDF Reports

**Files:**
- Modify: `requirements.txt:7`
- Create: `src/app/fonts/NotoSansDevanagari-Regular.ttf`
- Create: `src/app/fonts/NotoSansDevanagari-Bold.ttf`
- Create: `src/app/fonts/NotoSansTelugu-Regular.ttf`
- Create: `src/app/fonts/NotoSansTelugu-Bold.ttf`
- Create: `src/app/fonts/OFL.txt`
- Modify: `src/app/reports.py:95-220`
- Modify: `src/app/server.py:218-225`
- Modify: `tests/test_server.py:300-324`

**Interfaces:**
- Consumes: locale normalization and translations from Task 1.
- Changes: `pdf_report(ranking: list[RankedLot], locale: str = "en") -> bytes`.
- Produces: `_fonts(locale: str) -> tuple[str, str, bool]`, returning regular font name, bold font name, and shaping flag.
- Changes: `/api/report.pdf?lang=<locale>` returns a localized filename and body.

- [ ] **Step 1: Pin and install the PDF extraction test dependency**

Add:

```text
pypdf==6.14.2
```

Install it:

```bash
.venv/bin/pip install pypdf==6.14.2
```

- [ ] **Step 2: Write failing localized PDF tests**

Add to `tests/test_server.py`:

```python
from pypdf import PdfReader


@pytest.mark.parametrize(
    ("lang", "title", "suffix"),
    [
        ("hi", "मक्का लॉट रिपोर्ट", "-hi.pdf"),
        ("te", "మొక్కజొన్న లాట్ నివేదిక", "-te.pdf"),
    ],
)
def test_pdf_report_uses_requested_language_and_embedded_unicode_font(
    client, lang, title, suffix
):
    _upload(client, "lot_5.0")
    response = client.get(f"/api/report.pdf?lang={lang}")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(response.content)).pages)

    assert response.content.startswith(b"%PDF-")
    assert title in text
    assert "lot_5.0" in text
    assert "Note:" not in text
    assert suffix in response.headers["content-disposition"]


def test_invalid_pdf_language_falls_back_to_english(client):
    _upload(client, "lot_5.0")
    response = client.get("/api/report.pdf?lang=fr")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(response.content)).pages)
    assert "Maize lot report" in text
    assert 'filename="maize-lot-report.pdf"' in response.headers["content-disposition"]
```

- [ ] **Step 3: Run PDF tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_server.py -q -k "pdf_report or pdf_language"
```

Expected: localized title and localized filename assertions fail.

- [ ] **Step 4: Pin shaping dependency and add official Noto font assets**

Add `uharfbuzz==0.55.0` to `requirements.txt`, install it, then download regular and bold static TTF files plus their license from the official `notofonts/noto-fonts` repository:

```bash
.venv/bin/pip install uharfbuzz==0.55.0
mkdir -p src/app/fonts
curl -L https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf -o src/app/fonts/NotoSansDevanagari-Regular.ttf
curl -L https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf -o src/app/fonts/NotoSansDevanagari-Bold.ttf
curl -L https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf -o src/app/fonts/NotoSansTelugu-Regular.ttf
curl -L https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Bold.ttf -o src/app/fonts/NotoSansTelugu-Bold.ttf
curl -L https://raw.githubusercontent.com/notofonts/noto-fonts/main/LICENSE -o src/app/fonts/OFL.txt
file src/app/fonts/*.ttf
```

Expected: four `TrueType Font data` results. Stop if any download is HTML or empty.

- [ ] **Step 5: Register fonts and make every PDF text draw shaping-aware**

In `src/app/reports.py`, register fonts once with `reportlab.pdfbase.pdfmetrics.registerFont` and `TTFont(..., shapable=True)`. Keep Helvetica for English:

```python
FONTS = Path(__file__).parent / "fonts"
FONT_MAP = {
    "en": ("Helvetica", "Helvetica-Bold", False),
    "hi": ("NotoSansDevanagari", "NotoSansDevanagari-Bold", True),
    "te": ("NotoSansTelugu", "NotoSansTelugu-Bold", True),
}


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
```

Remove Latin-1 coercion from `_safe`; it corrupts Indic text. Replace all direct PDF labels with `translate(locale, "pdf.<key>", ...)`. Route every call through small helpers that call `drawString`, `drawRightString`, or `drawCentredString` with `shaping=shaping`. Preserve existing colors, cards, pagination, rank data, and note-free output. Use translated catalog strings instead of `.upper()` for Indic headings.

Keep values composed from unchanged units:

```python
f"{row['temperature_c']} °C"
f"{row['moisture_pct_wb']}% wb"
f"{row['mechanical_damage_mass_pct']}% {translate(locale, 'pdf.mass')}"
```

- [ ] **Step 6: Pass locale through the PDF endpoint**

Mirror the CSV route:

```python
    @app.get("/api/report.pdf")
    def download_pdf(lang: str = "en", session: Session = Depends(visitor)):
        locale = normalize_locale(lang)
        suffix = "" if locale == "en" else f"-{locale}"
        return Response(
            content=pdf_report(report_ranking(session), locale),
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    f'attachment; filename="maize-lot-report{suffix}.pdf"'
            },
        )
```

- [ ] **Step 7: Run PDF and report tests**

Run:

```bash
.venv/bin/pytest tests/test_i18n.py tests/test_server.py -q
```

Expected: all tests pass and extracted Hindi/Telugu text includes the localized title.

- [ ] **Step 8: Render and inspect both Indic PDFs**

Start the app on a separate port, create one sample lot through the existing test fixture or browser, download both PDFs, and render them:

```bash
pdftoppm -png -r 150 /tmp/maize-lot-report-hi.pdf /tmp/maize-hi
pdftoppm -png -r 150 /tmp/maize-lot-report-te.pdf /tmp/maize-te
```

Inspect every rendered page. Verify conjuncts and vowel marks are shaped, no tofu boxes appear, labels do not overlap values, titles and cards do not clip, and page breaks match the English report.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt src/app/fonts src/app/reports.py src/app/server.py tests/test_server.py
git commit -m "feat: render Hindi and Telugu PDF reports"
```

---

### Task 4: Browser Language Switcher and Complete UI Localization

**Files:**
- Create: `src/app/static/i18n.js`
- Create: `tests/test_i18n_js.cjs`
- Modify: `src/app/static/index.html:1-157`
- Modify: `src/app/static/app.js:3-578`
- Modify: `src/app/static/app.css:120-170,1035-1086`
- Modify: `tests/test_server.py:181-200`

**Interfaces:**
- Consumes: `/static/locales/{locale}.json` from Task 1 and localized message descriptors from Task 2.
- Produces: global `window.GrainI18n` with `init()`, `setLocale(locale)`, `getLocale()`, `t(key, values)`, `message(descriptor)`, `apply(root)`, and `subscribe(listener)`.
- Produces: language buttons with `data-locale`, `aria-pressed`, and native-script labels.
- Produces: in-memory `currentSession`, `shownReading`, and `shownOverlaySrc` state used to re-render without another analysis request.

- [ ] **Step 1: Write failing dependency-free JavaScript localization tests**

Create `tests/test_i18n_js.cjs` using `node:test` and `node:assert/strict`. Stub `fetch` to return the three parsed locale files, then require `src/app/static/i18n.js`:

```javascript
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");

const catalogs = Object.fromEntries(["en", "hi", "te"].map((locale) => [
  `/static/locales/${locale}.json`,
  JSON.parse(fs.readFileSync(`src/app/static/locales/${locale}.json`, "utf8"))
]));

global.fetch = async (path) => ({
  ok: true,
  json: async () => catalogs[path]
});

const i18n = require("../src/app/static/i18n.js");

test("starts in English and does not read persisted state", async () => {
  await i18n.init();
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "Measured A-4.");
});

test("switches locale and interpolates without changing supplied data", async () => {
  await i18n.setLocale("te");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "A-4 కొలవబడింది.");
});

test("invalid locale and missing localized keys fall back to English", async () => {
  await i18n.setLocale("fr");
  assert.equal(i18n.getLocale(), "en");
  assert.equal(i18n.t("status.measured", { lot: "A-4" }), "Measured A-4.");
});
```

- [ ] **Step 2: Add failing HTML language-control tests**

Extend `tests/test_server.py`:

```python
def test_page_exposes_accessible_non_persistent_language_choices(client):
    response = client.get("/")
    assert 'class="language-switcher"' in response.text
    assert 'data-locale="en"' in response.text
    assert 'data-locale="hi"' in response.text
    assert 'data-locale="te"' in response.text
    assert "हिन्दी" in response.text
    assert "తెలుగు" in response.text
    assert 'src="/static/i18n.js"' in response.text
    assert "localStorage" not in response.text
```

- [ ] **Step 3: Run browser localization tests and verify RED**

Run:

```bash
node --test tests/test_i18n_js.cjs
.venv/bin/pytest tests/test_server.py -q -k "language_choices"
```

Expected: missing module and missing language-control failures.

- [ ] **Step 4: Implement the browser localization module**

Use a UMD wrapper so the same file works as `window.GrainI18n` in the browser and `module.exports` in Node. `init()` loads only English. `setLocale()` normalizes the code, fetches the selected catalog once, updates `document.documentElement.lang`, applies static translations, updates `aria-pressed`, and notifies subscribers. It must not access local storage, cookies, or URL locale state.

`apply(root)` must handle:

```text
[data-i18n]                 textContent
[data-i18n-placeholder]     placeholder
[data-i18n-aria-label]      aria-label
[data-i18n-content]         content attribute for meta description
```

`message(descriptor)` returns `t(descriptor.key, descriptor.values)` when the descriptor has a known key, otherwise `descriptor.fallback`.

- [ ] **Step 5: Mark every static string and add the language control**

Keep English text in `index.html` as progressive fallback. Add the segmented control inside `.topbar`:

```html
<div class="language-switcher" role="group" aria-label="Language" data-i18n-aria-label="language.aria">
  <button type="button" data-locale="en" aria-pressed="true">English</button>
  <button type="button" data-locale="hi" aria-pressed="false">हिन्दी</button>
  <button type="button" data-locale="te" aria-pressed="false">తెలుగు</button>
</div>
```

Add `language.aria` to all catalogs. Add localization attributes to the title, meta description, skip link, top bar, validation boundary, hero, process rail, capture form, result and ranking headers, report actions, reset button, footer, placeholders, and ARIA labels. Load scripts in dependency order:

```html
<script src="/static/i18n.js"></script>
<script src="/static/app.js"></script>
```

- [ ] **Step 6: Route every generated UI sentence through the translator**

In `app.js`, define `var t = window.GrainI18n.t` and replace literal user-facing strings in these functions:

```text
messageForStatus, request, errorText, rungMarkup, tieGroupMarkup,
renderRanking, biologicalSummary, figureMarkup, readingMarkup,
firstProblem, submit handler, remove handler, reset handler, startup error
```

Use descriptors from the server for model text:

```javascript
function translatedMessages(descriptors, fallbacks) {
  if (Array.isArray(descriptors) && descriptors.length) {
    return descriptors.map(window.GrainI18n.message);
  }
  return (fallbacks || []).map(String);
}
```

Map biological IDs with `t("class." + name)`, falling back to the original ID. Map modes through `mode.ranking` and `mode.ranking_absolute`. Never translate or alter `reading.lot_id`.

- [ ] **Step 7: Preserve current data and re-render it when locale changes**

Add in-memory state:

```javascript
var currentSession = { lots: [], ranking: [] };
var shownReading = null;
var shownOverlaySrc = null;
```

Update `applySession`, `showReading`, and `clearReading` to maintain it. Subscribe after `GrainI18n.init()`:

```javascript
window.GrainI18n.subscribe(function () {
  renderRanking(currentSession.ranking);
  if (shownReading) resultBody.innerHTML = readingMarkup(shownReading, shownOverlaySrc);
  updateReportLinks();
});
```

`updateReportLinks()` must set current-language query strings without changing endpoint paths:

```javascript
pdfLink.href = "/api/report.pdf?lang=" + encodeURIComponent(window.GrainI18n.getLocale());
csvLink.href = "/api/report.csv?lang=" + encodeURIComponent(window.GrainI18n.getLocale());
```

Initialize the application only after `GrainI18n.init()` resolves. On locale fetch failure, restore English, keep the interface usable, and show `error.language_load` from the English catalog.

- [ ] **Step 8: Style the segmented switcher for desktop, mobile, focus, and long scripts**

Add `.language-switcher` and button styles beside `.brand`/`.build-state`. Use existing forest/maize palette, visible `:focus-visible`, `[aria-pressed="true"]` active state, and minimum 44 px touch targets. At `max-width: 580px`, allow topbar wrapping and keep all three choices visible; do not abbreviate Hindi or Telugu. Add no new animation; existing reduced-motion behavior remains unchanged.

- [ ] **Step 9: Run JavaScript, server, and catalog tests**

Run:

```bash
node --test tests/test_i18n_js.cjs
.venv/bin/pytest tests/test_i18n.py tests/test_server.py -q
```

Expected: all tests pass.

- [ ] **Step 10: Manually verify instant switching**

Run the app on a test port:

```bash
.venv/bin/python -m uvicorn src.app.server:app --host 127.0.0.1 --port 8001
```

At 1440 px desktop and 390 px emulated mobile widths:

1. Confirm fresh load is English.
2. Measure a lot with a real tray image.
3. Switch to Hindi; confirm every static label, result, warning, remove label, live status, ranking line, and download button changes while all numbers and the lot name remain identical.
4. Switch to Telugu and repeat.
5. Download PDF and CSV in each selected language and confirm filenames and contents match.
6. Refresh and confirm English returns.
7. Confirm no horizontal overflow, clipped buttons, layout collisions, console errors, or new requests to `/api/lots` when switching language.

- [ ] **Step 11: Commit**

```bash
git add src/app/static/i18n.js src/app/static/index.html src/app/static/app.js src/app/static/app.css tests/test_i18n_js.cjs tests/test_server.py
git commit -m "feat: add interface language switcher"
```

---

### Task 5: Documentation and Final Verification

**Files:**
- Modify: `src/app/README.md:17-31`
- Modify: `/Users/harshsingh/Documents/grain-capstone/MASTER-PLAN-v0.3.md`
- Modify: `/Users/harshsingh/Documents/grain-capstone/GRAIN-CAPSTONE-HANDOFF.md`
- Modify: `/Users/harshsingh/Desktop/GRAIN-CAPSTONE-HANDOFF.md`

**Interfaces:**
- Consumes: completed multilingual UI and report behavior from Tasks 1-4.
- Produces: durable project context for future AI sessions and users.

- [ ] **Step 1: Document the finished behavior and constraints**

Update `src/app/README.md` to state:

- The header offers English, Hindi, and Telugu.
- English is restored on refresh and language is not persisted.
- All visible/accessibility text and generated report text is curated and offline.
- PDF and CSV use the selected language; CSV is UTF-8 BOM encoded.
- Lot names, numbers, units, model names, and citations are preserved.
- Indic PDF fonts are bundled and ReportLab shaping requires `uharfbuzz`.

Add the same completed-feature summary, endpoint query contract, new source files, dependency pins, test commands, and known experimental PDF-shaping caveat to the master plan and both handoff copies. Keep the two handoff files byte-identical after editing.

- [ ] **Step 2: Run complete automated verification**

Run:

```bash
.venv/bin/pytest -q
node --test tests/test_i18n_js.cjs
git diff --check
```

Expected: all Python and Node tests pass; `git diff --check` prints nothing.

- [ ] **Step 3: Verify generated report files outside test assertions**

For English, Hindi, and Telugu, verify:

```bash
file /tmp/maize-lot-report-*.pdf /tmp/maize-lot-report-*.csv
pdffonts /tmp/maize-lot-report-hi.pdf
pdffonts /tmp/maize-lot-report-te.pdf
```

Expected: valid PDFs, UTF-8 CSVs, and embedded Noto Devanagari/Telugu fonts. Open both CSVs in a spreadsheet application and confirm headers and warnings render correctly without mojibake.

- [ ] **Step 4: Re-check acceptance criteria line by line**

Confirm all nine acceptance criteria in the design spec against test output and manual screenshots. Record any unmet item instead of claiming completion.

- [ ] **Step 5: Commit repository documentation**

```bash
git add src/app/README.md
git commit -m "docs: document multilingual interface and reports"
```

- [ ] **Step 6: Confirm final worktree and handoff copies**

Run:

```bash
git status --short
cmp /Users/harshsingh/Documents/grain-capstone/GRAIN-CAPSTONE-HANDOFF.md /Users/harshsingh/Desktop/GRAIN-CAPSTONE-HANDOFF.md
```

Expected: clean worktree and `cmp` exit code 0.
