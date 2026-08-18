# Multilingual Interface and Reports Design

**Date:** 2026-08-18  
**Status:** Approved for implementation planning

## Goal

Add a three-choice language switcher—English, Hindi, and Telugu—that translates all application-owned user-facing content, including generated PDF and CSV reports. English remains the default on every page load, and the selection is not persisted.

## Scope

The selected language applies to:

- Static page headings, labels, instructions, buttons, validation boundaries, and footer text.
- Dynamic analysis results, ranking summaries, status messages, warnings, errors, legends, and accessibility announcements.
- PDF titles, headings, table labels, generated explanations, warnings, and other report-owned text.
- CSV headers and generated textual values such as boolean labels, mode labels, and model warnings.

The following content is not translated:

- User-entered lot names.
- Numeric measurements and identifiers.
- Scientific units and notation such as `°C`, `%`, and `wb`.
- Model names, formulas, and literature citations such as `Steele 1967` and `Bern 2002`.

## User Experience

The header contains a compact segmented language control with three choices:

- `English`
- `हिन्दी`
- `తెలుగు`

English is selected whenever the page loads. Selecting Hindi or Telugu updates the complete interface immediately without a page reload. Existing analysis results and session rankings are re-rendered in the chosen language; analysis is not repeated and stored measurements are not changed.

The active language controls PDF and CSV downloads. Download requests include the selected language as `?lang=en`, `?lang=hi`, or `?lang=te`. Changing language after an analysis therefore changes report presentation only, never report data.

The document language attribute and all application-owned accessibility labels, live-region announcements, and status messages update with the interface. The language control remains labelled in each language's native script so users can always identify the choices.

## Localization Architecture

Use one curated translation catalog as the source of truth for browser and server rendering. Store locale resources as JSON files for `en`, `hi`, and `te`. Every translatable concept has a stable semantic key and may contain named interpolation parameters.

The browser localization layer will:

- Load the English catalog and the selected locale catalog from local static assets.
- Translate static elements marked with localization keys, including text, placeholders, titles, and ARIA attributes.
- Expose a small translation function for JavaScript-generated content.
- Re-render current results and ranking when the locale changes.
- Fall back to English when a locale or individual key is unavailable.
- Keep locale in memory only; it must not use local storage, cookies, or server-side persistence.

The server localization layer will:

- Validate report locale query parameters against `en`, `hi`, and `te`.
- Read the same locale resources used by the browser.
- Use English when the parameter is absent or unsupported.
- Translate report labels and known generated messages through the same stable keys.

This avoids separate browser and Python dictionaries, which would drift, and avoids live translation services, which would add network dependency, cost, privacy concerns, and inconsistent scientific terminology.

## Dynamic Messages

All system-generated text must pass through the localization layer. This includes client validation errors, network failures, HTTP status messages, result caveats, calibration warnings, tie explanations, reset/removal confirmations, and empty states.

Existing backend warnings that currently arrive as English prose must be mapped to stable localization keys before presentation. Parameters such as counts, temperatures, percentages, and lot names remain data and are interpolated without translation. Unknown messages fall back to their original English text so the interface remains usable, while catalog-coverage tests prevent known messages from silently escaping localization.

## PDF Reports

PDFs retain the current polished, note-free report structure. English continues to use the current typography. Hindi and Telugu use bundled open-source fonts that cover Devanagari and Telugu glyphs, rather than operating-system fonts, so output is portable across development and deployment environments.

ReportLab character shaping for Indic scripts must be enabled where required. Because this support is experimental, automated PDF generation tests are supplemented by rendered-page visual inspection for both languages. The implementation must verify that conjuncts, vowel marks, line wrapping, table widths, and page breaks render correctly.

Report filenames remain predictable and include the locale, for example `maize-lot-report-hi.pdf` and `maize-lot-report-te.pdf`.

## CSV Reports

CSV output uses translated headers and system-generated textual values for the requested language. The underlying values remain semantically identical across languages, and user-entered content remains untouched.

CSV responses include a UTF-8 byte-order mark so Indic scripts open correctly in common spreadsheet applications, especially Microsoft Excel. CSV quoting remains standards-compliant. Filenames include the locale, for example `maize-lot-report-hi.csv`.

## Error Handling

- Missing or invalid report locale: use English.
- Missing translation key: use the English value.
- Missing English key: expose the key during development and fail catalog-integrity tests.
- Locale resource fetch failure: retain or restore English and show a localized English error.
- Font registration or shaping failure: report generation fails explicitly instead of producing corrupted or unreadable text.

## Translation Quality

Hindi and Telugu copy will use simple, natural language suitable for farmers, students, and grain operators. Scientific terms must preserve the meaning of moisture content, kernel damage, biological damage, degradation rate, calibration, and uncertainty. Translation is curated in project resources and never generated at request time.

## Testing and Verification

Automated checks will cover:

- Catalog completeness: every English key exists in Hindi and Telugu.
- Placeholder parity: interpolated variables match across all locales.
- Browser localization of static text, attributes, dynamic results, warnings, errors, and accessibility announcements.
- Instant switching among all three languages without changing analysis data.
- English default after a fresh page load and no persisted locale state.
- Report endpoints for `en`, `hi`, `te`, missing locale, and invalid locale.
- PDF generation and extractable translated text for all three languages.
- UTF-8 BOM, translated headers, generated values, quoting, and untouched user data in CSV output.
- Regression coverage for existing English UI and report behavior.

Manual verification will cover desktop and mobile layouts in Hindi and Telugu, plus rendered PDF pages for glyph shaping, clipping, wrapping, hierarchy, and readability.

## Acceptance Criteria

1. User can select English, Hindi, or Telugu from every normal application state.
2. All application-owned visible and accessible text changes to the selected language without reloading.
3. Existing results and rankings remain numerically identical when language changes.
4. PDF and CSV downloads use the currently selected language.
5. Hindi and Telugu PDF text is readable, correctly shaped, and not clipped.
6. Hindi and Telugu CSV files display correctly in UTF-8-capable spreadsheet software.
7. A page refresh restores English.
8. No live translation API or external runtime service is required.
9. Existing English workflows and report layout remain functional.

