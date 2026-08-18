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


@pytest.mark.parametrize("locale", ("hi", "te"))
def test_model_warning_prose_is_localized(locale):
    moisture = translate(
        locale,
        "message.range.moisture",
        value="20",
        dry="25",
        low="10",
        high="18",
    )
    estimate = translate(
        locale,
        "message.model.days_estimate",
        error="7.5",
        threshold="0.5",
    )
    assert "wet basis" not in moisture
    assert "dry-matter-loss" not in estimate


@pytest.mark.parametrize(
    ("locale", "boundary_markers", "ranking_markers", "footer_markers"),
    [
        (
            "en",
            ("synthetic trays", "laboratory rig", "phone photograph", "ground truth", "provisional"),
            ("degradation rate", "Steele", "temperature", "moisture", "damage ranges"),
            ("published deterioration model", "hand-counting", "measurement", "not the storage model"),
        ),
        (
            "hi",
            ("कृत्रिम ट्रे", "प्रयोगशाला", "फ़ोन", "ग्राउंड ट्रुथ", "अस्थायी"),
            ("क्षरण दर", "Steele", "तापमान", "नमी", "क्षति"),
            ("प्रकाशित क्षरण मॉडल", "हाथ से", "मापन", "भंडारण मॉडल नहीं"),
        ),
        (
            "te",
            ("కృత్రిమ ట్రే", "ప్రయోగశాల", "ఫోన్", "గ్రౌండ్ ట్రూత్", "తాత్కాలిక"),
            ("క్షీణత రేటు", "Steele", "ఉష్ణోగ్రత", "తేమ", "నష్టం"),
            ("ప్రచురిత క్షీణత నమూనా", "చేతితో", "కొలత", "నిల్వ నమూనాను కాదు"),
        ),
    ],
)
def test_interface_preserves_scientific_boundary_copy(
    locale, boundary_markers, ranking_markers, footer_markers
):
    assert all(marker in translate(locale, "boundary.text") for marker in boundary_markers)
    assert all(
        marker in translate(locale, "ranking.description") for marker in ranking_markers
    )
    assert all(marker in translate(locale, "footer.description") for marker in footer_markers)


@pytest.mark.parametrize(
    ("locale", "markers"),
    [
        ("en", ("Steele (1967)", "Appendix D", "page 108", "Thompson (1972)", "not been obtained", "not used")),
        ("hi", ("Steele (1967)", "परिशिष्ट D", "पृष्ठ 108", "Thompson (1972)", "प्राप्त नहीं", "उपयोग नहीं")),
        ("te", ("Steele (1967)", "అనుబంధం D", "108వ పేజీ", "Thompson (1972)", "పొందలేదు", "ఉపయోగించలేదు")),
    ],
)
def test_steele_warning_keeps_source_and_missing_model_caveats(locale, markers):
    warning = translate(locale, "message.model.steele_only")
    assert all(marker in warning for marker in markers)
