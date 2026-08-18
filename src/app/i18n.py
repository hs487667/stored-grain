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
