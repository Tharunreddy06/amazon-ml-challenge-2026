
"""
Text preprocessing utilities for business entity resolution.

Person A - Task 2
"""

import re
import unicodedata
from typing import Optional


# ---------------------------------------------------------
# State normalization
# ---------------------------------------------------------

STATE_MAPPING = {
    # United States
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
}


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

ADDRESS_ABBREVIATIONS = {
    "road": "rd",
    "avenue": "ave",
    "street": "st",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "parkway": "pkwy",
    "place": "pl",
    "court": "ct",
    "circle": "cir",
    "terrace": "ter",
    "apartment": "apt",
    "suite": "ste",
}


# ---------------------------------------------------------
# Business suffix normalization
# ---------------------------------------------------------

BUSINESS_SUFFIX_MAPPING = {
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "limited": "ltd",
    "ltd": "ltd",
    "incorporated": "inc",
    "inc": "inc",
    "llc": "llc",
    "corporation": "corp",
    "corp": "corp",
}


# Optional noise words.
# These are NOT removed by default.
NOISE_WORDS = {
    "center",
    "dba",
    "group",
}


# ---------------------------------------------------------
# General normalization
# ---------------------------------------------------------

def normalize_text(value: Optional[str]) -> str:
    """
    General text normalization.

    - Handles missing values.
    - Applies Unicode normalization.
    - Uses casefold().
    - Replaces punctuation with spaces.
    - Collapses repeated whitespace.
    """

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()

    # Replace punctuation and symbols with spaces.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Replace underscores with spaces.
    value = value.replace("_", " ")

    # Collapse repeated whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ---------------------------------------------------------
# State normalization
# ---------------------------------------------------------

def normalize_states(text: str) -> str:
    """
    Convert full US state names to their abbreviations.

    Example:
        Illinois -> il
        Virginia -> va
    """

    for state_name, abbreviation in sorted(
        STATE_MAPPING.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"\b{re.escape(state_name)}\b"
        text = re.sub(pattern, abbreviation, text)

    return text


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_address_terms(address: str) -> str:
    """
    Normalize common address terms.

    Example:
        road -> rd
        avenue -> ave
    """

    for full_term, abbreviation in sorted(
        ADDRESS_ABBREVIATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"\b{re.escape(full_term)}\b"
        address = re.sub(pattern, abbreviation, address)

    return address


# ---------------------------------------------------------
# Business name normalization
# ---------------------------------------------------------

def normalize_business_suffixes(name: str) -> str:
    """
    Normalize common business suffixes.

    Example:
        private limited -> pvt ltd
        incorporated -> inc
    """

    for full_term, normalized_term in sorted(
        BUSINESS_SUFFIX_MAPPING.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"\b{re.escape(full_term)}\b"
        name = re.sub(pattern, normalized_term, name)

    return name


def remove_noise_words(
    name: str,
    enabled: bool = False,
) -> str:
    """
    Optionally remove selected noise words.

    Disabled by default to reduce the risk of false matches.
    """

    if not enabled:
        return name

    tokens = name.split()

    filtered_tokens = [
        token for token in tokens
        if token not in NOISE_WORDS
    ]

    return " ".join(filtered_tokens)


# ---------------------------------------------------------
# Field-specific normalization
# ---------------------------------------------------------

def normalize_business_name(
    name: Optional[str],
    remove_noise: bool = False,
) -> str:
    """
    Normalize a business name.
    """

    normalized = normalize_text(name)
    normalized = normalize_business_suffixes(normalized)
    normalized = remove_noise_words(
        normalized,
        enabled=remove_noise,
    )

    return normalized


def normalize_business_address(
    address: Optional[str],
) -> str:
    """
    Normalize a business address.
    """

    normalized = normalize_text(address)
    normalized = normalize_states(normalized)
    normalized = normalize_address_terms(normalized)

    return normalized


def normalize_country(country: Optional[str]) -> str:
    """
    Normalize country labels.
    """

    return normalize_text(country)


# ---------------------------------------------------------
# Complete record preprocessing
# ---------------------------------------------------------

def preprocess_record(record: dict) -> dict:
    """
    Normalize relevant fields in one business record.

    Original values are preserved.
    Normalized fields are added.
    """

    processed = record.copy()

    processed["normalized_name"] = normalize_business_name(
        record.get("business_name")
    )

    processed["normalized_address"] = normalize_business_address(
        record.get("business_address")
    )

    processed["normalized_country"] = normalize_country(
        record.get("country")
    )

    return processed