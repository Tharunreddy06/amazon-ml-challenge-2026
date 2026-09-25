
"""
Text preprocessing utilities for business entity resolution.

Person A - Task 1
"""

import re
import unicodedata
from typing import Optional


def normalize_text(value: Optional[str]) -> str:
    """
    General text normalization.

    - Handles missing values.
    - Applies Unicode normalization.
    - Converts text to lowercase using casefold().
    - Replaces punctuation with spaces.
    - Collapses repeated whitespace.
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # Handle empty or whitespace-only values.
    value = value.strip()

    if not value:
        return ""

    # Normalize Unicode characters while preserving language characters.
    value = unicodedata.normalize("NFKC", value)

    # Case-insensitive normalization.
    value = value.casefold()

    # Replace punctuation and symbols with spaces.
    # Unicode letters and numbers are preserved.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Replace underscores with spaces.
    value = value.replace("_", " ")

    # Collapse repeated whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalize a business name.
    """
    return normalize_text(name)


def normalize_business_address(address: Optional[str]) -> str:
    """
    Normalize a business address.

    Missing addresses are represented as an empty string.
    """
    return normalize_text(address)


def normalize_country(country: Optional[str]) -> str:
    """
    Normalize country labels.
    """
    return normalize_text(country)


def preprocess_record(record: dict) -> dict:
    """
    Normalize the relevant fields in one business record.

    Original values are preserved.
    New normalized fields are added.
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