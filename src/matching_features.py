
"""
Feature generation for business entity matching.

This module compares two preprocessed business records and
generates matching features for the entity resolution model.
"""

from difflib import SequenceMatcher
from typing import Dict, Any


def safe_string(value: Any) -> str:
    """
    Convert a value into a safe lowercase string.
    Missing values are converted to an empty string.
    """
    if value is None:
        return ""

    return str(value).strip().lower()


def exact_match(value_a: Any, value_b: Any) -> int:
    """
    Check whether two values match exactly.

    Returns:
        1: Exact match
        0: No match

    Empty values do not count as a match.
    """
    a = safe_string(value_a)
    b = safe_string(value_b)

    if not a or not b:
        return 0

    return int(a == b)


def missing_indicator(value: Any) -> int:
    """
    Identify whether a value is missing.

    Returns:
        1: Value is missing
        0: Value is present
    """
    value = safe_string(value)
    return int(not value)


def similarity_score(value_a: Any, value_b: Any) -> float:
    """
    Calculate character-level similarity using SequenceMatcher.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    a = safe_string(value_a)
    b = safe_string(value_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_jaccard_similarity(value_a: Any, value_b: Any) -> float:
    """
    Calculate token-based Jaccard similarity.

    Formula:
        intersection(tokens_a, tokens_b) /
        union(tokens_a, tokens_b)

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    a = safe_string(value_a)
    b = safe_string(value_b)

    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    union = tokens_a | tokens_b

    if not union:
        return 0.0

    intersection = tokens_a & tokens_b

    return len(intersection) / len(union)


def generate_matching_features(
    record_a: Dict[str, Any],
    record_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate matching features for two business records.

    Expected normalized fields:
        normalized_name
        normalized_address
        normalized_country

    The function also supports raw fields as fallback.
    """

    name_a = record_a.get(
        "normalized_name",
        record_a.get("business_name", "")
    )

    name_b = record_b.get(
        "normalized_name",
        record_b.get("business_name", "")
    )

    address_a = record_a.get(
        "normalized_address",
        record_a.get("business_address", "")
    )

    address_b = record_b.get(
        "normalized_address",
        record_b.get("business_address", "")
    )

    country_a = record_a.get(
        "normalized_country",
        record_a.get("country", "")
    )

    country_b = record_b.get(
        "normalized_country",
        record_b.get("country", "")
    )

    features = {
        # Exact matching features
        "exact_name_match": exact_match(name_a, name_b),
        "exact_address_match": exact_match(address_a, address_b),
        "exact_country_match": exact_match(country_a, country_b),

        # Character-level similarity
        "name_similarity": similarity_score(name_a, name_b),
        "address_similarity": similarity_score(address_a, address_b),

        # Token-level similarity
        "name_token_similarity": token_jaccard_similarity(
            name_a,
            name_b
        ),
        "address_token_similarity": token_jaccard_similarity(
            address_a,
            address_b
        ),

        # Missing value indicators
        "name_a_missing": missing_indicator(name_a),
        "name_b_missing": missing_indicator(name_b),
        "address_a_missing": missing_indicator(address_a),
        "address_b_missing": missing_indicator(address_b),
        "country_a_missing": missing_indicator(country_a),
        "country_b_missing": missing_indicator(country_b),

        # Basic length features
        "name_length_difference": abs(
            len(safe_string(name_a)) -
            len(safe_string(name_b))
        ),

        "address_length_difference": abs(
            len(safe_string(address_a)) -
            len(safe_string(address_b))
        ),
    }

    return features