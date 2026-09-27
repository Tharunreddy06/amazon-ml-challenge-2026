"""
Feature generation for business entity matching.

This module compares two preprocessed business records and
generates matching features for the entity resolution model.
"""

from difflib import SequenceMatcher
from typing import Dict, Any
import re


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


def similarity_score(
    value_a: Any,
    value_b: Any
) -> float:
    """
    Calculate character-level similarity using SequenceMatcher.

    Returns:
        Similarity score between 0.0 and 1.0.
    """

    a = safe_string(value_a)
    b = safe_string(value_b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def token_jaccard_similarity(
    value_a: Any,
    value_b: Any
) -> float:
    """
    Calculate token-based Jaccard similarity.

    Formula:
        intersection(tokens_a, tokens_b) /
        union(tokens_a, tokens_b)
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


def containment_similarity(
    value_a: Any,
    value_b: Any
) -> float:
    """
    Measure whether one string is contained inside the other.

    Example:

        "abc technologies"
        "abc technologies private limited"

    The shorter string is contained inside the longer string,
    producing a high score.

    Returns:
        0.0 to 1.0
    """

    a = safe_string(value_a)
    b = safe_string(value_b)

    if not a or not b:
        return 0.0

    if a in b:
        return len(a) / len(b)

    if b in a:
        return len(b) / len(a)

    return 0.0


def token_overlap_count(
    value_a: Any,
    value_b: Any
) -> int:
    """
    Count the number of shared tokens.
    """

    a = safe_string(value_a)
    b = safe_string(value_b)

    if not a or not b:
        return 0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    return len(tokens_a & tokens_b)


def extract_address_numbers(
    address: Any
) -> set:
    """
    Extract numeric components from an address.

    Example:

        "123 main street chicago"

    returns:

        {"123"}
    """

    address = safe_string(address)

    if not address:
        return set()

    return set(
        re.findall(
            r"\d+",
            address
        )
    )


def address_number_match(
    address_a: Any,
    address_b: Any
) -> int:
    """
    Check whether the addresses share at least one
    numeric component.

    Returns:
        1: At least one number matches
        0: No matching number
    """

    numbers_a = extract_address_numbers(
        address_a
    )

    numbers_b = extract_address_numbers(
        address_b
    )

    if not numbers_a or not numbers_b:
        return 0

    return int(
        bool(numbers_a & numbers_b)
    )


def first_token_match(
    value_a: Any,
    value_b: Any
) -> int:
    """
    Check whether the first meaningful tokens match.
    """

    tokens_a = safe_string(value_a).split()
    tokens_b = safe_string(value_b).split()

    if not tokens_a or not tokens_b:
        return 0

    return int(
        tokens_a[0] == tokens_b[0]
    )


def last_token_match(
    value_a: Any,
    value_b: Any
) -> int:
    """
    Check whether the last meaningful tokens match.
    """

    tokens_a = safe_string(value_a).split()
    tokens_b = safe_string(value_b).split()

    if not tokens_a or not tokens_b:
        return 0

    return int(
        tokens_a[-1] == tokens_b[-1]
    )


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
        record_a.get(
            "business_name",
            ""
        )
    )

    name_b = record_b.get(
        "normalized_name",
        record_b.get(
            "business_name",
            ""
        )
    )

    address_a = record_a.get(
        "normalized_address",
        record_a.get(
            "business_address",
            ""
        )
    )

    address_b = record_b.get(
        "normalized_address",
        record_b.get(
            "business_address",
            ""
        )
    )

    country_a = record_a.get(
        "normalized_country",
        record_a.get(
            "country",
            ""
        )
    )

    country_b = record_b.get(
        "normalized_country",
        record_b.get(
            "country",
            ""
        )
    )

    features = {

        # --------------------------------------------------
        # Exact matching
        # --------------------------------------------------

        "exact_name_match": exact_match(
            name_a,
            name_b
        ),

        "exact_address_match": exact_match(
            address_a,
            address_b
        ),

        "exact_country_match": exact_match(
            country_a,
            country_b
        ),

        # --------------------------------------------------
        # Character-level similarity
        # --------------------------------------------------

        "name_similarity": similarity_score(
            name_a,
            name_b
        ),

        "address_similarity": similarity_score(
            address_a,
            address_b
        ),

        # --------------------------------------------------
        # Token-level similarity
        # --------------------------------------------------

        "name_token_similarity": (
            token_jaccard_similarity(
                name_a,
                name_b
            )
        ),

        "address_token_similarity": (
            token_jaccard_similarity(
                address_a,
                address_b
            )
        ),

        # --------------------------------------------------
        # NEW: Containment features
        # --------------------------------------------------

        "name_containment": containment_similarity(
            name_a,
            name_b
        ),

        "address_containment": containment_similarity(
            address_a,
            address_b
        ),

        # --------------------------------------------------
        # NEW: Shared token counts
        # --------------------------------------------------

        "name_token_overlap_count": (
            token_overlap_count(
                name_a,
                name_b
            )
        ),

        "address_token_overlap_count": (
            token_overlap_count(
                address_a,
                address_b
            )
        ),

        # --------------------------------------------------
        # NEW: Address number matching
        # --------------------------------------------------

        "address_number_match": (
            address_number_match(
                address_a,
                address_b
            )
        ),

        # --------------------------------------------------
        # NEW: Name boundary matching
        # --------------------------------------------------

        "name_first_token_match": (
            first_token_match(
                name_a,
                name_b
            )
        ),

        "name_last_token_match": (
            last_token_match(
                name_a,
                name_b
            )
        ),

        # --------------------------------------------------
        # Missing value indicators
        # --------------------------------------------------

        "name_a_missing": missing_indicator(
            name_a
        ),

        "name_b_missing": missing_indicator(
            name_b
        ),

        "address_a_missing": missing_indicator(
            address_a
        ),

        "address_b_missing": missing_indicator(
            address_b
        ),

        "country_a_missing": missing_indicator(
            country_a
        ),

        "country_b_missing": missing_indicator(
            country_b
        ),

        # --------------------------------------------------
        # Length features
        # --------------------------------------------------

        "name_length_difference": abs(
            len(safe_string(name_a))
            -
            len(safe_string(name_b))
        ),

        "address_length_difference": abs(
            len(safe_string(address_a))
            -
            len(safe_string(address_b))
        ),
    }

    return features