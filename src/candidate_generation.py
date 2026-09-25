
"""
Candidate generation for business entity resolution.

This module generates possible matching records using blocking
rather than comparing every record against every other record.

The implementation uses:
    1. Country blocking
    2. Business name token blocking
    3. Multiple blocking keys to improve recall
"""

from collections import defaultdict
from typing import Dict, List, Any, Iterable, Set


def safe_string(value: Any) -> str:
    """Convert a value to a normalized safe string."""
    if value is None:
        return ""

    return str(value).strip().lower()


def get_name_tokens(record: Dict[str, Any]) -> Set[str]:
    """
    Extract tokens from a normalized business name.

    Very short tokens are ignored to reduce noisy candidates.
    """
    name = safe_string(
        record.get(
            "normalized_name",
            record.get("business_name", "")
        )
    )

    tokens = {
        token
        for token in name.split()
        if len(token) >= 3
    }

    return tokens


def get_country(record: Dict[str, Any]) -> str:
    """Extract the normalized country from a record."""
    return safe_string(
        record.get(
            "normalized_country",
            record.get("country", "")
        )
    )


def build_candidate_index(
    reference_records: Iterable[Dict[str, Any]]
) -> Dict[str, Dict[str, Set[int]]]:
    """
    Build an inverted index for reference records.

    Index structure:

    {
        "country": {
            "us": {0, 1, 5},
            "in": {2, 3}
        },
        "name_token": {
            "amazon": {0, 4},
            "market": {1, 5}
        }
    }

    The integer values represent positions in reference_records.
    """

    country_index = defaultdict(set)
    name_token_index = defaultdict(set)

    for record_index, record in enumerate(reference_records):
        country = get_country(record)

        if country:
            country_index[country].add(record_index)

        name_tokens = get_name_tokens(record)

        for token in name_tokens:
            name_token_index[token].add(record_index)

    return {
        "country": dict(country_index),
        "name_token": dict(name_token_index)
    }


def generate_candidates(
    query_record: Dict[str, Any],
    reference_records: List[Dict[str, Any]],
    candidate_index: Dict[str, Dict[str, Set[int]]],
    max_candidates: int = 200
) -> List[Dict[str, Any]]:
    """
    Generate possible reference candidates for one query record.

    Blocking rules:
        1. Candidates must share the same country when country
           information is available.
        2. Candidates should share at least one meaningful
           business name token.
        3. Multiple matching name tokens are used to prioritize
           candidates.

    Args:
        query_record: Source2 or Source3 record.
        reference_records: Source1 records.
        candidate_index: Index built from Source1 records.
        max_candidates: Maximum number of returned candidates.

    Returns:
        List of candidate reference records.
    """

    query_country = get_country(query_record)
    query_tokens = get_name_tokens(query_record)

    country_index = candidate_index["country"]
    name_token_index = candidate_index["name_token"]

    # Restrict candidates by country where possible.
    if query_country and query_country in country_index:
        allowed_records = country_index[query_country]
    else:
        allowed_records = set(range(len(reference_records)))

    # Count shared name tokens for each candidate.
    candidate_token_counts = defaultdict(int)

    for token in query_tokens:
        matching_records = name_token_index.get(token, set())

        for record_index in matching_records:
            if record_index in allowed_records:
                candidate_token_counts[record_index] += 1

    # Sort candidates by the number of shared name tokens.
    ranked_candidates = sorted(
        candidate_token_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    selected_indices = [
        record_index
        for record_index, _ in ranked_candidates[:max_candidates]
    ]

    return [
        reference_records[index]
        for index in selected_indices
    ]