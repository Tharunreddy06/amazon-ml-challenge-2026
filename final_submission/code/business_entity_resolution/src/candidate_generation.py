
"""
Candidate generation for business entity resolution.

This module generates possible matching records using blocking
rather than comparing every record against every other record.

Blocking strategies:
    1. Country blocking
    2. Business name token blocking
    3. Address token blocking
    4. Exact normalized name matching
    5. Exact normalized address matching
    6. Combined candidate scoring
"""

from collections import defaultdict
from typing import Dict, List, Any, Iterable, Set
import re


def safe_string(value: Any) -> str:
    """Convert a value to a safe lowercase string."""

    if value is None:
        return ""

    return str(value).strip().lower()


def get_normalized_name(record: Dict[str, Any]) -> str:
    """Return the normalized business name."""

    return safe_string(
        record.get(
            "normalized_name",
            record.get("business_name", "")
        )
    )


def get_normalized_address(record: Dict[str, Any]) -> str:
    """Return the normalized business address."""

    return safe_string(
        record.get(
            "normalized_address",
            record.get("business_address", "")
        )
    )


def get_name_tokens(record: Dict[str, Any]) -> Set[str]:
    """
    Extract meaningful tokens from a business name.

    Tokens with fewer than 3 characters are ignored.
    """

    name = get_normalized_name(record)

    tokens = {
        token
        for token in name.split()
        if len(token) >= 3
    }

    return tokens


def get_address_tokens(record: Dict[str, Any]) -> Set[str]:
    """
    Extract meaningful tokens from a business address.

    Numeric-only tokens and very short tokens are ignored.
    """

    address = get_normalized_address(record)

    raw_tokens = re.findall(
        r"[a-zA-Z0-9]+",
        address
    )

    tokens = {
        token
        for token in raw_tokens
        if len(token) >= 3 and not token.isdigit()
    }

    return tokens


def get_domain_style_tokens(
    record: Dict[str, Any]
) -> Set[str]:
    """
    Extract tokens useful for matching website-style names.

    Example:
        maurewilliamscolombier.com
        becomes:
        maurewilliamscolombier

    This function also creates compact combinations
    from regular business-name tokens.

    Example:
        maure williams colombier
        becomes:
        maurewilliamscolombier
    """

    name = get_normalized_name(record)

    if not name:
        return set()

    # Remove common domain extensions.
    name_without_extension = re.sub(
        r"\.(com|in|org|net|co|biz)$",
        "",
        name
    )

    # Remove separators commonly used in domains.
    compact_name = re.sub(
        r"[^a-z0-9]",
        "",
        name_without_extension
    )

    tokens = set()

    if len(compact_name) >= 5:
        tokens.add(compact_name)

    name_tokens = [
        token
        for token in name.split()
        if len(token) >= 3
    ]

    if len(name_tokens) >= 2:
        combined_tokens = "".join(name_tokens)

        if len(combined_tokens) >= 5:
            tokens.add(combined_tokens)

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
    Build inverted indexes for reference records.

    Indexes:
        country
        name_token
        address_token
        domain_token
        exact_name
        exact_address
    """

    country_index = defaultdict(set)
    name_token_index = defaultdict(set)
    address_token_index = defaultdict(set)
    domain_token_index = defaultdict(set)
    exact_name_index = defaultdict(set)
    exact_address_index = defaultdict(set)

    for record_index, record in enumerate(reference_records):

        country = get_country(record)

        if country:
            country_index[country].add(record_index)

        normalized_name = get_normalized_name(record)

        if normalized_name:
            exact_name_index[normalized_name].add(
                record_index
            )

        normalized_address = get_normalized_address(record)

        if normalized_address:
            exact_address_index[normalized_address].add(
                record_index
            )

        name_tokens = get_name_tokens(record)

        for token in name_tokens:
            name_token_index[token].add(record_index)

        address_tokens = get_address_tokens(record)

        for token in address_tokens:
            address_token_index[token].add(record_index)

        domain_tokens = get_domain_style_tokens(record)

        for token in domain_tokens:
            domain_token_index[token].add(record_index)

    return {
        "country": dict(country_index),
        "name_token": dict(name_token_index),
        "address_token": dict(address_token_index),
        "domain_token": dict(domain_token_index),
        "exact_name": dict(exact_name_index),
        "exact_address": dict(exact_address_index),
    }


def generate_candidates(
    query_record: Dict[str, Any],
    reference_records: List[Dict[str, Any]],
    candidate_index: Dict[str, Dict[str, Set[int]]],
    max_candidates: int = 200
) -> List[Dict[str, Any]]:
    """
    Generate possible reference candidates for one query record.

    Candidate selection uses:
        1. Country filtering
        2. Name-token matches
        3. Address-token matches
        4. Domain-style matches
        5. Exact name matches
        6. Exact address matches

    Candidates are ranked using combined blocking scores.
    """

    query_country = get_country(query_record)

    query_name = get_normalized_name(query_record)
    query_address = get_normalized_address(query_record)

    query_name_tokens = get_name_tokens(query_record)
    query_address_tokens = get_address_tokens(query_record)
    query_domain_tokens = get_domain_style_tokens(query_record)

    country_index = candidate_index["country"]
    name_token_index = candidate_index["name_token"]
    address_token_index = candidate_index["address_token"]
    domain_token_index = candidate_index["domain_token"]
    exact_name_index = candidate_index["exact_name"]
    exact_address_index = candidate_index["exact_address"]

    # --------------------------------------------------
    # COUNTRY FILTERING
    # --------------------------------------------------

    if query_country and query_country in country_index:
        allowed_records = country_index[query_country]
    else:
        allowed_records = set(
            range(len(reference_records))
        )

    # --------------------------------------------------
    # COLLECT CANDIDATE SCORES
    # --------------------------------------------------

    candidate_scores = defaultdict(float)

    # Name token matches receive a strong score.
    for token in query_name_tokens:

        matching_records = name_token_index.get(
            token,
            set()
        )

        for record_index in matching_records:

            if record_index in allowed_records:
                candidate_scores[record_index] += 3.0

    # Address token matches provide an additional signal.
    for token in query_address_tokens:

        matching_records = address_token_index.get(
            token,
            set()
        )

        for record_index in matching_records:

            if record_index in allowed_records:
                candidate_scores[record_index] += 1.5

    # Domain-style matches help with website-like names.
    for token in query_domain_tokens:

        matching_records = domain_token_index.get(
            token,
            set()
        )

        for record_index in matching_records:

            if record_index in allowed_records:
                candidate_scores[record_index] += 4.0

    # --------------------------------------------------
    # EXACT MATCH BOOSTS
    # --------------------------------------------------

    if query_name:

        matching_records = exact_name_index.get(
            query_name,
            set()
        )

        for record_index in matching_records:

            if record_index in allowed_records:
                candidate_scores[record_index] += 10.0

    if query_address:

        matching_records = exact_address_index.get(
            query_address,
            set()
        )

        for record_index in matching_records:

            if record_index in allowed_records:
                candidate_scores[record_index] += 8.0

    # --------------------------------------------------
    # RANK CANDIDATES
    # --------------------------------------------------

    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    selected_indices = [
        record_index
        for record_index, score in ranked_candidates[
            :max_candidates
        ]
    ]

    return [
        reference_records[index]
        for index in selected_indices
    ]