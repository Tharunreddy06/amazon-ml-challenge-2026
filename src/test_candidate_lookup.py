import re
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import preprocess_record


DB_FILE = Path("data/test_source1_index.db")


def get_name_tokens(name):
    if not name:
        return []

    tokens = re.findall(r"[a-z0-9]+", name.lower())

    return sorted(
        set(token for token in tokens if len(token) >= 3)
    )


def get_address_tokens(address):
    if not address:
        return []

    tokens = re.findall(r"[a-z0-9]+", address.lower())

    return sorted(
        set(
            token
            for token in tokens
            if len(token) >= 3 and not token.isdigit()
        )
    )


def get_domain_tokens(name):
    if not name:
        return []

    value = name.lower()

    for extension in [
        ".com",
        ".in",
        ".org",
        ".net",
        ".co",
        ".biz",
    ]:
        value = value.replace(extension, " ")

    tokens = re.findall(r"[a-z0-9]+", value)

    result = set()

    for token in tokens:
        if len(token) >= 3:
            result.add(token)

    compact = "".join(tokens)

    if len(compact) >= 3:
        result.add(compact)

    return sorted(result)


def generate_candidates(conn, query_record, max_candidates=200):

    normalized = preprocess_record(query_record)

    name = normalized.get("normalized_name", "")
    address = normalized.get("normalized_address", "")
    country = normalized.get("normalized_country", "")

    name_tokens = get_name_tokens(name)
    address_tokens = get_address_tokens(address)
    domain_tokens = get_domain_tokens(name)

    scores = {}

    # --------------------------------------------------
    # Name token candidates
    # --------------------------------------------------

    for token in name_tokens:

        rows = conn.execute(
            """
            SELECT entity_id
            FROM name_tokens
            WHERE token = ?
            """,
            (token,),
        ).fetchall()

        for (entity_id,) in rows:
            scores[entity_id] = scores.get(entity_id, 0) + 3

    # --------------------------------------------------
    # Address token candidates
    # --------------------------------------------------

    for token in address_tokens:

        rows = conn.execute(
            """
            SELECT entity_id
            FROM address_tokens
            WHERE token = ?
            """,
            (token,),
        ).fetchall()

        for (entity_id,) in rows:
            scores[entity_id] = scores.get(entity_id, 0) + 1.5

    # --------------------------------------------------
    # Domain candidates
    # --------------------------------------------------

    for token in domain_tokens:

        rows = conn.execute(
            """
            SELECT entity_id
            FROM domain_tokens
            WHERE token = ?
            """,
            (token,),
        ).fetchall()

        for (entity_id,) in rows:
            scores[entity_id] = scores.get(entity_id, 0) + 4

    # --------------------------------------------------
    # Exact name
    # --------------------------------------------------

    if name:

        rows = conn.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE normalized_name = ?
            """,
            (name,),
        ).fetchall()

        for (entity_id,) in rows:
            scores[entity_id] = scores.get(entity_id, 0) + 10

    # --------------------------------------------------
    # Exact address
    # --------------------------------------------------

    if address:

        rows = conn.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE normalized_address = ?
            """,
            (address,),
        ).fetchall()

        for (entity_id,) in rows:
            scores[entity_id] = scores.get(entity_id, 0) + 8

    # --------------------------------------------------
    # Retrieve candidate records
    # --------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    ranked = ranked[:max_candidates]

    if not ranked:
        return []

    candidate_ids = [entity_id for entity_id, _ in ranked]

    placeholders = ",".join(["?"] * len(candidate_ids))

    rows = conn.execute(
        f"""
        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            normalized_name,
            normalized_address,
            normalized_country
        FROM entities
        WHERE entity_id IN ({placeholders})
        """,
        candidate_ids,
    ).fetchall()

    records = {
        row[0]: row
        for row in rows
    }

    results = []

    for entity_id, score in ranked:

        row = records.get(entity_id)

        if row is None:
            continue

        results.append(
            {
                "entity_id": row[0],
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3],
                "normalized_name": row[4],
                "normalized_address": row[5],
                "normalized_country": row[6],
                "candidate_score": score,
            }
        )

    return results


def main():

    conn = sqlite3.connect(DB_FILE)

    test_queries = [
        {
            "entity_id": "S2-192345572",
            "business_name": "Brahma Infosoft",
            "business_address": "COIMATORE COLONY, HUNSUR TQ MYSORE DIST., Karnataka",
            "country": "India",
        },
        {
            "entity_id": "S2-566025912",
            "business_name": "Marina Ecole France Sarl",
            "business_address": "63 R. DE DIEPPE, LILLE, Hauts-de-France",
            "country": "France",
        },
        {
            "entity_id": "S3-374810425",
            "business_name": "Shri Sai Infratech Co",
            "business_address": "3/115, East Delhi, DL",
            "country": "India",
        },
    ]

    for query in test_queries:

        print("\n" + "=" * 80)

        print(
            f"QUERY: {query['entity_id']}"
        )

        print(
            f"NAME: {query['business_name']}"
        )

        print(
            f"COUNTRY: {query['country']}"
        )

        candidates = generate_candidates(
            conn,
            query,
            max_candidates=10,
        )

        print(
            f"\nCandidates found: {len(candidates)}"
        )

        for i, candidate in enumerate(candidates, start=1):

            print(
                f"\n{i}. "
                f"{candidate['entity_id']} "
                f"(score={candidate['candidate_score']})"
            )

            print(
                f"   Name: {candidate['business_name']}"
            )

            print(
                f"   Address: {candidate['business_address']}"
            )

            print(
                f"   Country: {candidate['country']}"
            )

    conn.close()


if __name__ == "__main__":
    main()
    