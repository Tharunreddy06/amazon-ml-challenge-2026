
from candidate_generation import (
    build_candidate_index,
    generate_candidates
)


def main():
    reference_records = [
        {
            "entity_id": "S1-001",
            "normalized_name": "abc pvt ltd",
            "normalized_address": "123 main st chicago il",
            "normalized_country": "us"
        },
        {
            "entity_id": "S1-002",
            "normalized_name": "xyz technologies",
            "normalized_address": "45 market road mumbai",
            "normalized_country": "in"
        },
        {
            "entity_id": "S1-003",
            "normalized_name": "abc business center",
            "normalized_address": "500 lake road chicago il",
            "normalized_country": "us"
        }
    ]

    query_record = {
        "normalized_name": "abc limited",
        "normalized_address": "123 main street chicago il",
        "normalized_country": "us"
    }

    candidate_index = build_candidate_index(reference_records)

    candidates = generate_candidates(
        query_record=query_record,
        reference_records=reference_records,
        candidate_index=candidate_index,
        max_candidates=10
    )

    print("\nCANDIDATE GENERATION TEST")
    print("-" * 40)

    print(f"Number of candidates: {len(candidates)}")

    for candidate in candidates:
        print(candidate)


if __name__ == "__main__":
    main()