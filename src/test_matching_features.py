
from matching_features import generate_matching_features


def main():
    record_a = {
        "normalized_name": "abc pvt ltd",
        "normalized_address": "123 main st chicago il",
        "normalized_country": "us"
    }

    record_b = {
        "normalized_name": "abc private limited",
        "normalized_address": "123 main street chicago il",
        "normalized_country": "us"
    }

    features = generate_matching_features(record_a, record_b)

    print("\nMATCHING FEATURES")
    print("-" * 40)

    for feature_name, feature_value in features.items():
        print(f"{feature_name}: {feature_value}")


if __name__ == "__main__":
    main()
    