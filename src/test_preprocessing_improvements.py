
from preprocessing import (
    normalize_business_name,
    normalize_business_address,
)


def main():
    print("BUSINESS NAME TESTS")
    print("=" * 60)

    names = [
        "ABC Private Limited",
        "Prime Money Pvt Ltd",
        "Example Incorporated",
        "Sample LLC",
        "ABC Business Center",
        "ABC Group",
    ]

    for name in names:
        print(f"Original: {name}")
        print(f"Normalized: {normalize_business_name(name)}")
        print(
            "Without noise: "
            f"{normalize_business_name(name, remove_noise=True)}"
        )
        print("-" * 60)

    print("\nADDRESS TESTS")
    print("=" * 60)

    addresses = [
        "123 Illinois Road, Chicago",
        "45 Virginia Avenue, Richmond",
        "10 Main Street, New York",
        "200 North Carolina Boulevard",
    ]

    for address in addresses:
        print(f"Original: {address}")
        print(f"Normalized: {normalize_business_address(address)}")
        print("-" * 60)


if __name__ == "__main__":
    main()