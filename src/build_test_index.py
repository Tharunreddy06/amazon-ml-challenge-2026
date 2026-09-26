import csv
import os
import re
import sqlite3
import sys
from pathlib import Path

# Allow imports from src/
sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import preprocess_record


INPUT_FILE = Path("data/test_source1.tsv")
OUTPUT_DIR = Path("data")
DB_FILE = OUTPUT_DIR / "test_source1_index.db"

BATCH_SIZE = 10000
MAX_RECORDS = None


def get_name_tokens(name):
    """Extract useful tokens from a normalized business name."""
    if not name:
        return []

    tokens = re.findall(r"[a-z0-9]+", name.lower())

    # Ignore very short tokens
    return sorted(set(token for token in tokens if len(token) >= 3))


def get_address_tokens(address):
    """Extract useful tokens from a normalized address."""
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
    """
    Extract domain-style matching tokens.

    Example:
        brightseafood.com
        -> brightseafood
        -> bright seafood
    """
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


def create_tables(conn):
    cursor = conn.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS entities;
        DROP TABLE IF EXISTS name_tokens;
        DROP TABLE IF EXISTS address_tokens;
        DROP TABLE IF EXISTS domain_tokens;

        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            normalized_name TEXT,
            normalized_address TEXT,
            normalized_country TEXT
        );

        CREATE TABLE name_tokens (
            token TEXT,
            entity_id TEXT
        );

        CREATE TABLE address_tokens (
            token TEXT,
            entity_id TEXT
        );

        CREATE TABLE domain_tokens (
            token TEXT,
            entity_id TEXT
        );
        """
    )

    conn.commit()


def create_indexes(conn):
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE INDEX idx_name_tokens
        ON name_tokens(token);

        CREATE INDEX idx_address_tokens
        ON address_tokens(token);

        CREATE INDEX idx_domain_tokens
        ON domain_tokens(token);

        CREATE INDEX idx_entities_country
        ON entities(normalized_country);

        CREATE INDEX idx_entities_name
        ON entities(normalized_name);

        CREATE INDEX idx_entities_address
        ON entities(normalized_address);
        """
    )

    conn.commit()


def process_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DB_FILE.exists():
        print(f"Removing existing database: {DB_FILE}")
        DB_FILE.unlink()

    print("Creating SQLite database...")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {DB_FILE}")
    if MAX_RECORDS is None:
        print("Maximum records: ALL")
    else:
        print(f"Maximum records: {MAX_RECORDS:,}")

    conn = sqlite3.connect(DB_FILE)

    # Improve SQLite write performance during index creation.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")

    create_tables(conn)

    entity_rows = []
    name_rows = []
    address_rows = []
    domain_rows = []

    processed = 0

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:

            

            if MAX_RECORDS is not None and processed >= MAX_RECORDS:
                break

            processed += 1

            entity_id = row.get("entity_id", "")
            business_name = row.get("business_name", "")
            business_address = row.get("business_address", "")
            country = row.get("country", "")

            normalized = preprocess_record(
                {
                    "entity_id": entity_id,
                    "business_name": business_name,
                    "business_address": business_address,
                    "country": country,
                }
            )

            normalized_name = normalized.get(
                "normalized_name", ""
            )

            normalized_address = normalized.get(
                "normalized_address", ""
            )

            normalized_country = normalized.get(
                "normalized_country", ""
            )

            entity_rows.append(
                (
                    entity_id,
                    business_name,
                    business_address,
                    country,
                    normalized_name,
                    normalized_address,
                    normalized_country,
                )
            )

            for token in get_name_tokens(normalized_name):
                name_rows.append((token, entity_id))

            for token in get_address_tokens(normalized_address):
                address_rows.append((token, entity_id))

            for token in get_domain_tokens(normalized_name):
                domain_rows.append((token, entity_id))

            if processed % BATCH_SIZE == 0:

                conn.executemany(
                    """
                    INSERT INTO entities
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    entity_rows,
                )

                conn.executemany(
                    """
                    INSERT INTO name_tokens
                    VALUES (?, ?)
                    """,
                    name_rows,
                )

                conn.executemany(
                    """
                    INSERT INTO address_tokens
                    VALUES (?, ?)
                    """,
                    address_rows,
                )

                conn.executemany(
                    """
                    INSERT INTO domain_tokens
                    VALUES (?, ?)
                    """,
                    domain_rows,
                )

                conn.commit()

                entity_rows.clear()
                name_rows.clear()
                address_rows.clear()
                domain_rows.clear()

                print(
                    f"Processed {processed:,} records..."
                )

    # Insert remaining rows
    if entity_rows:
        conn.executemany(
            """
            INSERT INTO entities
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            entity_rows,
        )

    if name_rows:
        conn.executemany(
            """
            INSERT INTO name_tokens
            VALUES (?, ?)
            """,
            name_rows,
        )

    if address_rows:
        conn.executemany(
            """
            INSERT INTO address_tokens
            VALUES (?, ?)
            """,
            address_rows,
        )

    if domain_rows:
        conn.executemany(
            """
            INSERT INTO domain_tokens
            VALUES (?, ?)
            """,
            domain_rows,
        )

    conn.commit()

    print("\nCreating indexes...")
    create_indexes(conn)

    # Remove WAL files after closing.
    conn.close()

    print("\n===================================")
    print("TEST SOURCE1 INDEX COMPLETE")
    print("===================================")
    print(f"Records processed: {processed:,}")
    print(f"Database: {DB_FILE}")

    if DB_FILE.exists():
        size_mb = DB_FILE.stat().st_size / (1024 * 1024)
        print(f"Database size: {size_mb:.2f} MB")


if __name__ == "__main__":
    process_file()
