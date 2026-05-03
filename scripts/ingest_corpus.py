"""Ingest VN pharmacy corpus into Qdrant: chunk → embed (bge-m3) → upsert."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(f"Ingest corpus from {args.source} (TODO)")


if __name__ == "__main__":
    main()
