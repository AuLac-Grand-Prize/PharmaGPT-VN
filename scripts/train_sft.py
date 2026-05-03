"""Supervised fine-tuning on dược sĩ Q&A pairs."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sft_pairs.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()
    print(f"SFT data={args.data} epochs={args.epochs} lora={args.lora} (TODO)")


if __name__ == "__main__":
    main()
