"""Build the non-destructive V7 content-hash and decode audit manifest."""

from __future__ import annotations

import argparse
import json

from v7_data import assert_no_validation_hash_overlap, build_manifest, write_manifest


EXPECTED_AIC_V7 = {
    "raw_files": 148695,
    "classes": 750,
    "duplicate_groups": 2128,
    "same_class_extra_files": 251,
    "cross_class_groups": 1877,
    "cross_class_files": 3884,
    "clean_unique_files": 144560,
    "ambiguous_representatives": 1877,
    "unreadable_files": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--verify-expected-aic-v7",
        action="store_true",
        help="Fail unless the official dataset reproduces the locked V7 audit counts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest(args.train_dir, args.seed)
    assert_no_validation_hash_overlap(payload)
    if args.verify_expected_aic_v7:
        summary = payload["summary"]
        mismatches = {
            key: {"expected": expected, "observed": summary.get(key)}
            for key, expected in EXPECTED_AIC_V7.items()
            if summary.get(key) != expected
        }
        if summary.get("classes_without_clean_rows") != []:
            mismatches["classes_without_clean_rows"] = {
                "expected": [],
                "observed": summary.get("classes_without_clean_rows"),
            }
        if mismatches:
            raise ValueError(f"official AIC V7 audit counts do not match: {mismatches}")
    write_manifest(payload, args.output)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"dataset_signature={payload['dataset_signature']}", flush=True)
    print(f"manifest={args.output}", flush=True)


if __name__ == "__main__":
    main()
