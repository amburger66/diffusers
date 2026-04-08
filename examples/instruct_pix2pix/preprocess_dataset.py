#!/usr/bin/env python3
import argparse
import os
from datasets import load_dataset, load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify_images", action="store_true")
    args = parser.parse_args()

    data_files = {
        "train": os.path.join(args.train_data_dir, "train", "**"),
        "test": os.path.join(args.train_data_dir, "test", "**"),
    }

    print("Loading imagefolder dataset...", flush=True)
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    train_ds = dataset["train"]
    print(f"Loaded train split with {len(train_ds)} rows", flush=True)
    print(f"Columns: {train_ds.column_names}", flush=True)

    if args.max_train_samples is not None:
        train_ds = train_ds.shuffle(seed=args.seed).select(
            range(args.max_train_samples)
        )
        print(f"Selected first {len(train_ds)} shuffled rows", flush=True)

    if args.verify_images:
        print("Verifying image decode for all rows...", flush=True)
        bad_rows = []

        image_columns = [
            col
            for col in train_ds.column_names
            if getattr(train_ds.features.get(col), "__class__", None).__name__
            == "Image"
        ]
        print(f"Image columns detected: {image_columns}", flush=True)

        for i in range(len(train_ds)):
            try:
                row = train_ds[i]
                for col in image_columns:
                    img = row[col]
                    img.load()
            except Exception as e:
                bad_rows.append((i, str(e)))
                if len(bad_rows) <= 10:
                    print(f"Bad row {i}: {e}", flush=True)

        if bad_rows:
            raise RuntimeError(
                f"Found {len(bad_rows)} bad rows during decode verification. "
                f"First few: {bad_rows[:10]}"
            )
        print("Image verification passed", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving dataset to {args.output_dir} ...", flush=True)
    train_ds.save_to_disk(args.output_dir)
    print("Done", flush=True)

    # Sanity check reload
    reloaded = load_from_disk(args.output_dir)
    print(f"Reloaded dataset with {len(reloaded)} rows", flush=True)
    print(f"Reloaded columns: {reloaded.column_names}", flush=True)


if __name__ == "__main__":
    main()
