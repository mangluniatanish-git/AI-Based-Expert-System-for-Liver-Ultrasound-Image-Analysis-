import os
import argparse

import numpy as np

from pipeline import process_image_path
from feature_extraction import FEATURE_NAMES

CLASSES = ["Benign", "Malignant", "Normal"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def build_dataset(dataset_root):
    X, y = [], []
    for cls in CLASSES:
        img_dir = os.path.join(dataset_root, cls, cls, "image")
        if not os.path.isdir(img_dir):
            print(f"  [SKIP] {cls}: folder not found at {img_dir}")
            continue

        filenames = sorted(f for f in os.listdir(img_dir)
                            if f.lower().endswith(IMAGE_EXTENSIONS))
        print(f"Processing {cls} ({len(filenames)} images)...")

        n_ok, n_fail = 0, 0
        for fname in filenames:
            img_path = os.path.join(img_dir, fname)
            try:
                result = process_image_path(img_path)
            except Exception:
                n_fail += 1
                continue
            X.append(result["feature_vector"])
            y.append(cls)
            n_ok += 1

        print(f"  -> {n_ok} used, {n_fail} skipped (unreadable image)")

    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Build the cached training feature dataset from the "
                    "labeled Benign/Malignant/Normal dataset, using the "
                    "same automatic pipeline predict.py uses."
    )
    default_dataset = r"E:\AI_IVP_Proj\Dataset\7272660" if os.path.exists(r"E:\AI_IVP_Proj\Dataset\7272660") else None
    parser.add_argument("--dataset", default=default_dataset, required=default_dataset is None,
                         help="Path to the dataset root folder (containing "
                              "Benign/, Malignant/, Normal/ subfolders).")
    parser.add_argument("--output", default="features_dataset.npz",
                         help="Where to save the cached feature matrix. "
                              "Default: ./features_dataset.npz")
    args = parser.parse_args()

    print("Building training feature dataset (automatic pipeline)...")
    print(f"IVP feature columns ({len(FEATURE_NAMES)}): {FEATURE_NAMES}\n")

    X, y = build_dataset(args.dataset)

    print(f"\nTotal samples: {len(y)}")
    for cls in CLASSES:
        print(f"  {cls}: {int(np.sum(y == cls))}")

    np.savez_compressed(args.output, X=X, y=y,
                         feature_names=np.array(FEATURE_NAMES))
    print(f"\nSaved feature dataset to: {args.output}")


if __name__ == "__main__":
    main()
