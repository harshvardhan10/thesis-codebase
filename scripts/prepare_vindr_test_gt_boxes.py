import argparse
import logging
import os

import numpy as np
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def scale_boxes_to_256(ann_df, meta_df):
    """
    Join annotations with image dimensions and rescale boxes
    from (width, height) -> (256, 256).
    """
    logging.info("Merging annotations with meta information (image dimensions).")

    merged = ann_df.merge(meta_df, on="image_id", how="left", validate="many_to_one")

    if merged["dim1"].isna().any() or merged["dim0"].isna().any():
        missing_ids = merged.loc[
            merged["dim1"].isna() | merged["dim0"].isna(), "image_id"
        ].unique()
        raise ValueError(
            f"Missing width/height for some image_ids in test_meta.csv. "
            f"Examples: {missing_ids[:10]}"
        )

    # Avoid division by zero
    if (merged["dim1"] <= 0).any() or (merged["dim0"] <= 0).any():
        bad = merged[(merged["dim1"] <= 0) | (merged["dim0"] <= 0)]
        raise ValueError(
            f"Found non-positive width/height for some rows, e.g.:\n{bad.head()}"
        )

    target_size = 256.0

    logging.info("Scaling bounding boxes to 256x256 coordinate space.")
    merged["scale_x"] = target_size / merged["dim1"]
    merged["scale_y"] = target_size / merged["dim0"]

    for coord in ["x_min", "x_max"]:
        merged[f"{coord}_256"] = merged[coord] * merged["scale_x"]

    for coord in ["y_min", "y_max"]:
        merged[f"{coord}_256"] = merged[coord] * merged["scale_y"]

    # Clip to [0, 256] (just to be safe)
    for coord in ["x_min_256", "x_max_256", "y_min_256", "y_max_256"]:
        merged[coord] = merged[coord].clip(0.0, target_size)

    # Ensure min <= max after potential rounding/clipping
    merged["x_min_256"], merged["x_max_256"] = np.minimum(
        merged["x_min_256"], merged["x_max_256"]
    ), np.maximum(merged["x_min_256"], merged["x_max_256"])
    merged["y_min_256"], merged["y_max_256"] = np.minimum(
        merged["y_min_256"], merged["y_max_256"]
    ), np.maximum(merged["y_min_256"], merged["y_max_256"])

    out_cols = [
        "image_id",
        "class_name",
        "x_min_256",
        "y_min_256",
        "x_max_256",
        "y_max_256",
    ]

    out_df = merged[out_cols].copy()
    out_df = out_df.rename(
        columns={
            "class_name": "label_name",
            "x_min_256": "x_min",
            "y_min_256": "y_min",
            "x_max_256": "x_max",
            "y_max_256": "y_max",
        }
    )

    logging.info(f"Total GT boxes: {len(out_df)}")
    logging.info(f"Unique images in GT: {out_df['image_id'].nunique()}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Prepare VinDr GT boxes at 256x256.")
    parser.add_argument(
        "--annotations_test",
        type=str,
        required=True,
        help="Path to annotations_test.csv",
    )
    parser.add_argument(
        "--test_meta",
        type=str,
        required=True,
        help="Path to test_meta.csv (with image_id,width,height)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="gt_boxes_256.csv",
        help="Output CSV for resized GT boxes",
    )
    args = parser.parse_args()
    setup_logging()

    logging.info("Loading input CSVs...")
    ann_df = pd.read_csv(args.annotations_test)
    meta_df = pd.read_csv(args.test_meta)

    gt_256_df = scale_boxes_to_256(ann_df, meta_df)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    gt_256_df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved GT boxes to {args.output_csv}")


if __name__ == "__main__":
    main()
