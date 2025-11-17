import argparse
import json
import os
from glob import glob

import pandas as pd
from tqdm import tqdm


LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Other",
    "Support Devices",
    "No Finding",
]

FRONTAL_VIEWS = {"PA", "AP"}


def build_caption_from_row(row, include_uncertain=True, drop_no_finding_if_any=True):
    """
    Build a caption as a list of label strings from a NegBio CSV row.

    Values:
      1.0  -> positive
      -1.0 -> uncertain
      0.0  -> explicitly negative
      NaN  -> not mentioned
    """
    positives = []
    uncertain = []
    no_finding = []

    for lbl in LABEL_COLUMNS:
        val = row.get(lbl, float("nan"))
        if pd.isna(val):
            continue

        if lbl == "No Finding":
            if val == 1.0:
                no_finding.append(lbl)
            continue

        if val == 1.0:
            positives.append(lbl)
        elif val == -1.0 and include_uncertain:
            uncertain.append(f"uncertain {lbl}")

    caption_labels = positives + uncertain

    # Handle "No Finding"
    if no_finding:
        if not caption_labels or not drop_no_finding_if_any:
            caption_labels.append("No Finding")

    return caption_labels


def load_metadata(metadata_csv: str):
    """
    Load metadata CSV and return a mapping:
      dicom_id -> ViewPosition (e.g. 'PA', 'AP', 'LATERAL', ...)
    """
    print(f"Loading metadata from: {metadata_csv}")
    mdf = pd.read_csv(metadata_csv)
    if "dicom_id" not in mdf.columns or "ViewPosition" not in mdf.columns:
        raise ValueError("Metadata CSV must contain 'dicom_id' and 'ViewPosition' columns")

    meta = {}
    for _, row in mdf.iterrows():
        did = str(row["dicom_id"])
        view = row["ViewPosition"] if isinstance(row["ViewPosition"], str) else ""
        meta[did] = view
    print(f"Loaded metadata for {len(meta)} dicom_ids")
    return meta


def find_study_dir(base_dir, subject_id, study_id):
    """
    Given integer-like subject_id and study_id, return path to study directory
    in MIMIC-CXR-JPG layout.
    """
    subj_name = f"p{int(subject_id)}"
    study_name = f"s{int(study_id)}"

    patterns = [
        os.path.join(base_dir, "p*", subj_name, study_name),  # p10/p10000032/s50414267
    ]

    for pat in patterns:
        matches = glob(pat)
        if matches:
            return matches[0]

    return None


def collect_pairs_from_negbio(
    base_dir: str,
    negbio_csv: str,
    metadata_csv: str,
    relative_to: str = None,
    max_studies: int = None,
    first_image_only: bool = True,
):
    base_dir = os.path.abspath(base_dir)
    if relative_to is not None:
        relative_to = os.path.abspath(relative_to)

    # Load metadata and NegBio labels
    metadata = load_metadata(metadata_csv)

    print(f"Loading NegBio CSV from: {negbio_csv}")
    df = pd.read_csv(negbio_csv)

    assert "subject_id" in df.columns and "study_id" in df.columns, \
        "NegBio CSV must contain 'subject_id' and 'study_id' columns"

    df = df.sort_values(["subject_id", "study_id"]).reset_index(drop=True)

    if max_studies is not None:
        df = df.head(max_studies)

    pairs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Studies"):
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])

        caption_labels = build_caption_from_row(row)
        if not caption_labels:
            # no useful labels -> skip
            continue

        study_dir = find_study_dir(base_dir, subject_id, study_id)
        if study_dir is None or not os.path.isdir(study_dir):
            # study not present in this subset
            continue

        # All jpg/jpeg images in this study dir
        image_paths = sorted(
            glob(os.path.join(study_dir, "*.jpg"))
            + glob(os.path.join(study_dir, "*.jpeg"))
        )
        if not image_paths:
            continue

        # Filter to frontal views using metadata
        frontal_images = []
        for img_path in image_paths:
            dicom_id = os.path.splitext(os.path.basename(img_path))[0]
            view = metadata.get(dicom_id, "")
            if view in FRONTAL_VIEWS:
                frontal_images.append(img_path)

        if not frontal_images:
            # No frontal image in this study
            continue

        if first_image_only:
            frontal_images = frontal_images[:1]

        for img_path in frontal_images:
            if relative_to is not None:
                img_stored = os.path.relpath(img_path, relative_to)
            else:
                img_stored = img_path

            pairs.append(
                {
                    "image": img_stored,
                    "caption": caption_labels,  # list of label strings
                }
            )

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare image-text pairs from MIMIC-CXR-JPG using NegBio labels + metadata (frontal only)."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Root directory of MIMIC-CXR extracted files (e.g. .../extracted/2.1.0/files)",
    )
    parser.add_argument(
        "--negbio-csv",
        type=str,
        required=True,
        help="Path to mimic-cxr-2.0.0-negbio.csv (or .csv.gz)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
        help="Path to mimic-cxr-2.0.0-metadata.csv (or .csv.gz)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--relative-to",
        type=str,
        default=None,
        help="If set, store image paths relative to this directory.",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Limit number of studies (rows in NegBio CSV) for quick testing.",
    )
    parser.add_argument(
        "--first-image-only",
        action="store_true",
        help="Only use the first frontal image per study.",
    )

    args = parser.parse_args()

    pairs = collect_pairs_from_negbio(
        base_dir=args.base_dir,
        negbio_csv=args.negbio_csv,
        metadata_csv=args.metadata_csv,
        relative_to=args.relative_to,
        max_studies=args.max_studies,
        first_image_only=args.first_image_only,
    )

    print(f"Collected {len(pairs)} image-text pairs.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved pairs to {args.output_json}")


if __name__ == "__main__":
    main()