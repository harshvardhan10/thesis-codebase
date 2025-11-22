#!/usr/bin/env python

import argparse
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from ALBEF.models.model_pretrain import ALBEF
from ALBEF.models.tokenization_bert import BertTokenizer
import yaml

from sklearn.metrics import roc_auc_score, f1_score


def infer_label_columns(df: pd.DataFrame) -> (pd.DataFrame, list):
    """
    Try to infer label columns from a VinDr image_labels_*.csv.

    Supports two formats:
      1. Wide format: columns [image_id, rad_id, label1, label2, ...]
      2. A column named 'labels' containing a string of 0/1 separated by spaces or commas.

    Returns: (labels_array (N,L), label_names list)
    """
    cols = df.columns.tolist()

    # Case 1: 'labels' column with vector as string
    if "labels" in cols and df["labels"].dtype == object:
        # You MUST provide the label names in the correct order here:
        # Adjust this list to the exact VinDr label order in your dataset.
        VINDR_LABELS = [
            "Aortic enlargement",
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung Opacity",
            "Nodule/Mass",
            "Pleural effusion",
            "Pleural thickening",
            "Pneumothorax",
            "Pulmonary fibrosis",
            "Other lesion",
            "Fracture",
            "Lung cyst",
            "Mediastinal shift",
            "No finding",
        ]
        def parse_vec(s):
            # Example formats: "0 1 0 ..." or "0,1,0,..."
            s = str(s).strip()
            if "," in s:
                parts = s.split(",")
            else:
                parts = s.split()
            return np.array([int(p) for p in parts], dtype=np.int64)

        labels_mat = np.stack(df["labels"].apply(parse_vec).values, axis=0)
        assert labels_mat.shape[1] == len(VINDR_LABELS), \
            f"labels vector length {labels_mat.shape[1]} != len(VINDR_LABELS) {len(VINDR_LABELS)}"
        return labels_mat, VINDR_LABELS

    # Case 2: wide format – treat all non-ID columns as labels
    # Common id columns:
    id_like = {"image_id", "imageID", "rad_id", "radID", "study_id"}
    label_cols = [c for c in cols if c not in id_like]
    labels_mat = df[label_cols].values.astype(np.int64)
    return labels_mat, label_cols


class VinDrTestDataset(Dataset):
    def __init__(self, csv_path: str, images_root: str, image_res: int = 256, img_ext=".png"):
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.img_ext = img_ext

        labels_mat, label_cols = infer_label_columns(self.df)
        self.labels = labels_mat
        self.label_names = label_cols

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        # assume 'image_id' column exists; adjust if named differently
        if "image_id" in self.df.columns:
            self.image_ids = self.df["image_id"].tolist()
        else:
            # try common variants
            for c in self.df.columns:
                if re.fullmatch(r"image[_ ]?id", c, re.IGNORECASE):
                    self.image_ids = self.df[c].tolist()
                    break
            else:
                raise ValueError("Could not find image_id column in CSV.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.images_root / f"{image_id}{self.img_ext}"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label_vec = self.labels[idx]
        return img, torch.from_numpy(label_vec).float(), image_id


def build_prompts(label_names):
    prompts = []
    for label in label_names:
        if re.search("no finding", label, re.IGNORECASE):
            prompts.append("A normal chest x-ray with no abnormal findings.")
        else:
            prompts.append(f"A chest x-ray showing {label}.")
    return prompts


def get_image_text_features(model, images, text_inputs):
    """
    Extract image & text embeddings from ALBEF for zero-shot.

    This assumes your ALBEF implementation has:
      - model.visual_encoder
      - model.vision_proj
      - model.text_encoder
      - model.text_proj

    If your attribute names differ, adapt accordingly.
    """
    # Image features
    image_embeds = model.visual_encoder(images)
    # CLS token = first token
    image_feat = model.vision_proj(image_embeds[:, 0, :])
    image_feat = F.normalize(image_feat, dim=-1)

    # Text features
    text_output = model.text_encoder(
        text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
        return_dict=True,
    )
    text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
    text_feat = F.normalize(text_feat, dim=-1)

    return image_feat, text_feat


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    image_res = config.get("image_res", 256)
    text_encoder_name = args.text_encoder or config.get("text_encoder", "bert-base-uncased")

    # Dataset
    dataset = VinDrTestDataset(
        csv_path=args.labels_csv,
        images_root=args.images_root,
        image_res=image_res,
        img_ext=args.img_ext,
    )
    label_names = dataset.label_names
    print(f"Loaded {len(dataset)} VinDr test samples with {len(label_names)} labels")
    print("Labels:", label_names)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model + tokenizer
    tokenizer = BertTokenizer.from_pretrained(text_encoder_name)
    model = ALBEF(config=config, text_encoder=text_encoder_name, tokenizer=tokenizer, init_deit=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    # Prompts & text inputs
    prompts = build_prompts(label_names)
    text_inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=25,
        return_tensors="pt",
    ).to(device)

    all_labels = []
    all_scores = []

    with torch.no_grad():
        # Precompute text features once
        dummy_images = torch.zeros(1, 3, image_res, image_res, device=device)
        _, text_feat = get_image_text_features(model, dummy_images, text_inputs)
        text_feat = text_feat.detach()  # (L, D)

        for images, labels, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            labels_np = labels.numpy()

            image_feat, _ = get_image_text_features(model, images, text_inputs)
            # Normalize again if needed
            image_feat = F.normalize(image_feat, dim=-1)
            sims = image_feat @ text_feat.t()  # (B, L)
            scores_np = sims.cpu().numpy()

            all_labels.append(labels_np)
            all_scores.append(scores_np)

    all_labels = np.concatenate(all_labels, axis=0)  # (N, L)
    all_scores = np.concatenate(all_scores, axis=0)  # (N, L)

    results = {}
    for j, label in enumerate(label_names):
        y_true = all_labels[:, j]
        y_score = all_scores[:, j]

        # Skip labels with no positives or no negatives
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            auc = float("nan")
            best_f1 = 0.0
            best_thr = 0.0
        else:
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = float("nan")

            # Best F1 over thresholds
            best_f1 = 0.0
            best_thr = 0.0
            # choose thresholds between percentiles of scores
            thr_grid = np.linspace(np.percentile(y_score, 5), np.percentile(y_score, 95), 50)
            for thr in thr_grid:
                y_pred = (y_score >= thr).astype(int)
                if y_pred.sum() == 0 and y_true.sum() == 0:
                    continue
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr

        results[label] = {
            "auc": float(auc),
            "best_f1": float(best_f1),
            "best_threshold": float(best_thr),
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_path}")

    print("\nLabel\tAUC\tF1")
    for label in label_names:
        r = results[label]
        print(f"{label}\t{r['auc']:.3f}\t{r['best_f1']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ALBEF pretrain config yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to ALBEF checkpoint_XX.pth")
    parser.add_argument("--labels_csv", default="Annotations/image_labels_test.csv")
    parser.add_argument("--images_root", default="Test", help="Folder with VinDr test PNGs")
    parser.add_argument("--img_ext", default=".png")
    parser.add_argument("--output_json", default="results/vindr_zero_shot_test.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_encoder", default=None)
    args = parser.parse_args()
    main(args)
