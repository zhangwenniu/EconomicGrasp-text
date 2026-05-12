"""
Offline precomputation of CLIP text features from scene descriptions.

Usage:
    python dataset/precompute_text_features.py \
        --dataset_root /path/to/graspnet \
        --descriptions descriptions.json \
        --clip_model openai/clip-vit-base-patch32

Output layout (mirrors graspness/):
    {dataset_root}/text_features/{scene}/{camera}/{frame}.npz
        scene_feat  : [512]            mean-pooled scene embedding (for FiLM)
        obj_feats   : [MAX_OBJ, 512]   per-object embeddings, zero-padded
        obj_mask    : [MAX_OBJ]        1 for valid slots, 0 for padding
        num_objects : scalar int

Dependencies:
    pip install transformers
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import CLIPTokenizer, CLIPTextModel

MAX_OBJ = 20   # max objects per scene; rare scenes with more are truncated


def build_text(obj: dict) -> str:
    return f"a {obj['color']} {obj['name']}"


def encode_batch(texts, model, tokenizer, device):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        feats = model(**inputs).pooler_output          # [K, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def parse_key(key: str):
    """'scene_0000_kinect_0000.png'  ->  ('scene_0000', 'kinect', '0000')"""
    name = key.replace(".png", "")       # scene_0000_kinect_0000
    parts = name.split("_")             # ['scene', '0000', 'kinect', '0000']
    scene  = f"scene_{parts[1]}"
    camera = parts[2]
    frame  = parts[3]
    return scene, camera, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",  required=True,
                        help="Root directory of the GraspNet dataset")
    parser.add_argument("--descriptions",  default="descriptions.json",
                        help="Path to the JSON file with scene descriptions")
    parser.add_argument("--clip_model",    default="openai/clip-vit-base-patch32",
                        help="HuggingFace model ID for CLIP text encoder")
    args = parser.parse_args()

    # ── load descriptions ────────────────────────────────────────────────────
    with open(args.descriptions, "r") as f:
        descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} entries from {args.descriptions}")

    # ── load CLIP text encoder ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP text encoder on {device} ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    model = CLIPTextModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    output_root = os.path.join(args.dataset_root, "text_features")
    skipped = 0

    for key, objs in tqdm(descriptions.items(), desc="Encoding"):
        try:
            scene, camera, frame = parse_key(key)
        except Exception:
            skipped += 1
            continue

        # encode
        if objs:
            texts = [build_text(o) for o in objs]
            obj_feats_raw = encode_batch(texts, model, tokenizer, device)  # [K, 512]
        else:
            obj_feats_raw = np.zeros((1, 512), dtype=np.float32)

        K = len(obj_feats_raw)
        scene_feat = obj_feats_raw.mean(axis=0).astype(np.float32)         # [512]

        # zero-pad to MAX_OBJ
        padded = np.zeros((MAX_OBJ, 512), dtype=np.float32)
        mask   = np.zeros(MAX_OBJ,       dtype=np.float32)
        n = min(K, MAX_OBJ)
        padded[:n] = obj_feats_raw[:n]
        mask[:n]   = 1.0

        # save
        out_dir = os.path.join(output_root, scene, camera)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"{frame}.npz"),
            scene_feat  = scene_feat,
            obj_feats   = padded,
            obj_mask    = mask,
            num_objects = np.array(n, dtype=np.int32),
        )

    print(f"Done. Text features saved to: {output_root}")
    if skipped:
        print(f"  ({skipped} entries skipped due to key parse errors)")


if __name__ == "__main__":
    main()
