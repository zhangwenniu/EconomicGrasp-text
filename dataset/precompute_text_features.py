"""
Offline precomputation of CLIP text features from scene descriptions.

Usage:
    pip install open_clip_torch
    python dataset/precompute_text_features.py \
        --dataset_root /path/to/graspnet \
        --descriptions descriptions.json

    # specify a local checkpoint if the server has no internet:
    python dataset/precompute_text_features.py \
        --dataset_root /path/to/graspnet \
        --descriptions descriptions.json \
        --clip_ckpt /path/to/ViT-B-32.pt

Output layout (mirrors graspness/):
    {dataset_root}/text_features/{scene}/{camera}/{frame}.npz
        scene_feat  : [512]              mean-pooled scene embedding (for FiLM)
        obj_feats   : [MAX_OBJ, 512]     per-object embeddings, zero-padded
        obj_mask    : [MAX_OBJ]          1 for valid slots, 0 for padding
        num_objects : scalar int

Dependencies:
    pip install open_clip_torch
    (open_clip_torch does NOT require torchvision at runtime)
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch

MAX_OBJ = 20   # max objects per frame; extra objects are truncated


def load_clip(model_name: str, pretrained: str, ckpt_path: str | None, device):
    import open_clip
    if ckpt_path and os.path.exists(ckpt_path):
        # load from a local .pt file (no internet needed)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=ckpt_path)
        print(f"Loaded CLIP from local checkpoint: {ckpt_path}")
    else:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        print(f"Loaded CLIP: {model_name} / {pretrained}")
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, tokenizer


def build_text(obj: dict) -> str:
    return f"a {obj['color']} {obj['name']}"


@torch.no_grad()
def encode_batch(texts, model, tokenizer, device) -> np.ndarray:
    tokens = tokenizer(texts).to(device)            # [K, 77]
    feats  = model.encode_text(tokens)              # [K, 512]
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def parse_key(key: str):
    """'scene_0000_kinect_0000.png'  ->  ('scene_0000', 'kinect', '0000')"""
    name  = key.replace(".png", "")        # scene_0000_kinect_0000
    parts = name.split("_")               # ['scene', '0000', 'kinect', '0000']
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
    parser.add_argument("--clip_model",    default="ViT-B-32",
                        help="open_clip model name (default: ViT-B-32)")
    parser.add_argument("--clip_pretrained", default="openai",
                        help="open_clip pretrained tag (default: openai)")
    parser.add_argument("--clip_ckpt",     default=None,
                        help="Path to a local CLIP .pt checkpoint "
                             "(overrides --clip_pretrained, no internet needed)")
    args = parser.parse_args()

    # ── load descriptions ────────────────────────────────────────────────────
    with open(args.descriptions, "r") as f:
        descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} entries from {args.descriptions}")

    # ── load CLIP text encoder ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_clip(
        args.clip_model, args.clip_pretrained, args.clip_ckpt, device
    )

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
            texts         = [build_text(o) for o in objs]
            obj_feats_raw = encode_batch(texts, model, tokenizer, device)  # [K, 512]
        else:
            obj_feats_raw = np.zeros((1, 512), dtype=np.float32)

        K          = len(obj_feats_raw)
        scene_feat = obj_feats_raw.mean(axis=0).astype(np.float32)        # [512]

        # zero-pad to MAX_OBJ
        padded    = np.zeros((MAX_OBJ, 512), dtype=np.float32)
        mask      = np.zeros(MAX_OBJ,       dtype=np.float32)
        n         = min(K, MAX_OBJ)
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
