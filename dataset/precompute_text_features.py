"""
Offline precomputation of CLIP text features from scene descriptions.

The descriptions JSON typically covers one representative frame per
scene+camera (e.g. frame 0000).  This script propagates those features
to ALL 256 frames of each scene+camera so that every training sample
receives a text embedding.

Usage:
    pip install open_clip_torch
    python dataset/precompute_text_features.py \
        --dataset_root /path/to/graspnet \
        --descriptions descriptions.json

    # Use a local checkpoint when the server has no internet:
    python dataset/precompute_text_features.py \
        --dataset_root /path/to/graspnet \
        --descriptions descriptions.json \
        --clip_ckpt /path/to/ViT-B-32.pt

Output  (mirrors graspness/ layout):
    {dataset_root}/text_features/{scene}/{camera}/{frame}.npz
        scene_feat  : [512]           mean-pooled scene embedding (for FiLM)
        obj_feats   : [MAX_OBJ, 512]  per-object embeddings, zero-padded
        obj_mask    : [MAX_OBJ]       1 = valid slot, 0 = padding
        num_objects : scalar int
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch

MAX_OBJ    = 20    # max objects per frame; excess objects are truncated
NUM_FRAMES = 256   # frames per scene in GraspNet


# ── CLIP loading ─────────────────────────────────────────────────────────────

def load_clip(model_name: str, pretrained: str, ckpt_path: str | None, device):
    import open_clip
    if ckpt_path and os.path.exists(ckpt_path):
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=ckpt_path)
        print(f"Loaded CLIP from local checkpoint: {ckpt_path}")
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        print(f"Loaded CLIP: {model_name} / {pretrained}")
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, tokenizer


# ── helpers ──────────────────────────────────────────────────────────────────

def build_text(obj: dict) -> str:
    return f"a {obj['color']} {obj['name']}"


@torch.no_grad()
def encode_batch(texts, model, tokenizer, device) -> np.ndarray:
    tokens = tokenizer(texts).to(device)
    feats  = model.encode_text(tokens)              # [K, 512]
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def make_npz(obj_feats_raw: np.ndarray) -> dict:
    """Pack raw [K, 512] object features into the saveable dict."""
    scene_feat       = obj_feats_raw.mean(axis=0).astype(np.float32)
    K                = len(obj_feats_raw)
    n                = min(K, MAX_OBJ)
    padded           = np.zeros((MAX_OBJ, 512), dtype=np.float32)
    mask             = np.zeros(MAX_OBJ,        dtype=np.float32)
    padded[:n]       = obj_feats_raw[:n]
    mask[:n]         = 1.0
    return dict(
        scene_feat  = scene_feat,
        obj_feats   = padded,
        obj_mask    = mask,
        num_objects = np.array(n, dtype=np.int32),
    )


def parse_key(key: str):
    """'scene_0000_kinect_0000.png'  ->  ('scene_0000', 'kinect', '0000')"""
    parts  = key.replace(".png", "").split("_")   # ['scene','0000','kinect','0000']
    scene  = f"scene_{parts[1]}"
    camera = parts[2]
    frame  = parts[3]
    return scene, camera, frame


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",    required=True)
    parser.add_argument("--descriptions",    default="descriptions.json")
    parser.add_argument("--clip_model",      default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="openai")
    parser.add_argument("--clip_ckpt",       default=None,
                        help="Local .pt checkpoint path (skips download)")
    args = parser.parse_args()

    # load descriptions
    with open(args.descriptions, "r") as f:
        descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} entries from {args.descriptions}")

    # load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_clip(
        args.clip_model, args.clip_pretrained, args.clip_ckpt, device)

    output_root = os.path.join(args.dataset_root, "text_features")

    # ── Step 1: encode each unique (scene, camera) description ───────────────
    # descriptions.json typically has one entry per (scene, camera);
    # we store the encoded result keyed by (scene, camera).
    scene_camera_feats: dict[tuple, dict] = {}
    skipped = 0

    for key, objs in tqdm(descriptions.items(), desc="Encoding descriptions"):
        try:
            scene, camera, _ = parse_key(key)
        except Exception:
            skipped += 1
            continue

        sc_key = (scene, camera)
        if sc_key in scene_camera_feats:
            continue                    # already encoded this scene+camera

        if objs:
            texts         = [build_text(o) for o in objs]
            obj_feats_raw = encode_batch(texts, model, tokenizer, device)
        else:
            obj_feats_raw = np.zeros((1, 512), dtype=np.float32)

        scene_camera_feats[sc_key] = make_npz(obj_feats_raw)

    print(f"Unique (scene, camera) pairs encoded: {len(scene_camera_feats)}")
    if skipped:
        print(f"  ({skipped} entries skipped due to key parse errors)")

    # ── Step 2: write features for ALL 256 frames of each scene+camera ───────
    total_written = 0
    for (scene, camera), npz_dict in tqdm(
            scene_camera_feats.items(), desc="Writing frames"):
        out_dir = os.path.join(output_root, scene, camera)
        os.makedirs(out_dir, exist_ok=True)
        for frame_id in range(NUM_FRAMES):
            out_path = os.path.join(out_dir, f"{frame_id:04d}.npz")
            np.savez_compressed(out_path, **npz_dict)
            total_written += 1

    print(f"Done. {total_written} .npz files saved to: {output_root}")


if __name__ == "__main__":
    main()
