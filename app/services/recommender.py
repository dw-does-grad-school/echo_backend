from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FAISS_PATH = MODELS_DIR / "wikiart_clip.faiss"
META_PATH = MODELS_DIR / "wikiart_meta.npz"

# Globals
_index: faiss.Index = None
_meta: Dict[str, np.ndarray] = {}
_clip_model: CLIPModel = None
_clip_processor: CLIPProcessor = None

# 🔥 Use GPU if available (will be True on RunPod)
_device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_art_index():
    """Load FAISS, metadata, and CLIP model."""
    global _index, _meta, _clip_model, _clip_processor

    print(f"🔧 load_art_index() called. torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"🔧 Using device: '{_device}'")

    # ---- Load FAISS ----
    if _index is None:
        if not FAISS_PATH.exists():
            raise RuntimeError(f"FAISS index not found at {FAISS_PATH}")
        _index = faiss.read_index(str(FAISS_PATH))
        print(f"🔥 FAISS index dimension: {_index.d}")

    # ---- Load metadata ----
    if not _meta:
        if not META_PATH.exists():
            raise RuntimeError(f"Meta file not found at {META_PATH}")
        meta_npz = np.load(META_PATH, allow_pickle=True)
        _meta["artist"] = meta_npz["artist"]
        _meta["style"] = meta_npz["style"]
        _meta["genre"] = meta_npz["genre"]
        print("📁 Metadata loaded with lengths:", {k: len(v) for k, v in _meta.items()})

    # ---- Load CLIP ----
    if _clip_model is None or _clip_processor is None:
        print(f"📦 Loading CLIP model: {MODEL_NAME}")
        _clip_model = CLIPModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # explicit dtype
        ).to(_device)

        _clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _clip_model.eval()
        print("✅ CLIP model loaded and set to eval()")


def encode_text(query: str) -> np.ndarray:
    """Encode text to CLIP embedding."""
    if _clip_model is None:
        load_art_index()

    inputs = _clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # send tensors to the correct device
    for k in inputs:
        inputs[k] = inputs[k].to(_device)

    with torch.no_grad():
        text_features = _clip_model.get_text_features(**inputs)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print("✨ Text embedding shape:", text_features.shape)
    return text_features.cpu().numpy().astype("float32")


def search_similar_artworks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search FAISS and return metadata."""
    if _index is None:
        load_art_index()

    query_emb = encode_text(query)  # shape (1, d)
    distances, indices = _index.search(query_emb, k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1:
            continue

        results.append({
            "rank": rank,
            "index": int(idx),
            "distance": float(dist),
            "artist": _meta["artist"][idx].item() if hasattr(_meta["artist"][idx], "item") else _meta["artist"][idx],
            "style": _meta["style"][idx].item() if hasattr(_meta["style"][idx], "item") else _meta["style"][idx],
            "genre": _meta["genre"][idx].item() if hasattr(_meta["genre"][idx], "item") else _meta["genre"][idx],
        })

    return results
