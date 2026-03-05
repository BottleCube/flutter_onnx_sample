"""
Build a precomputed normalized embedding index for offline Flutter search.

Usage example:
    uv run build_index.py \
      --catalog ../assets/data/catalog.json \
      --onnx-model ../assets/models/BAAI_bge-small-en-v1.5.onnx \
      --tokenizer-model BAAI/bge-small-en-v1.5 \
      --output ../assets/data/search_index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized embedding search index from a catalog JSON file."
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to catalog JSON (list of items or {\"items\": [...]}).",
    )
    parser.add_argument(
        "--onnx-model",
        default=str(Path(__file__).parent.parent / "assets" / "models" / "BAAI_bge-small-en-v1.5.onnx"),
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="BAAI/bge-small-en-v1.5",
        help="Tokenizer model id or local tokenizer path.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "assets" / "data" / "search_index.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max length (default: 128).",
    )
    parser.add_argument(
        "--top-fields",
        nargs="+",
        default=["title", "summary", "genre"],
        help="Fields used to compose embedding input text.",
    )
    return parser.parse_args()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    else:
        raise ValueError("Catalog JSON must be a list or an object with an 'items' list.")
    return [item for item in items if isinstance(item, dict)]


def compose_text(item: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = item.get(field)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue
        parts.append(f"{field}: {value}")
    return "\n".join(parts)


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    token_vectors = last_hidden_state[0]  # [seq, hidden]
    mask = attention_mask[0].astype(np.float32)  # [seq]
    valid = mask.sum()
    if valid <= 0:
        raise ValueError("attention_mask has no valid tokens.")
    pooled = (token_vectors * mask[:, None]).sum(axis=0) / valid
    return pooled.astype(np.float32)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def encode_item(
    *,
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
) -> np.ndarray:
    encoded = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    outputs = session.run(
        None,
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        },
    )
    last_hidden_state = outputs[0]  # [batch, seq, hidden]
    pooled = mean_pool(last_hidden_state, encoded["attention_mask"])
    return l2_normalize(pooled)


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog)
    onnx_model_path = Path(args.onnx_model)
    output_path = Path(args.output)

    items = load_catalog(catalog_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    session = ort.InferenceSession(str(onnx_model_path))

    index_items: list[dict[str, Any]] = []
    dimension = 0
    for i, item in enumerate(items):
        text = compose_text(item, args.top_fields)
        if not text:
            continue
        vector = encode_item(
            session=session,
            tokenizer=tokenizer,
            text=text,
            max_length=args.max_length,
        )
        if dimension == 0:
            dimension = int(vector.shape[0])
        index_items.append(
            {
                "id": str(item.get("id", f"item-{i}")),
                "title": str(item.get("title", "")),
                "summary": str(item.get("summary", "")),
                "genre": str(item.get("genre", "")),
                "embedding": [round(float(v), 6) for v in vector.tolist()],
            }
        )
        if (i + 1) % 100 == 0:
            print(f"[progress] encoded {i + 1} items")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": str(onnx_model_path.name),
        "dimension": dimension,
        "items": index_items,
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"wrote index: {output_path}")
    print(f"items: {len(index_items)}")
    print(f"dimension: {dimension}")


if __name__ == "__main__":
    main()
