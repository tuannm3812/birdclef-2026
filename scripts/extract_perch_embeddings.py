from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from birdclef2026.audio import load_audio
from birdclef2026.config import ensure_dir, load_config
from birdclef2026.data import read_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/perch_probe.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--perch-model-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install TensorFlow or run `pip install -e .[perch]` to use Perch extraction.") from exc

    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg["data_root"])
    model_dir = args.perch_model_dir or cfg.get("perch", {}).get("model_dir")
    if not model_dir:
        raise SystemExit("Pass --perch-model-dir or set perch.model_dir in configs/perch_probe.yaml")

    output_path = Path(args.output or cfg["perch"]["embeddings_path"])
    ensure_dir(output_path.parent)
    df, _, _ = read_metadata(data_root)

    model = tf.saved_model.load(str(model_dir))
    infer = model.signatures["serving_default"]
    audio_cfg = cfg["audio"]
    embeddings = []

    for path in tqdm(df["filepath"], desc="perch"):
        y = load_audio(path, int(audio_cfg["sample_rate"]), float(audio_cfg["duration"]))
        batch = tf.convert_to_tensor(y[None, :], dtype=tf.float32)
        _, keyword_specs = infer.structured_input_signature
        if keyword_specs:
            input_name = next(iter(keyword_specs))
            outputs = infer(**{input_name: batch})
        else:
            outputs = infer(batch)
        arrays = {name: np.asarray(value) for name, value in outputs.items()}
        embedding_name = next(
            (name for name in arrays if "embedding" in name.lower() or "embed" in name.lower()),
            next(iter(arrays)),
        )
        embedding = to_2d_embeddings(arrays[embedding_name])[0]
        embeddings.append(embedding)

    np.savez_compressed(
        output_path,
        embeddings=np.stack(embeddings).astype(np.float32),
        labels=df["primary_label"].to_numpy(),
        filenames=df["filename"].to_numpy(),
    )
    print(f"Wrote embeddings for {len(df):,} clips to {output_path}")
    zip_path = output_path.parent / f"{output_path.parent.name}_embeddings.zip"
    zip_artifacts(output_path.parent, zip_path)
    print(f"Zipped artifacts to {zip_path}")


def zip_artifacts(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, arcname=path.relative_to(source_dir.parent))


def to_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 2:
        return embeddings
    if embeddings.ndim == 3:
        return embeddings.mean(axis=1)
    return embeddings.reshape(embeddings.shape[0], -1)


if __name__ == "__main__":
    main()
