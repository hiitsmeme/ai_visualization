"""Step 1: run CLIP zero-shot on all FairFace images and save the model outputs.

Saves one row per image: FairFace labels plus one logit column per prompt (the prompts
and column names come from config.all_prompts()). A logit is CLIP's `logits_per_image`:
logit_scale (100 for this model) * cosine similarity of the image and the prompt.
Only these outputs are stored, no embeddings, so Phase I stays black-box.

Usage:
    py -3.12 score_fairface.py              # all 97,698 images (val + train) -> results/scores.parquet
    py -3.12 score_fairface.py --limit 512  # quick test run -> results/scores_test_512.parquet
"""
import argparse
import io
import time

import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config

PROGRESS_EVERY = 40  # print a progress line every this many batches


# ---------------------------------------------------------------- input
def iter_batches(batch_size, limit=None):
    """Yield (split, rows) for every FairFace image, `batch_size` rows at a time.

    `rows` is a list of parquet rows as dicts: {"image": {"bytes", "path"}, "age", "gender",
    "race", ...}, with the labels as integer codes (see config). Stops after `limit` images.
    """
    seen = 0
    for split, files in config.PARQUET_FILES.items():
        for name in files:
            parquet = pq.ParquetFile(config.DATA_DIR / name)
            for batch in parquet.iter_batches(batch_size=batch_size):
                rows = batch.to_pylist()
                if limit is not None:
                    rows = rows[: limit - seen]  # last batch: keep only what is left of the limit
                seen += len(rows)
                yield split, rows
                if limit is not None and seen >= limit:
                    return


def decode_images(rows):
    """Decode the JPEG bytes of each parquet row into an RGB PIL image."""
    return [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]


def load_model(device):
    """CLIP model (in eval mode, on `device`) and its processor."""
    # See config.WEIGHTS_REVISION for why the weights come from a PR branch.
    model = CLIPModel.from_pretrained(
        config.MODEL_ID, revision=config.WEIGHTS_REVISION, use_safetensors=True
    ).to(device).eval()
    # use_fast=False keeps the original (slow) processors; the fast ones can give slightly different inputs.
    processor = CLIPProcessor.from_pretrained(config.MODEL_ID, use_fast=False)
    return model, processor


def to_records(split, rows, prompt_columns, logits):
    """One output row per image: image id, FairFace labels as text and one logit per prompt column.

    logits: array of shape (len(rows), len(prompt_columns)), in the order of `prompt_columns`.
    """
    return [
        {
            "image_id": f"{split}/{r['image']['path']}",
            "split": split,
            "gender": config.GENDERS[r["gender"]],
            "race": config.RACES[r["race"]],
            "age": config.AGE_GROUPS[r["age"]],
            **dict(zip(prompt_columns, row_logits.astype("float32"))),
        }
        for r, row_logits in zip(rows, logits)
    ]


def output_path(limit):
    """Where to save the scores. Test runs (--limit) get their own file so they never overwrite the full run."""
    return config.SCORES_PATH if limit is None else config.RESULTS_DIR / f"scores_test_{limit}.parquet"


@torch.no_grad()  # inference only, no gradients needed
def main(limit=None, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(device)

    # The prompts are the same for every batch, so they are tokenized only once.
    prompts = config.all_prompts()  # column name -> prompt text
    prompt_columns = list(prompts)
    text_inputs = processor(text=list(prompts.values()), return_tensors="pt", padding=True).to(device)

    records, start = [], time.time()
    for batch_number, (split, rows) in enumerate(iter_batches(batch_size, limit), start=1):
        pixels = processor(images=decode_images(rows), return_tensors="pt")["pixel_values"].to(device)
        # A full forward pass of image and text: we only keep the final output (logits_per_image,
        # shape images x prompts) and never look at the embeddings, so CLIP stays a black box.
        logits = model(pixel_values=pixels, **text_inputs).logits_per_image.cpu().numpy()
        records += to_records(split, rows, prompt_columns, logits)
        if batch_number % PROGRESS_EVERY == 0:
            rate = len(records) / (time.time() - start)
            print(f"{len(records):>6} images  {rate:6.0f} img/s", flush=True)

    scores = pd.DataFrame.from_records(records)
    out = output_path(limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out, index=False)
    print(f"saved {len(scores)} rows x {len(prompts)} prompts to {out} "
          f"in {time.time() - start:.0f} s on {device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N images")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="images per forward pass (lower it if the GPU runs out of memory)")
    args = parser.parse_args()
    main(limit=args.limit, batch_size=args.batch_size)
