#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    requested_device = str(args.device).strip().lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype="auto",
    )
    model.to(requested_device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            raw_image_paths = payload.get("image_paths")
            if raw_image_paths is None:
                raw_image_paths = [payload["image_path"]]
            images = [Image.open(Path(image_path)).convert("RGB") for image_path in raw_image_paths]
            messages = [
                {
                    "role": "user",
                    "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": payload["prompt"]}]),
                }
            ]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=text, images=images, return_tensors="pt")
            inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            max_new_tokens = int(payload.get("max_new_tokens", args.max_new_tokens))
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            trimmed_ids = []
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
                trimmed_ids.append(out_ids[len(in_ids):])
            raw_output = processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            sys.stdout.write(json.dumps({"raw_output": raw_output}) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({"error": str(exc)}) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
