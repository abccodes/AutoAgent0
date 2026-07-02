#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from queue import Empty
from pathlib import Path
from threading import Thread

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration, TextIteratorStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--enable-thinking", default="false")
    return parser.parse_args()


def _is_qwen36_model(model_id: str) -> bool:
    return "qwen3.6" in str(model_id).strip().lower()


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write(f"[vlm_worker {timestamp}] {message}\n")
    sys.stderr.flush()


def _strip_empty_think_block(text: str) -> str:
    return re.sub(r"<think>\s*</think>\s*", "", text, count=1, flags=re.DOTALL)


def main() -> int:
    args = parse_args()
    started_main = time.time()
    _log(f"python_executable={sys.executable}")
    _log(f"python_version={sys.version.replace(chr(10), ' ')}")
    _log(f"sys_path_head={sys.path[:5]}")
    _log(
        f"startup model_id={args.model_id} device={args.device} max_new_tokens={args.max_new_tokens} "
        f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} enable_thinking={args.enable_thinking}"
    )

    import torch

    force_cpu_offload = os.environ.get("PLANNER_VLM_FORCE_CPU_OFFLOAD", "").strip().lower() in {"1", "true", "yes"}
    requested_device = str(args.device).strip().lower()
    if requested_device == "auto":
        if force_cpu_offload:
            requested_device = "cpu"
        else:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_thinking = str(args.enable_thinking).strip().lower() in {"1", "true", "yes", "on"}
    use_qwen36_loader = _is_qwen36_model(args.model_id)

    load_started = time.time()
    _log(f"model_load_begin use_qwen36_loader={use_qwen36_loader} requested_device={requested_device}")
    if use_qwen36_loader:
        model_kwargs = {
            "dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        # Qwen3.6-27B-FP8 does not fit comfortably beside RAP on a single 48 GB GPU
        # without offload. Let accelerate split the model when requested or when the
        # caller leaves device on auto.
        if requested_device == "cpu":
            model_kwargs["device_map"] = "cpu"
        elif force_cpu_offload or requested_device == "auto" or requested_device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            **model_kwargs,
        )
    else:
        model_kwargs = {
            "dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        use_device_map = (
            force_cpu_offload
            or requested_device in {"auto", "cpu"}
        )
        if requested_device == "cpu":
            model_kwargs["device_map"] = "cpu"
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_id,
                **model_kwargs,
            )
        elif use_device_map:
            model_kwargs["device_map"] = "auto"
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_id,
                **model_kwargs,
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_id,
                dtype="auto",
            )
            model.to(requested_device)
    model.eval()
    _log(f"model_load_done elapsed_sec={time.time() - load_started:.3f}")
    processor_started = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    _log(f"processor_load_done elapsed_sec={time.time() - processor_started:.3f} total_startup_sec={time.time() - started_main:.3f}")
    tokenizer = getattr(processor, "tokenizer", None)
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request_started = time.time()
            payload = json.loads(line)
            raw_image_paths = payload.get("image_paths")
            if raw_image_paths is None:
                raw_image_paths = [payload["image_path"]]
            images = [Image.open(Path(image_path)).convert("RGB") for image_path in raw_image_paths]
            _log(
                "request_begin "
                f"num_images={len(images)} image_sizes={[image.size for image in images]} "
                f"prompt_chars={len(str(payload.get('prompt', '')))}"
            )
            tokenize_started = time.time()
            messages = [
                {
                    "role": "user",
                    "content": (
                        ([{"type": "image", "image": image} for image in images] + [{"type": "text", "text": payload["prompt"]}])
                        if use_qwen36_loader
                        else ([{"type": "image"} for _ in images] + [{"type": "text", "text": payload["prompt"]}])
                    ),
                }
            ]
            if use_qwen36_loader:
                try:
                    rendered_text = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    rendered_text = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                rendered_text = _strip_empty_think_block(rendered_text)
                inputs = processor(text=rendered_text, images=images, return_tensors="pt")
            else:
                try:
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                inputs = processor(text=text, images=images, return_tensors="pt")
            _log(
                "tokenize_done "
                f"elapsed_sec={time.time() - tokenize_started:.3f} "
                f"input_ids_len={getattr(inputs.get('input_ids'), 'shape', ['?','?'])[-1] if 'input_ids' in inputs else 'na'} "
                f"pixel_values_shape={tuple(inputs['pixel_values'].shape) if 'pixel_values' in inputs and hasattr(inputs['pixel_values'], 'shape') else 'na'}"
            )
            prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

            input_device = getattr(model, "device", None)
            if input_device is None:
                try:
                    input_device = next(model.parameters()).device
                except Exception:
                    input_device = requested_device
            inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            max_new_tokens = int(payload.get("max_new_tokens", args.max_new_tokens))
            temperature = float(payload.get("temperature", args.temperature))
            top_p = float(payload.get("top_p", args.top_p))
            top_k = int(payload.get("top_k", args.top_k))
            do_sample = bool(temperature > 0.0 or top_p < 1.0)
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if use_qwen36_loader and tokenizer is not None:
                bad_words_ids = []
                for phrase in ("<think>", "</think>"):
                    token_ids = tokenizer.encode(phrase, add_special_tokens=False)
                    if token_ids:
                        bad_words_ids.append(token_ids)
                if bad_words_ids:
                    generate_kwargs["bad_words_ids"] = bad_words_ids
                if do_sample:
                    generate_kwargs["temperature"] = max(temperature, 1e-5)
                    generate_kwargs["top_p"] = min(max(top_p, 0.0), 1.0)
                    if top_k > 0:
                        generate_kwargs["top_k"] = top_k
            streamer = None
            if tokenizer is not None and not use_qwen36_loader:
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    timeout=5.0,
                )
                generate_kwargs["streamer"] = streamer
            generate_started = time.time()
            _log(
                "generate_begin "
                f"device={input_device} do_sample={do_sample} "
                f"max_new_tokens={max_new_tokens} temperature={temperature} top_p={top_p} top_k={top_k}"
            )
            generated_ids = None
            stream_parts = []
            generation_error = {}

            def _run_generate() -> None:
                try:
                    with torch.inference_mode():
                        generation_error["generated_ids"] = model.generate(
                            **inputs,
                            **generate_kwargs,
                        )
                except Exception as exc:
                    generation_error["error"] = exc
                    generation_error["traceback"] = traceback.format_exc()

            generate_thread = Thread(target=_run_generate, daemon=True)
            generate_thread.start()

            if streamer is not None:
                last_wait_log = time.time()
                while generate_thread.is_alive():
                    try:
                        chunk = next(streamer)
                        stream_parts.append(chunk)
                        _log(
                            "stream_chunk "
                            f"chunk_index={len(stream_parts)} chunk_chars={len(chunk)} "
                            f"total_chars={sum(len(part) for part in stream_parts)} "
                            f"generate_elapsed_sec={time.time() - generate_started:.3f}"
                        )
                    except StopIteration:
                        break
                    except Empty:
                        now = time.time()
                        if now - last_wait_log >= 5.0:
                            _log(
                                "generate_waiting "
                                f"elapsed_sec={now - generate_started:.3f} "
                                f"chunk_count={len(stream_parts)} "
                                f"total_chars={sum(len(part) for part in stream_parts)}"
                            )
                            last_wait_log = now
                generate_thread.join()
                while True:
                    try:
                        chunk = next(streamer)
                        stream_parts.append(chunk)
                        _log(
                            "stream_chunk "
                            f"chunk_index={len(stream_parts)} chunk_chars={len(chunk)} "
                            f"total_chars={sum(len(part) for part in stream_parts)} "
                            f"generate_elapsed_sec={time.time() - generate_started:.3f}"
                        )
                    except (StopIteration, Empty):
                        break
            else:
                generate_thread.join()

            if "error" in generation_error:
                _log("generate_thread_error")
                _log(str(generation_error.get("traceback", "")))
                raise generation_error["error"]
            generated_ids = generation_error.get("generated_ids")
            _log(
                "generate_done "
                f"elapsed_sec={time.time() - generate_started:.3f} "
                f"stream_chunks={len(stream_parts)} stream_chars={sum(len(part) for part in stream_parts)}"
            )
            trimmed_ids = []
            if stream_parts:
                raw_output = "".join(stream_parts)
                if tokenizer is not None and raw_output:
                    try:
                        trimmed_ids = [tokenizer.encode(raw_output, add_special_tokens=False)]
                    except Exception as exc:
                        _log(f"completion_token_estimate_error {repr(exc)}")
            else:
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
                    trimmed_ids.append(out_ids[len(in_ids):])
                if use_qwen36_loader:
                    try:
                        first_in_len = int(len(inputs["input_ids"][0]))
                        first_out_len = int(len(generated_ids[0]))
                        first_trimmed = trimmed_ids[0]
                        first_trimmed_list = [int(tok) for tok in first_trimmed[:64].tolist()]
                        tokenizer = getattr(processor, "tokenizer", None)
                        tokenizer_decode = ""
                        if tokenizer is not None:
                            try:
                                tokenizer_decode = tokenizer.decode(
                                    first_trimmed,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False,
                                )
                            except Exception as exc:
                                tokenizer_decode = f"<tokenizer_decode_error:{exc}>"
                        batch_preview = ""
                        try:
                            batch_preview = processor.batch_decode(
                                [first_trimmed],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )[0]
                        except Exception as exc:
                            batch_preview = f"<batch_decode_error:{exc}>"
                        _log(
                            "decode_debug "
                            f"input_ids_len={first_in_len} generated_ids_len={first_out_len} "
                            f"trimmed_ids_len={int(len(first_trimmed))} "
                            f"first_trimmed_token_ids={first_trimmed_list}"
                        )
                        _log(
                            "decode_debug_preview "
                            f"batch_decode_preview={batch_preview[:200]!r} "
                            f"tokenizer_decode_preview={tokenizer_decode[:200]!r}"
                        )
                    except Exception as exc:
                        _log(f"decode_debug_error {repr(exc)}")
                raw_output = processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            completion_tokens = int(len(trimmed_ids[0])) if trimmed_ids else 0
            _log(
                "decode_done "
                f"request_elapsed_sec={time.time() - request_started:.3f} raw_output_chars={len(raw_output)}"
            )
            sys.stdout.write(
                json.dumps(
                    {
                        "raw_output": raw_output,
                        "token_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
        except Exception as exc:
            _log(f"request_error {repr(exc)}")
            _log(traceback.format_exc().rstrip())
            sys.stdout.write(json.dumps({"error": str(exc)}) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
