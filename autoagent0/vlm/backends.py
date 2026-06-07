from __future__ import annotations

import fcntl
import json
import logging
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

from PIL import Image

from autoagent0.vlm.parsing import normalize_token_usage, try_parse_json


LOG = logging.getLogger(__name__)


class Qwen3TrajectorySelector:
    def __init__(self, model_id: str, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        requested_device = str(device).strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
        )
        self.model.to(requested_device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.device = requested_device
        self._torch = torch

    def _run_inference(self, image_paths: Sequence[Path], prompt: str) -> Dict[str, object]:
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]),
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed_ids = []
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
            trimmed_ids.append(out_ids[len(in_ids):])
        raw_output = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        completion_tokens = int(len(trimmed_ids[0])) if trimmed_ids else 0
        return {
            "raw_output": raw_output,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def infer_prompt(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, object]:
        del timeout_sec
        started = time.time()
        prev_max_new_tokens = self.max_new_tokens
        try:
            if max_new_tokens is not None:
                self.max_new_tokens = int(max_new_tokens)
            inference_result = self._run_inference(image_paths, prompt)
        finally:
            if max_new_tokens is not None:
                self.max_new_tokens = prev_max_new_tokens
        raw_output = str(inference_result.get("raw_output", ""))
        return {
            "raw_output": raw_output,
            "parsed_output": try_parse_json(raw_output),
            "elapsed_sec": time.time() - started,
            "prompt": prompt,
            "token_usage": normalize_token_usage(inference_result.get("token_usage")),
        }


class SubprocessQwen3TrajectorySelector:
    def __init__(
        self,
        python_bin: str,
        worker_script: Path,
        model_id: str,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
    ) -> None:
        if not python_bin:
            raise ValueError("VLM python_bin is not set")
        self.python_bin = python_bin
        self.worker_script = worker_script
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.enable_thinking = bool(enable_thinking)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stderr_file = None
        self._ready = False
        self._stdout_buffer = ""

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        worker_log_path = self.worker_script.with_name("vlm_worker.stderr.log")
        self._stderr_file = worker_log_path.open("a", encoding="utf-8")
        self._proc = subprocess.Popen(
            [
                self.python_bin,
                "-B",
                str(self.worker_script),
                "--model-id",
                self.model_id,
                "--device",
                self.device,
                "--max-new-tokens",
                str(self.max_new_tokens),
                "--temperature",
                str(self.temperature),
                "--top-p",
                str(self.top_p),
                "--top-k",
                str(self.top_k),
                "--enable-thinking",
                "true" if self.enable_thinking else "false",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        self._ready = False
        self._stdout_buffer = ""
        return self._proc

    def _readline_with_timeout(self, proc: subprocess.Popen[str], timeout_sec: Optional[float]) -> str:
        if proc.stdout is None:
            raise RuntimeError("VLM worker stdout is unavailable")
        if "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            return line + "\n"
        deadline = None if timeout_sec is None else time.monotonic() + max(float(timeout_sec), 0.0)
        stdout_fd = proc.stdout.fileno()
        while True:
            if deadline is None:
                ready, _, _ = select.select([stdout_fd], [], [])
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"VLM worker subprocess timeout after {float(timeout_sec):.3f}s")
                ready, _, _ = select.select([stdout_fd], [], [], remaining)
                if not ready:
                    continue
            if ready:
                chunk = os.read(stdout_fd, 4096)
                if not chunk:
                    if self._stdout_buffer:
                        line = self._stdout_buffer
                        self._stdout_buffer = ""
                        return line
                    return ""
                self._stdout_buffer += chunk.decode("utf-8", errors="replace")
                if "\n" in self._stdout_buffer:
                    line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
                    return line + "\n"

    def preload(self, timeout_sec: Optional[float] = None) -> None:
        if self._ready and self._proc is not None and self._proc.poll() is None:
            return
        lock_path = os.environ.get("PLANNER_VLM_PRELOAD_LOCK_PATH", "").strip()
        lock_file = None
        if lock_path:
            lock_parent = os.path.dirname(lock_path)
            if lock_parent:
                os.makedirs(lock_parent, exist_ok=True)
            lock_file = open(lock_path, "a", encoding="utf-8")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            LOG.info("Acquired VLM preload lock: %s", lock_path)
        try:
            proc = self._ensure_proc()
            line = self._readline_with_timeout(proc, timeout_sec)
        finally:
            if lock_file is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                LOG.info("Released VLM preload lock: %s", lock_path)
        if not line:
            self.close()
            raise RuntimeError("VLM worker exited before signaling readiness")
        response = json.loads(line)
        if response.get("status") == "ready":
            self._ready = True
            return
        if response.get("error"):
            self.close()
            raise RuntimeError(str(response["error"]))
        self.close()
        raise RuntimeError(f"Unexpected VLM worker preload response: {response!r}")

    def infer_prompt(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, object]:
        self.preload(timeout_sec=timeout_sec)
        proc = self._ensure_proc()
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("VLM worker stdio is unavailable")

        started = time.time()
        payload = {"image_paths": [str(image_path) for image_path in image_paths], "prompt": prompt}
        if max_new_tokens is not None:
            payload["max_new_tokens"] = int(max_new_tokens)
        payload["temperature"] = self.temperature
        payload["top_p"] = self.top_p
        payload["top_k"] = self.top_k
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        try:
            line = self._readline_with_timeout(proc, timeout_sec)
        except TimeoutError:
            self.close()
            raise
        if not line:
            raise RuntimeError("VLM worker exited unexpectedly")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        raw_output = str(response.get("raw_output", ""))
        return {
            "raw_output": raw_output,
            "parsed_output": try_parse_json(raw_output),
            "elapsed_sec": time.time() - started,
            "prompt": prompt,
            "token_usage": normalize_token_usage(response.get("token_usage")),
        }

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        if self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

