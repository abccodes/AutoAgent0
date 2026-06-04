from __future__ import annotations

import pickle
import os
import select
import struct
import time
from typing import Any, Optional


def write_pipe_message_file(pipe: Any, payload_obj: Any) -> None:
    payload = pickle.dumps(payload_obj, protocol=pickle.HIGHEST_PROTOCOL)
    pipe.write(struct.pack("<Q", len(payload)))
    pipe.write(payload)
    pipe.flush()
    try:
        fd = pipe.fileno()
        stat_result = os.fstat(fd)
        print(
            f"[write_pipe_message_file] fd={fd} inode={stat_result.st_ino} "
            f"mode={oct(stat_result.st_mode)} bytes={len(payload)} t={time.time():.6f}"
        )
    except Exception:
        print(f"[write_pipe_message_file] logging failed t={time.time():.6f}")


def raise_if_process_exited(process: Any, context: str) -> None:
    if process is None:
        return
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(f"Planner process exited with return code {return_code} while {context}")


def read_exact_pipe_bytes(
    pipe: Any,
    size: int,
    *,
    producer_process: Optional[Any] = None,
    context: str = "reading pipe",
    timeout_sec: Optional[float] = None,
) -> bytes:
    chunks = bytearray()
    fd = pipe.fileno()
    deadline = None
    if timeout_sec is not None and float(timeout_sec) > 0:
        deadline = time.monotonic() + float(timeout_sec)
    while len(chunks) < size:
        raise_if_process_exited(producer_process, context)
        select_timeout = 1.0
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {float(timeout_sec):.1f}s while {context}; "
                    f"received {len(chunks)}/{size} bytes"
                )
            select_timeout = min(select_timeout, remaining)
        ready, _, _ = select.select([fd], [], [], select_timeout)
        if not ready:
            continue
        chunk = pipe.read(size - len(chunks))
        if not chunk:
            raise EOFError(f"Incomplete pipe payload from open pipe handle while {context}")
        chunks.extend(chunk)
    return bytes(chunks)


def read_pipe_message_file(pipe: Any, *, producer_process: Optional[Any] = None, timeout_sec: Optional[float] = None) -> Any:
    try:
        print(f"[read_pipe_message_file] waiting fd={pipe.fileno()} t={time.time():.6f}")
    except Exception:
        print(f"[read_pipe_message_file] waiting fd=? t={time.time():.6f}")
    header = read_exact_pipe_bytes(
        pipe,
        8,
        producer_process=producer_process,
        context="waiting for planner response header",
        timeout_sec=timeout_sec,
    )
    if len(header) != 8:
        raise EOFError("Incomplete pipe header from open pipe handle")
    payload_size = struct.unpack("<Q", header)[0]
    payload = read_exact_pipe_bytes(
        pipe,
        payload_size,
        producer_process=producer_process,
        context="waiting for planner response payload",
        timeout_sec=timeout_sec,
    )
    try:
        print(f"[read_pipe_message_file] received fd={pipe.fileno()} bytes={payload_size} t={time.time():.6f}")
    except Exception:
        print(f"[read_pipe_message_file] received fd=? bytes={payload_size} t={time.time():.6f}")
    return pickle.loads(payload)
