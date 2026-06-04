from __future__ import annotations

import pickle
import select
import struct
import time
from pathlib import Path
from typing import Any, Optional


def read_obs(obs_pipe: Path, *, timeout_sec: Optional[float] = None, logger: Optional[Any] = None) -> Any:
    with open(obs_pipe, "rb") as pipe:
        return read_obs_file(pipe, source=str(obs_pipe), timeout_sec=timeout_sec, logger=logger)


def read_obs_file(
    pipe: Any,
    *,
    source: str = "open obs pipe handle",
    timeout_sec: Optional[float] = None,
    logger: Optional[Any] = None,
) -> Any:
    if timeout_sec is not None:
        fd = pipe.fileno()
        ready, _, _ = select.select([fd], [], [], float(timeout_sec))
        if not ready:
            if logger is not None:
                try:
                    import os

                    fd = pipe.fileno()
                    stat_result = os.fstat(fd)
                    logger.error(
                        "Timeout waiting for FIFO readability after %.1fs: fd=%s inode=%s mode=%s",
                        float(timeout_sec),
                        fd,
                        stat_result.st_ino,
                        oct(stat_result.st_mode),
                    )
                except Exception:
                    logger.error("Timeout waiting for FIFO readability after %.1fs", float(timeout_sec))
            raise TimeoutError(f"No obs_pipe data became readable within {float(timeout_sec)} seconds")

    header = pipe.read(8)
    if len(header) != 8:
        raise EOFError(f"Incomplete pipe header from {source}")
    payload_size = struct.unpack("<Q", header)[0]
    payload = bytearray()
    while len(payload) < payload_size:
        chunk = pipe.read(payload_size - len(payload))
        if not chunk:
            raise EOFError(f"Incomplete pipe payload from {source}")
        payload.extend(chunk)
    return pickle.loads(payload)


def write_plan(plan_pipe: Path, plan: Any) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    with open(plan_pipe, "wb") as pipe:
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)


def write_plan_file(
    pipe: Any,
    plan: Any,
    *,
    logger: Optional[Any] = None,
    log_fd_flags: bool = False,
) -> None:
    payload = pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL)
    if logger is not None:
        try:
            fd = pipe.fileno()
            flags = None
            if log_fd_flags:
                try:
                    import fcntl

                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                except Exception:
                    flags = None
            if log_fd_flags:
                logger.info("write_plan_file: fd=%s flags=%s bytes=%d t=%.6f", fd, str(flags), len(payload), time.time())
            else:
                logger.info("write_plan_file: fd=%s bytes=%d t=%.6f", fd, len(payload), time.time())
        except Exception:
            pass
    pipe.write(struct.pack("<Q", len(payload)))
    pipe.write(payload)
    pipe.flush()
    if logger is not None:
        try:
            logger.info("write_plan_file complete: fd=%s bytes=%d t=%.6f", pipe.fileno(), len(payload), time.time())
        except Exception:
            pass
