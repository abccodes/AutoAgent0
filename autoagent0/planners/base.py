"""Base abstractions for HUGSIM planner subprocesses.

A planner backend (RAP, DrivoR, rule-based, ...) runs as its own subprocess
because each needs a different, mutually incompatible Python environment. The
subprocess does ONE thing: turn an observation into trajectory proposals and
scores. Candidate selection, VLM/AutoAgent0 reasoning, payload construction and
simulator stepping all live on the pipeline side (the pixi process).

The FIFO contract is therefore minimal:

    sim  --(obs, info[, privileged_info])-->  planner
    sim  <--(proposals[N, T, 2], scores[N])--  planner

``proposals`` are already in HUGSIM local coordinates ([x_right, y_forward]) so
the pipeline stays planner-agnostic.
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Sequence

import numpy as np

from autoagent0.adapters.hugsim.io import read_obs_file, write_plan_file


@dataclass
class PlannerResult:
    """Pure inference output of one planning step.

    ``proposals``: ``np.ndarray`` of shape ``[N, T, 2]`` in HUGSIM local
    coordinates. ``scores``: ``np.ndarray`` of shape ``[N]``.
    """

    proposals: np.ndarray
    scores: np.ndarray


class PlannerService(ABC):
    """A planner backend driven by :func:`run_subprocess`.

    Lifecycle: ``setup()`` once -> ``process()`` per frame -> ``finalize()`` once.
    Any history the model needs is accumulated by the runtime and passed to
    ``process``; the service itself stays stateless across frames.
    """

    #: ego-history frames the model expects (sizes the runtime's info deque).
    history_frames: int = 4

    def setup(self) -> None:
        """Load models / heavy resources. Called once before the read loop."""

    @abstractmethod
    def process(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        info_history: Sequence[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> PlannerResult:
        """Run inference for the current frame.

        ``info_history`` is ordered oldest->newest, padded to ``history_frames``.
        ``extra`` carries auxiliary inputs such as ``privileged_info`` (which only
        the rule-based planner consumes; learned planners ignore it).
        """

    def finalize(self) -> None:
        """Release resources. Called once when the loop exits (even on error)."""


def _open_fifo_rdwr(path: Path, mode: str):
    # O_RDWR avoids open-order deadlocks against the simulator side.
    return os.fdopen(os.open(path, os.O_RDWR), mode, buffering=0)


def run_subprocess(service: PlannerService, output_dir: Path, *, logger: logging.Logger | None = None) -> int:
    """Drive ``service`` over the HUGSIM FIFO protocol. Returns a process exit code.

    This is the thin loop that runs inside every planner subprocess: read an
    observation, run :meth:`PlannerService.process`, write back
    ``(proposals, scores)``. Everything else is the pipeline's job.
    """
    log = logger or logging.getLogger("planner")
    output_dir = Path(output_dir).resolve()

    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"
    log.info("Waiting for scene FIFOs to appear: obs=%s plan=%s", obs_pipe, plan_pipe)
    while not obs_pipe.exists() or not plan_pipe.exists():
        time.sleep(0.05)

    obs_pipe_reader = _open_fifo_rdwr(obs_pipe, "rb")
    plan_pipe_writer = _open_fifo_rdwr(plan_pipe, "wb")
    log.info("Opened persistent scene FIFOs for obs and plan exchange")

    info_history: Deque[Dict[str, object]] = deque(maxlen=service.history_frames)

    try:
        log.info("Entering planner read loop; waiting for observations on %s", obs_pipe)
        while True:
            try:
                message = read_obs_file(obs_pipe_reader)
                if message == "Done":
                    log.info("Received shutdown signal")
                    break

                if isinstance(message, dict) and message.get("message_type") == "hugsim_preflight":
                    log.info(
                        "Received HUGSIM preflight diagnostic: output_dir=%s camera_count=%s timestamp=%s",
                        message.get("output_dir"), message.get("camera_count"), message.get("timestamp"),
                    )
                    continue

                privileged_info = None
                if isinstance(message, (list, tuple)):
                    if len(message) >= 3:
                        obs, info, privileged_info = message[:3]
                    elif len(message) == 2:
                        obs, info = message
                    else:
                        raise ValueError(f"Unexpected planner message length: {len(message)}")
                else:
                    raise ValueError(f"Unexpected planner message type: {type(message)}")

                info_history.append(dict(info))
                while len(info_history) < service.history_frames:
                    info_history.appendleft(dict(info_history[0]))

                result = service.process(
                    obs, info, list(info_history), {"privileged_info": privileged_info}
                )
                write_plan_file(
                    plan_pipe_writer,
                    (
                        np.asarray(result.proposals, dtype=np.float32),
                        np.asarray(result.scores, dtype=np.float32),
                    ),
                )
            except Exception:
                log.error("Planner frame failed")
                log.error(traceback.format_exc())
                try:
                    write_plan_file(plan_pipe_writer, None)
                except Exception:
                    log.error("Failed to notify HUGSIM about planner failure")
                return 1
    finally:
        for closer in (obs_pipe_reader.close, plan_pipe_writer.close, service.finalize):
            try:
                closer()
            except Exception:
                log.exception("Error during planner shutdown")

    return 0
