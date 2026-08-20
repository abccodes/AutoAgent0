from __future__ import annotations

import os
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from autoagent0.calibration.features import (
    raw_observation_features,
    score_features,
    temporal_change,
)
from autoagent0.calibration.labels import compute_future_failure_labels
from autoagent0.calibration.labels import annotate_frames_with_score_details
from autoagent0.calibration.labels import planned_object_overlap_evidence
from autoagent0.calibration.offline import make_group_folds
from autoagent0.calibration.offline import event_metrics
from autoagent0.adapters.hugsim.results import attach_execution_outcome
from autoagent0.config import AutoAgent0Config, build_prefixed_autoagent0_env
from autoagent0.reproducibility import apply_benchmark_seed
from autoagent0.scorer.agent_schemas import FrameUncertainty
from autoagent0.scorer.planner_selection import LearnedPlannerSelector
from autoagent0.scorer.selection_strategies.recovery import RecoveryStrategy
from autoagent0.scorer.vlm_selector import VLMSelectorConfig
from scripts.analyze_uncertainty_repeatability import _log_duration
from scripts.analyze_uncertainty_repeatability import _trajectory_comparison


class FeatureTest(unittest.TestCase):
    def test_equal_scores_have_max_entropy_and_zero_margin(self) -> None:
        features = score_features([2.0, 2.0])
        self.assertAlmostEqual(features["score_entropy"], 1.0)
        self.assertAlmostEqual(features["effective_candidate_count"], 2.0)
        self.assertAlmostEqual(features["score_margin_normalized"], 0.0)

    def test_temporal_change_uses_world_alignment(self) -> None:
        first = np.asarray([[0.0, 0.0], [0.0, 1.0]])
        change, previous_world = temporal_change(first, np.eye(4), None)
        self.assertIsNone(change)
        shifted_pose = np.eye(4)
        shifted_pose[0, 3] = 1.0
        same_world_path = np.asarray([[-1.0, 0.0], [-1.0, 1.0]])
        change, _ = temporal_change(same_world_path, shifted_pose, previous_world)
        self.assertAlmostEqual(change, 0.0)


class RepeatabilityAnalyzerTest(unittest.TestCase):
    def test_benchmark_seed_repeats_python_and_numpy_streams(self) -> None:
        with patch.dict(os.environ, {"HUGSIM_BENCHMARK_SEED": "17"}, clear=False):
            self.assertEqual(apply_benchmark_seed(), 17)
            first = (random.random(), float(np.random.rand()))
            self.assertEqual(apply_benchmark_seed(), 17)
            second = (random.random(), float(np.random.rand()))
        self.assertEqual(first, second)

    def test_log_duration_uses_fifo_loop_not_model_initialization(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "planner.log"
            path.write_text(
                "2026-08-19 10:00:00,000 INFO Starting planner\n"
                "2026-08-19 10:00:10,000 INFO Opened persistent scene FIFOs\n"
                "2026-08-19 10:00:25,500 INFO Received shutdown signal\n",
                encoding="utf-8",
            )
            self.assertAlmostEqual(_log_duration(path), 15.5)

    def test_trajectory_comparison_reports_first_divergent_frame(self) -> None:
        def frame(endpoint: float) -> dict:
            return {"planner_debug": {"local_plan": [[0.0, 0.0], [0.0, endpoint]]}}

        comparison = _trajectory_comparison(
            [frame(1.0), frame(1.0), frame(2.0)],
            [frame(1.0), frame(1.5), frame(2.5)],
            tolerance=1e-6,
        )
        self.assertEqual(comparison["first_plan_divergence_frame"], 1)
        self.assertGreater(comparison["mean_selected_plan_distance_m"], 0.0)


class LabelTest(unittest.TestCase):
    def test_decomposed_future_labels_and_lead_time(self) -> None:
        frames = [
            {"collision": False, "score_details": {"nc": [1.0], "dac": [1.0]}},
            {"collision": False, "score_details": {"nc": [0.5], "dac": [1.0]}},
            {"collision": True, "score_details": {"nc": [1.0], "dac": [1.0]}},
        ]
        labels = compute_future_failure_labels(frames, horizon_steps=2)
        self.assertEqual(labels["future_unsafe"], [1, 1, 1])
        self.assertEqual(labels["future_collision"], [0, 1, 1])
        self.assertEqual(labels["future_nc_failure"], [1, 1, 0])
        self.assertEqual(labels["steps_to_unsafe"], [1, 0, 0])

    def test_post_step_collision_overrides_pre_step_snapshot(self) -> None:
        frame = {
            "collision": False,
            "score_details": {"nc": [1.0], "dac": [1.0]},
        }
        attach_execution_outcome(
            frame,
            {
                "timestamp": 0.25,
                "collision": True,
                "background_collision": True,
                "termination_reason": "collision",
                "rc": 0.4,
            },
            reward=-100,
            terminated=True,
            truncated=False,
        )
        labels = compute_future_failure_labels([frame], horizon_steps=1)
        self.assertEqual(labels["collision_now"], [1])
        self.assertTrue(frame["execution_outcome"]["terminated"])
        self.assertFalse(frame["execution_outcome"]["runner_timeout"])
        self.assertEqual(frame["execution_outcome"]["termination_reason"], "collision")

    def test_evaluator_nc_metadata_is_not_coerced_to_float(self) -> None:
        frames = [{"time_stamp": 0.0}]
        annotate_frames_with_score_details(
            frames,
            {
                0.0: {
                    "nc": 0.0,
                    "dac": 1.0,
                    "nc_failure_type": "background",
                    "nc_failure_step": 4,
                }
            },
        )
        self.assertEqual(frames[0]["score_details"]["nc"], [0.0])
        self.assertEqual(frames[0]["score_details"]["nc_failure_type"], "background")
        self.assertEqual(frames[0]["score_details"]["nc_failure_step"], 4)

    def test_historical_object_overlap_replay(self) -> None:
        ego_box = [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]
        frame = {
            "time_stamp": 0.0,
            "ego_box": ego_box,
            "obj_boxes": [[4.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]],
            "obj_names": ["car"],
            "planned_traj": {
                "traj": [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                "timestep": 0.5,
            },
        }
        self.assertTrue(planned_object_overlap_evidence([frame], 0))
        frame["obj_boxes"] = []
        frame["obj_names"] = []
        self.assertFalse(planned_object_overlap_evidence([frame], 0))


class SplitTest(unittest.TestCase):
    def test_scene_variants_never_cross_fold_boundary(self) -> None:
        rows = []
        for scene in range(6):
            for variant in range(2):
                for frame in range(4):
                    rows.append(
                        {
                            "scene_group": f"scene-{scene:04d}",
                            "future_unsafe": int((scene + variant + frame) % 3 == 0),
                        }
                    )
        df = pd.DataFrame(rows)
        folds = make_group_folds(df, label_col="future_unsafe", n_splits=3, seed=7)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertFalse(set(fold.train_groups) & set(fold.test_groups))
            self.assertEqual(len(fold.train) + len(fold.test), len(df))


class EventTest(unittest.TestCase):
    def test_contiguous_failures_are_one_event_and_alert_must_precede_onset(self) -> None:
        df = pd.DataFrame(
            {
                "run_id": ["run"] * 8,
                "scene_group": ["scene"] * 8,
                "frame_idx": list(range(8)),
                "unsafe_now": [0, 0, 0, 1, 1, 0, 0, 0],
            }
        )
        alerts = np.asarray([0, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
        metrics, rows = event_metrics(df, alerts, lookback_steps=3, frame_dt_sec=0.5)
        self.assertEqual(metrics["event_count"], 1)
        self.assertEqual(metrics["detected_event_count"], 1)
        self.assertEqual(metrics["false_alert_episodes"], 1)
        self.assertEqual(rows[0]["lead_steps"], 2)

    def test_short_safe_gap_does_not_create_a_second_event(self) -> None:
        df = pd.DataFrame(
            {
                "run_id": ["run"] * 7,
                "scene_group": ["scene"] * 7,
                "frame_idx": list(range(7)),
                "unsafe_now": [0, 1, 1, 0, 0, 1, 0],
            }
        )
        metrics, _ = event_metrics(
            df, np.zeros(7, dtype=bool), lookback_steps=2, merge_gap_steps=2
        )
        self.assertEqual(metrics["event_count"], 1)


class PassiveRuntimeTest(unittest.TestCase):
    def test_vlm_disabled_observe_mode_preserves_argmax_and_records_telemetry(self) -> None:
        selector = LearnedPlannerSelector(
            vlm_selector=SimpleNamespace(),
            autoagent0_cfg=AutoAgent0Config(
                enabled=True,
                uncertainty_enabled=True,
                uncertainty_policy_mode="observe",
            ),
            vlm_cfg=VLMSelectorConfig(
                enabled=False,
                candidate_limit=1,
                carry_previous_enabled=False,
                include_default_candidates=False,
            ),
            rule_based_merge_cfg=SimpleNamespace(enabled=False, topk=5),
            current_source_name="drivor_current",
            plain_source="drivor_argmax",
        )
        proposals = np.asarray(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 2.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        scores = np.asarray([0.2, 0.9, 0.5], dtype=np.float32)

        payload = selector.select(
            proposals=proposals,
            scores=scores,
            obs={},
            info={
                "ego_rot": [0.0, 0.0, 0.0],
                "ego_pos": [0.0, 0.0, 0.0],
                "timestamp": 0.0,
            },
            info_history=[],
        )

        self.assertEqual(payload["selected_idx"], 1)
        self.assertEqual(payload["selected_source"], "drivor_argmax")
        self.assertAlmostEqual(payload["selected_score"], float(scores[1]))
        np.testing.assert_array_equal(payload["selected_plan"], proposals[1])
        observation = payload["autoagent0_uncertainty_observation"]
        self.assertEqual(observation["policy_mode"], "observe")
        self.assertFalse(observation["affects_candidate_allocation"])
        self.assertEqual(observation["learned_proposals"], proposals.tolist())
        self.assertEqual(observation["learned_scores"], scores.tolist())
        self.assertEqual(observation["baseline_proposal"]["proposal_index"], 1)
        self.assertEqual(
            observation["uncertainty"]["metadata"]["intra"]["member_count"], 3
        )
        self.assertGreater(observation["uncertainty"]["intra_learned_m"], 0.0)

    def test_observe_mode_never_exposes_uncertainty_to_allocator(self) -> None:
        uncertainty = FrameUncertainty(0.1, 0.2, 3, "rule_based_fallback", {})
        observe = RecoveryStrategy(
            SimpleNamespace(autoagent0_cfg=AutoAgent0Config(uncertainty_policy_mode="observe"))
        )
        active = RecoveryStrategy(
            SimpleNamespace(autoagent0_cfg=AutoAgent0Config(uncertainty_policy_mode="active"))
        )
        self.assertIsNone(observe._routing_uncertainty(uncertainty))
        self.assertIs(active._routing_uncertainty(uncertainty), uncertainty)
        observation = observe._uncertainty_observation_debug(
            proposals=np.zeros((2, 3, 2)),
            scores=np.asarray([0.8, 0.2]),
            rule_based_candidate_rows=[],
            default_row={
                "source": "learned",
                "proposal_index": 0,
                "proposal_score": 0.8,
                "local_plan": np.zeros((3, 2)),
            },
            frame_uncertainty=uncertainty,
        )
        self.assertFalse(observation["affects_candidate_allocation"])
        self.assertEqual(len(observation["learned_proposals"]), 2)
        raw_features = raw_observation_features(
            {"autoagent0_uncertainty_observation": observation}
        )
        self.assertEqual(raw_features["raw_candidate_count"], 2.0)
        self.assertAlmostEqual(raw_features["raw_score_entropy"], 0.5270653)
        env = build_prefixed_autoagent0_env({"uncertainty_policy_mode": "observe"})
        self.assertEqual(env["AUTOAGENT0_UNCERTAINTY_POLICY_MODE"], "observe")


if __name__ == "__main__":
    unittest.main()
