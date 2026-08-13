from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autoagent0.calibration.features import (
    raw_observation_features,
    score_features,
    temporal_change,
)
from autoagent0.calibration.labels import compute_future_failure_labels
from autoagent0.calibration.offline import make_group_folds
from autoagent0.calibration.offline import event_metrics
from autoagent0.config import AutoAgent0Config, build_prefixed_autoagent0_env
from autoagent0.scorer.agent_schemas import FrameUncertainty
from autoagent0.scorer.selection_strategies.recovery import RecoveryStrategy


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
