"""Live PDMS-style trajectory verifier.

The composition is the two-tier PDMS form

    PDMS = NC * DAC * (5*TTC + 2*Comfort) / 7

where NC and DAC are multiplicative hard gates and TTC/Comfort are soft terms
in a weighted average.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from autoagent0.verifiers.base import BaseVerifier
from sim.utils.score_calculator import ScoreCalculator
from sim.utils.sim_utils import traj_transform_to_global
import open3d as o3d
import numpy as np
import torch


HD_SCORE_SOFT_WEIGHTS = {"ttc": 5.0, "c": 2.0}


def _project_obstacles(obs_list, obs_vels, t):
    """Return obs_list with each (x, y) position shifted forward by obs_vels * t seconds."""
    projected = []
    for i, obs in enumerate(obs_list):
        x, y, z, w, l, h, yaw = obs
        vx, vy = obs_vels[i] if i < len(obs_vels) else (0.0, 0.0)
        projected.append([x + vx * t, y + vy * t, z, w, l, h, yaw])
    return projected


@dataclass
class VerificationResult:
    gates: Dict[str, float]
    soft: Dict[str, float]
    weights: Dict[str, float] = field(default_factory=lambda: dict(HD_SCORE_SOFT_WEIGHTS))

    def gate_product(self) -> float:
        product = 1.0
        for value in self.gates.values():
            product *= value
        return product

    def soft_weighted_average(self) -> float:
        denom = sum(self.weights[k] for k in self.soft)
        if denom <= 0:
            return 1.0
        return sum(self.weights[k] * self.soft[k] for k in self.soft) / denom

    def pdms(self) -> float:
        return self.gate_product() * self.soft_weighted_average()

    @property
    def vetoed(self) -> bool:
        """True iff any hard gate is fully tripped (value 0.0)."""
        return any(value == 0.0 for value in self.gates.values())


class PDMSVerifier(BaseVerifier):
    """PDMS = NC * DAC * weighted_avg(TTC, Comfort).

    Hard gates (NC, DAC) and soft terms (TTC, Comfort) are all computed from live
    state at verification time: ``current_info`` (ego_box, obj_boxes, obj_vels)
    plus the static scene/ground point clouds loaded once at construction.
    """
    
    def __init__(self, scene_ply_path: str, ground_ply_path: str, timestep: float = 0.5):
        scene_xyz = np.asarray(o3d.io.read_point_cloud(scene_ply_path).points)
        scene_xyz = np.stack([scene_xyz[:, 2], -scene_xyz[:, 0], -scene_xyz[:, 1]], axis=1)
        self._scene_xyz = torch.from_numpy(scene_xyz).float().cuda()

        ground_xyz = np.asarray(o3d.io.read_point_cloud(ground_ply_path).points)
        self._ground_xy = np.stack([ground_xyz[:, 2], -ground_xyz[:, 0]], axis=1)

        self._timestep = timestep
        self._calc = ScoreCalculator(data={})
        self.last_result: Optional[VerificationResult] = None

    def _build_global_plan(self, plan, ego_box):
        """Convert a HUGSIM-local plan [T,2] to a global (x,y,yaw) trajectory [T,3].

        Prepends the current ego state as the first waypoint, and flattens yaw for
        near-stationary plans (same guards as the original TTC port).
        """
        imu_plan_traj = plan[:, [1, 0]].copy()
        imu_plan_traj[:, 1] *= -1
        global_pts = traj_transform_to_global(imu_plan_traj, ego_box)

        ego_x, ego_y, ego_yaw = ego_box[0], ego_box[1], ego_box[6]
        planned_traj = np.concatenate(
            [np.array([[ego_x, ego_y, ego_yaw]]), np.array(global_pts)],
            axis=0,
        )
        if np.linalg.norm(planned_traj[-1, :2] - planned_traj[0, :2]) < 1.0:
            planned_traj[:, 2] = planned_traj[0, 2]
        return planned_traj

    def _ttc_velocity_proj(self, ego_box, planned_traj, obs_list, obs_vels, timestep):
        """TTC check replicating the NAVSIM formulation but with constant-velocity
        obstacle projection that is *time-aligned* with the ego sample it is tested against.

        For each horizon t in {0.5 s, 1.0 s} the whole ego trajectory is probed forward
        by ego_velocity * t, so the ego sample at trajectory index ``idx`` represents the
        ego at absolute time ``idx * timestep + t``.
        """
        n_points = planned_traj.shape[0]
        for t in [0.5, 1.0]:
            velocities = np.diff(planned_traj[:, :2], axis=0) / timestep
            velocities = np.vstack([velocities[0], velocities])
            new_traj = planned_traj.copy()
            new_traj[:, :2] += velocities * t
            obs_lists = [
                _project_obstacles(obs_list, obs_vels, idx * timestep + t)
                for idx in range(n_points)
            ]
            if self._calc._calculate_no_collision(
                ego_box, new_traj, obs_lists, self._scene_xyz
            ) == 0.0:
                return 0.0
        return 1.0

    def _all_pass_result(self) -> VerificationResult:
        return VerificationResult(gates={"nc": 1.0, "dac": 1.0}, soft={"ttc": 1.0, "c": 1.0})

    def verify(self, trajectory, current_info=None) -> VerificationResult:
        if trajectory is None or current_info is None:
            return self._all_pass_result()

        plan = np.asarray(trajectory, dtype=np.float64)
        if len(plan) < 2:
            return self._all_pass_result()

        ego_box = current_info["ego_box"]
        obs_list = current_info["obj_boxes"]
        obs_vels = current_info.get("obj_vels", [[0.0, 0.0]] * len(obs_list))

        planned_traj = self._build_global_plan(plan, ego_box)

        nc = self._calc._calculate_no_collision(
            ego_box, planned_traj, [obs_list], self._scene_xyz
        )

        dac = self._calc._calculate_drivable_area_compliance(
            self._ground_xy, planned_traj, ego_box[3], ego_box[4]
        )

        ttc = self._ttc_velocity_proj(
            ego_box, planned_traj, obs_list, obs_vels, self._timestep
        )

        ego_frame_traj = self._calc.transform_to_ego_frame(planned_traj, ego_box)
        comfort = self._calc._calculate_is_comfortable(ego_frame_traj, self._timestep)

        result = VerificationResult(
            gates={"nc": nc, "dac": dac},
            soft={"ttc": ttc, "c": comfort},
        )
        self.last_result = result
        return result

    def score(self, trajectory, current_info=None) -> float:
        return self.verify(trajectory, current_info).pdms()
