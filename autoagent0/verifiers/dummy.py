"""A placeholder verifier that always approves the trajectory."""
from typing import Any, Optional

from autoagent0.verifiers.base import BaseVerifier

'''
class DummyVerifier(BaseVerifier):
    """Always returns a high score, so the loop never triggers recovery.

    Stand-in until a real trajectory verifier is implemented. It ignores the
    trajectory contents and unconditionally approves it.
    """

    HIGH_SCORE = 1.0

    def score(self, trajectory: Any, current_info: Optional[Any] = None) -> float:
'''

from sim.utils.score_calculator import ScoreCalculator
from sim.utils.sim_utils import traj_transform_to_global
import open3d as o3d
import numpy as np
import torch
import os

class TTCVerifier(BaseVerifier):
    def __init__(self, scene_ply_path: str, timestep: float = 0.5):
        scene_xyz = np.asarray(o3d.io.read_point_cloud(scene_ply_path).points)
        # same coordinate transform as hugsim_evaluate (ply -> IMU frame)
        scene_xyz = np.stack([scene_xyz[:, 2], -scene_xyz[:, 0], -scene_xyz[:, 1]], axis=1)
        self._scene_xyz = torch.from_numpy(scene_xyz).float().cuda()
        self._timestep = timestep
        self._calc = ScoreCalculator(data={})

    def score(self, trajectory, current_info=None) -> float:
        if trajectory is None or current_info is None:
            return 1.0

        plan = np.asarray(trajectory, dtype=np.float64)
        if len(plan) < 2:
            return 1.0

        ego_box = current_info["ego_box"]
        obs_lists = [current_info["obj_boxes"]]

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

        return self._calc._calculate_time_to_collision(
            ego_box, planned_traj, obs_lists, self._scene_xyz, self._timestep,
        )