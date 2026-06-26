"""Live TTC verifier with constant-velocity obstacle projection."""
from typing import Any, Optional

from autoagent0.verifiers.base import BaseVerifier

from sim.utils.score_calculator import ScoreCalculator
from sim.utils.sim_utils import traj_transform_to_global
import open3d as o3d
import numpy as np
import torch
import os


def _project_obstacles(obs_list, obs_vels, t):
    """Return obs_list with each (x, y) position shifted forward by obs_vels * t seconds."""
    projected = []
    for i, obs in enumerate(obs_list):
        x, y, z, w, l, h, yaw = obs
        vx, vy = obs_vels[i] if i < len(obs_vels) else (0.0, 0.0)
        projected.append([x + vx * t, y + vy * t, z, w, l, h, yaw])
    return projected


class TTCVerifier(BaseVerifier):
    def __init__(self, scene_ply_path: str, timestep: float = 0.5):
        scene_xyz = np.asarray(o3d.io.read_point_cloud(scene_ply_path).points)
        # same coordinate transform as hugsim_evaluate (ply -> IMU frame)
        scene_xyz = np.stack([scene_xyz[:, 2], -scene_xyz[:, 0], -scene_xyz[:, 1]], axis=1)
        self._scene_xyz = torch.from_numpy(scene_xyz).float().cuda()
        self._timestep = timestep
        self._calc = ScoreCalculator(data={})

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

    def score(self, trajectory, current_info=None) -> float:
        if trajectory is None or current_info is None:
            return 1.0

        plan = np.asarray(trajectory, dtype=np.float64)
        if len(plan) < 2:
            return 1.0

        ego_box = current_info["ego_box"]
        obs_list = current_info["obj_boxes"]
        obs_vels = current_info.get("obj_vels", [[0.0, 0.0]] * len(obs_list))

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

        return self._ttc_velocity_proj(
            ego_box, planned_traj, obs_list, obs_vels, self._timestep,
        )