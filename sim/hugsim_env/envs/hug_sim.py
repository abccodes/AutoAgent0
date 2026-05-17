import torch
import numpy as np
from copy import deepcopy
import gymnasium
from gymnasium import spaces
from copy import deepcopy
from sim.utils.sim_utils import create_cam, rt2pose, pose2rt, load_camera_cfg, dense_cam_poses
from scipy.spatial.transform import Rotation as SCR
from sim.utils.score_calculator import create_rectangle, bg_collision_det
import os
import pickle
from sim.utils.plan import planner, UnifiedMap
from omegaconf import OmegaConf
import math
from gaussian_renderer import GaussianModel
from scene.obj_model import ObjModel
from gaussian_renderer import render
import open3d as o3d


def fg_collision_det(ego_box, objs):
    ego_x, ego_y, _, ego_w, ego_l, ego_h, ego_yaw = ego_box
    ego_poly = create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)
    for obs in objs:
        obs_x, obs_y, _, obs_w, obs_l, _, obs_yaw = obs
        obs_poly = create_rectangle(
            obs_x, obs_y, obs_w, obs_l, obs_yaw)
        if ego_poly.intersects(obs_poly):
            return True
    return False

class HUGSimEnv(gymnasium.Env):
    def __init__(self, cfg, output):
        super().__init__()
        
        plan_list = cfg.scenario.plan_list
        for control_param in plan_list:
            control_param[5] = os.path.join(cfg.base.realcar_path, control_param[5])

        # read ground infos
        with open(os.path.join(cfg.model_path, 'ground_param.pkl'), 'rb') as f:
            #numpy.ndarray, float, list
            cam_poses, cam_heights, commands = pickle.load(f)
            cam_poses, commands = dense_cam_poses(cam_poses, commands)
            self.ground_model = (cam_poses, cam_heights, commands)

        if cfg.scenario.load_HD_map:
            unified_map = UnifiedMap(cfg.base.HD_map.path, cfg.base.HD_map.version, cfg.scenario.scene_name)
        else:
            unified_map = None
        
        self.kinematic = OmegaConf.to_container(cfg.kinematic)
        self.kinematic['min_steer'] = -math.radians(cfg.kinematic.min_steer)
        self.kinematic['max_steer'] = math.radians(cfg.kinematic.max_steer)
        self.kinematic['start_vr']= np.array(cfg.scenario.start_euler) / 180 * np.pi
        self.kinematic['start_vab'] = np.array(cfg.scenario.start_ab)
        self.kinematic['start_velo'] = cfg.scenario.start_velo
        self.kinematic['start_steer'] = cfg.scenario.start_steer

        self.gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)

        """
        plan_list: a, b, height, yaw, v, model_path, controller, params
        Yaw is based on ego car's orientation. 0 means same direction as ego. 
        Right is positive and left is negative.
        """
        self.planner = planner(plan_list, scene_path=cfg.model_path, unified_map=unified_map, ground=self.ground_model, dt=cfg.kinematic.dt)
        
        (model_params, iteration) = torch.load(os.path.join(cfg.model_path, "scene.pth"), weights_only=False)
        self.gaussians.restore(model_params, None)
        
        dynamic_gaussians = {}
        for plan_id in self.planner.ckpts.keys():
            dynamic_gaussians[plan_id] = ObjModel(cfg.model.sh_degree, feat_mutable=False)
            (model_params, iteration) = torch.load(self.planner.ckpts[plan_id], weights_only=False)
            model_params = list(model_params)
            dynamic_gaussians[plan_id].restore(model_params, None)
            
        semantic_idx = torch.argmax(self.gaussians.get_full_3D_features, dim=-1, keepdim=True)
        ground_xyz = self.gaussians.get_full_xyz[(semantic_idx == 0)[:, 0]].detach().cpu().numpy()
        scene_xyz = self.gaussians.get_full_xyz[((semantic_idx > 1) & (semantic_idx != 10))[:, 0]].detach().cpu().numpy()
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'ground.ply'), ground_pcd)
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'scene.ply'), scene_pcd)

        if cfg.scenario.load_HD_map:
            self.planner.update_agent_route()
        
        self.cam_params, cam_align, self.cam_rect = load_camera_cfg(cfg.camera, cfg.model_path)
        
        self.ego_verts = np.array([[0.5, 0, 0.5], [0.5, 0, -0.5], [0.5, 1.0,  0.5], [0.5, 1.0, -0.5],
                    [-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 1.0, -0.5], [-0.5, 1.0, 0.5]])
        self.whl = np.array([1.6, 1.5, 3.0])
        self.ego_verts *= self.whl
        self.data_type = cfg.data_type

        self.action_space = spaces.Dict(
            {
                "steer_rate": spaces.Box(self.kinematic['min_steer'], self.kinematic['max_steer'], dtype=float),
                "acc": spaces.Box(self.kinematic['min_acc'], self.kinematic['max_acc'], dtype=float)
            }
        )
        self.observation_space = spaces.Dict(
            {
                'rgb': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=255, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W'], 3), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'semantic': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=50, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'depth': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=1000, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.float32
                    ) for cam_name, params in self.cam_params.items()
                }),
            }
        )
        self.fric = self.kinematic['fric']

        self.start_vr = self.kinematic['start_vr']
        self.start_vab = self.kinematic['start_vab']
        self.start_velo = self.kinematic['start_velo']
        self.vr = deepcopy(self.kinematic['start_vr'])
        self.vab = deepcopy(self.kinematic['start_vab'])
        self.velo = deepcopy(self.kinematic['start_velo'])
        self.steer = deepcopy(self.kinematic['start_steer'])
        self.dt = self.kinematic['dt']

        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.render_fn = render
        self.render_kwargs = {
            "pc": self.gaussians,
            "bg_color": torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            "dynamic_gaussians": dynamic_gaussians,
            "unicycles": {} # dummy input, unicycle planner is used for unicycle models
        }
        gaussians = self.gaussians
        semantic_idx = torch.argmax(gaussians.get_3D_features, dim=-1, keepdim=True)
        opacities = gaussians.get_opacity[:, 0]
        mask = ((semantic_idx > 1) & (semantic_idx != 10))[:, 0] & (opacities > 0.8)
        self.points = gaussians.get_xyz[mask]

        self.last_accel = 0
        self.last_steer_rate = 0

        self.timestamp = 0
    
    def ground_height(self, u, v):
        cam_poses, cam_height, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - u)**2 + (cam_poses[:, 2, 3] - v)**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        nearest_c2w = cam_poses[nearest_cam_idx]

        nearest_w2c = np.linalg.inv(nearest_c2w)
        uhv_local = nearest_w2c[:3, :3] @ np.array([u, 0, v]) + nearest_w2c[:3, 3]
        uhv_local[1] = 0
        uhv_world = nearest_c2w[:3, :3] @ uhv_local + nearest_c2w[:3, 3]
        
        return uhv_world[1]
    
    @property
    def route_completion(self):
        cam_poses, _, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - self.vab[0])**2 + (cam_poses[:, 2, 3] - self.vab[1])**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        return (nearest_cam_idx + 1) / (cam_poses.shape[0] * 0.9), cam_dist[nearest_cam_idx]
        

    @property
    def vt(self):
        vt = np.zeros(3)
        vt[[0, 2]] = self.vab
        vt[1] = self.ground_height(self.vab[0], self.vab[1])
        return vt
    
    @property
    def ego(self):
        return rt2pose(self.vr, self.vt)
    
    @property
    def ego_state(self):
        return torch.tensor([self.vab[0], self.vab[1], self.vr[1], self.velo])
    
    @property
    def ego_box(self):
        return [self.vt[2], -self.vt[0], -self.vt[1], self.whl[0], self.whl[2], self.whl[1], -self.vr[1]]

    @property
    def objs_list(self):
        obj_boxes = []
        objs = self.render_kwargs['planning'][0]
        for obj_id, obj_b2w in objs.items():
            yaw = SCR.from_matrix(obj_b2w[:3, :3].detach().cpu().numpy()).as_euler('YXZ')[0]
            # X, Y, Z in IMU, w, l, h
            wlh = self.planner.wlhs[obj_id]
            obj_boxes.append([obj_b2w[2, 3].item(), -obj_b2w[0, 3].item(), -obj_b2w[1, 3].item(), wlh[0], wlh[1], wlh[2], -yaw-0.5*np.pi])
        return obj_boxes

    def get_agent_privileged_info(self):
        """DEVELOPMENT ONLY helper: return privileged per-agent info for non-ego agents.

        Returns a list of dicts with keys mirroring the ego info layout where possible:
          - agent_id: str
          - agent_pos_world: [x, y, z]
          - agent_pos_imu: [x, y, z] (same ordering as `objs_list` entries)
          - agent_rot: [rx, ry, rz] Euler (XYZ) from agent pose matrix
          - agent_velo: scalar speed (if available from planner.stats)
          - agent_heading: float (radians, converted similar to `objs_list` convention)
          - agent_steer: steering angle if available, else None
          - accel: acceleration if available, else None
          - steer_rate: steering-rate if available, else None
          - b2w: full 4x4 transform (ndarray)
          - route: planned route array (if planner.route populated), else None

        Notes:
        - This is intentionally permissive: many fields may be None depending on
          the planner/controller implementation. Use `agent_id` to track agents
          over time.
        - The IMU ordering matches `objs_list` (x= b2w[2,3], y = -b2w[0,3], z = -b2w[1,3]).
        """
        out = []
        planning = self.render_kwargs.get('planning', [None, {}])
        b2ws = {}
        if planning and isinstance(planning, (list, tuple)):
            try:
                b2ws = planning[0] or {}
            except Exception:
                b2ws = {}
        elif isinstance(planning, dict):
            b2ws = planning

        for agent_id, b2w in b2ws.items():
            try:
                # tensors may be on GPU
                if hasattr(b2w, 'detach'):
                    b2w_np = b2w.detach().cpu().numpy()
                else:
                    b2w_np = np.asarray(b2w)
            except Exception:
                b2w_np = np.asarray(b2w)

            # world pose r,t (pose2rt returns euler XYZ and translation)
            try:
                agent_rot_euler, agent_trans = pose2rt(b2w_np)
                agent_rot = list(np.asarray(agent_rot_euler).tolist())
                agent_pos_world = [float(agent_trans[0]), float(agent_trans[1]), float(agent_trans[2])]
            except Exception:
                agent_rot = None
                agent_pos_world = [float(b2w_np[0, 3]), float(b2w_np[1, 3]), float(b2w_np[2, 3])]

            # IMU ordering used elsewhere in env (objs_list)
            imu_x = float(b2w_np[2, 3])
            imu_y = float(-b2w_np[0, 3])
            imu_z = float(-b2w_np[1, 3])

            # derive heading consistent with objs_list
            try:
                yaw = SCR.from_matrix(b2w_np[:3, :3]).as_euler('YXZ')[0]
                heading = float(-yaw - 0.5 * np.pi)
            except Exception:
                heading = None

            stat = self.planner.stats.get(agent_id) if hasattr(self, 'planner') else None
            speed = None
            if stat is not None:
                try:
                    speed = float(stat[4])
                except Exception:
                    speed = None

            # approximate velocity vector in world frame (vx, vy, vz)
            if speed is not None and heading is not None:
                vx = float(speed * np.sin(heading))
                vy = float(speed * np.cos(heading))
                vz = 0.0
                agent_velo = speed
                agent_vel_vec = [vx, vy, vz]
            else:
                agent_velo = None
                agent_vel_vec = None

            # attempt to extract controller-level inputs if present
            accel = None
            steer_rate = None
            agent_steer = None
            if hasattr(self, 'planner'):
                controller = self.planner.controller.get(agent_id) if hasattr(self.planner, 'controller') else None
                if controller is not None:
                    # common controller state names may vary; probe defensively
                    if hasattr(controller, 'last_accel'):
                        try:
                            accel = float(controller.last_accel)
                        except Exception:
                            accel = None
                    if hasattr(controller, 'last_steer_rate'):
                        try:
                            steer_rate = float(controller.last_steer_rate)
                        except Exception:
                            steer_rate = None
                    if hasattr(controller, 'steer'):
                        try:
                            agent_steer = float(controller.steer)
                        except Exception:
                            agent_steer = None

            route = None
            if hasattr(self, 'planner') and hasattr(self.planner, 'route'):
                try:
                    r = self.planner.route.get(agent_id)
                    if r is not None:
                        route = r.detach().cpu().numpy().tolist() if hasattr(r, 'detach') else (r.tolist() if hasattr(r, 'tolist') else r)
                except Exception:
                    route = None

            ckpt = None
            if hasattr(self, 'planner') and hasattr(self.planner, 'ckpts'):
                ckpt = self.planner.ckpts.get(agent_id)

            out.append({
                'agent_id': agent_id,
                'b2w': b2w_np,
                'agent_pos_world': agent_pos_world,
                'agent_pos_imu': [imu_x, imu_y, imu_z],
                'agent_rot': agent_rot,
                'agent_velo': agent_velo,
                'agent_vel_vec': agent_vel_vec,
                'agent_heading': heading,
                'agent_steer': agent_steer,
                'accel': accel,
                'steer_rate': steer_rate,
                'route': route,
                'ckpt': ckpt,
            })

        return out

    def _get_obs(self):
        rgbs, semantics, depths = {}, {}, {}
        v2front = self.cam_params['CAM_FRONT']["v2c"]
        for cam_name, params in self.cam_params.items():
            intrinsic, v2c = params['intrinsic'], params['v2c']
            c2front = v2front @ np.linalg.inv(v2c) @ self.cam_rect
            c2w = self.ego @ c2front
            viewpoint = create_cam(intrinsic, c2w)
            with torch.no_grad():
                render_pkg = self.render_fn(viewpoint=viewpoint, prev_viewpoint=None, **self.render_kwargs)
            rgb = (torch.permute(render_pkg['render'].clamp(0, 1), (1,2,0)).detach().cpu().numpy() * 255).astype(np.uint8)
            smt = torch.argmax(render_pkg['feats'], dim=0).detach().cpu().numpy().astype(np.uint8)
            depth = render_pkg['depth'][0].detach().cpu().numpy()
            if (self.data_type == 'waymo' or self.data_type == 'kitti360') and 'BACK' in cam_name:
                rgbs[cam_name] = np.zeros_like(rgb)
                semantics[cam_name] = np.zeros_like(smt)
                depths[cam_name] = np.zeros_like(depth)
            else:
                rgbs[cam_name] = rgb
                semantics[cam_name] = smt
                depths[cam_name] = depth

        return {
                'rgb': rgbs, 
                'semantic': semantics,
                'depth': depths,
                }
    
    def _get_info(self):
        wego_r, wego_t = pose2rt(self.ego)
        cam_poses, _, commands = self.ground_model
        dist = np.sum((cam_poses[:, :3, 3] - self.vt) ** 2, axis=-1)
        nearest_cam_idx = np.argmin(dist)
        command = commands[nearest_cam_idx]
        return {
            'ego_pos'  : wego_t.tolist(),
            'ego_rot'  : wego_r.tolist(),
            'ego_velo' : self.velo,
            'ego_steer': self.steer,
            'accelerate': self.last_accel,
            'steer_rate': self.last_steer_rate,
            'timestamp': self.timestamp,
            'command': command,
            'ego_box': self.ego_box,
            'obj_boxes': self.objs_list,
            'cam_params': self.cam_params,
            # 'ego_verts': verts,
        }
    
    def reset(self, seed=None, options=None):
        self.vr = deepcopy(self.start_vr)
        self.vab = deepcopy(self.start_vab)
        self.velo = deepcopy(self.start_velo)
        self.timestamp = 0

        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)

        observation = self._get_obs()
        info = self._get_info()
        # Keep reset() signature Gym-compliant: do not return privileged info.
        # Privileged info can be fetched by callers via `env.get_agent_privileged_info()`.
        return observation, info
    
    def step(self, action):
        self.timestamp += self.dt
        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)
        steer_rate = float(np.clip(action['steer_rate'], self.kinematic['min_steer'], self.kinematic['max_steer']))
        acc = float(np.clip(action['acc'], self.kinematic['min_acc'], self.kinematic['max_acc']))
        self.last_steer_rate, self.last_accel = steer_rate, acc
        L = self.kinematic['Lr'] + self.kinematic['Lf']
        self.velo += acc * self.dt
        self.steer += steer_rate * self.dt
        theta = self.vr[1]
        # print(theta / np.pi * 180, self.steer / np.pi * 180)
        self.vab[0] = self.vab[0] + self.velo * np.sin(theta) * self.dt
        self.vab[1] = self.vab[1] + self.velo * np.cos(theta) * self.dt
        self.vr[1] = theta + self.velo * np.tan(self.steer) / L * self.dt

        terminated = False
        reward = 0
        verts = (self.ego[:3, :3] @ self.ego_verts.T).T + self.ego[:3, 3]
        verts = torch.from_numpy(verts.astype(np.float32)).cuda()
        
        bg_collision = bg_collision_det(self.points, verts)
        if bg_collision:
            terminated = True
            print('Collision with background')
            reward = -100

        fg_collision = fg_collision_det(self.ego_box, self.objs_list)
        if fg_collision:
            terminated = True
            print('Collision with foreground')
            reward = -100

        rc, dist = self.route_completion
        if dist > 10:
            terminated=True
            print('Far from preset trajectory')
            reward = -50
            
        if rc >= 1:
            terminated = True
            print('Complete')
            reward = 1000

        observation = self._get_obs()
        info = self._get_info()
        info['rc'] = rc
        info['collision'] = bg_collision or fg_collision

        # Keep step() compliant with gymnasium: return a 4- or 5-tuple.
        # Privileged info may be fetched by callers via
        # `env.unwrapped.get_agent_privileged_info()` when needed.
        return observation, reward, terminated, False, info
