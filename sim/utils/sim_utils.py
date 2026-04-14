import json
import math
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SCR
from scene.cameras import Camera
from sim.ilqr.lqr import plan2control
from omegaconf import OmegaConf

def rt2pose(r, t, degrees=False):
    pose = np.eye(4)
    pose[:3, :3] = SCR.from_euler('XYZ', r, degrees=degrees).as_matrix()
    pose[:3, 3] = t
    return pose

def pose2rt(pose, degrees=False):
    r = SCR.from_matrix(pose[:3, :3]).as_euler('XYZ', degrees=degrees)
    t = pose[:3, 3]
    return r, t
    
def _load_scene_camera_rig(scene_meta_path, camera_names):
    with open(scene_meta_path, "r") as f:
        meta_data = json.load(f)

    first_frame_poses = {}
    first_frame_intrinsics = {}
    for frame in meta_data["frames"]:
        rgb_path = frame["rgb_path"].replace("\\", "/")
        cam_name = rgb_path.split("/")[2]
        if cam_name in camera_names and cam_name not in first_frame_poses:
            first_frame_poses[cam_name] = np.array(frame["camtoworld"])
            K = np.array(frame["intrinsics"])
            first_frame_intrinsics[cam_name] = {
                "H": int(frame["height"]),
                "W": int(frame["width"]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "fovx": float(focal2fov(K[0, 0], frame["width"])),
                "fovy": float(focal2fov(K[1, 1], frame["height"])),
            }
        if len(first_frame_poses) == len(camera_names):
            break

    if "CAM_FRONT" not in first_frame_poses:
        raise KeyError("Scene metadata does not include CAM_FRONT pose")

    missing = [cam_name for cam_name in camera_names if cam_name not in first_frame_poses]
    if missing:
        raise KeyError(f"Scene metadata missing cameras: {missing}")

    front_pose = first_frame_poses["CAM_FRONT"]
    cam_params = {}
    for cam_name in camera_names:
        cam_params[cam_name] = {
            "intrinsic": first_frame_intrinsics[cam_name],
            # Simulation state is anchored to the front-camera trajectory in ground_param.pkl.
            "front2cam": np.linalg.inv(front_pose) @ first_frame_poses[cam_name],
        }
    return cam_params


def load_camera_cfg(cfg, model_path=None):
    cam_params = {}
    cams = OmegaConf.to_container(cfg.cams, resolve=True)
    camera_names = list(cams.keys())
    for cam_name, cam in cams.items():
        v2c = rt2pose(cam['extrinsics']['v2c_rot'], cam['extrinsics']['v2c_trans'], degrees=True)
        l2c = rt2pose(cam['extrinsics']['l2c_rot'], cam['extrinsics']['l2c_trans'], degrees=True)
        cam_intrin = cam['intrinsics']
        cam_intrin['fovx'] = cam_intrin['fovx'] / 180.0 * np.pi
        cam_intrin['fovy'] = cam_intrin['fovy'] / 180.0 * np.pi
        cam_params[cam_name] = {'intrinsic': cam_intrin, 'v2c': v2c, 'l2c': l2c}

    if model_path is not None:
        scene_meta_path = os.path.join(model_path, "meta_data.json")
        if os.path.exists(scene_meta_path):
            scene_cam_params = _load_scene_camera_rig(scene_meta_path, camera_names)
            for cam_name, params in scene_cam_params.items():
                cam_params[cam_name]['intrinsic'] = params['intrinsic']
                cam_params[cam_name]['front2cam'] = params['front2cam']
        
    rect_mat = np.eye(4)
    if 'cam_rect' in cfg:
        rect_mat[:3, :3] = SCR.from_euler('XYZ', cfg.cam_rect.rot, degrees=True).as_matrix()
        rect_mat[:3, 3] = np.array(cfg.cam_rect.trans)
        
    return cam_params, OmegaConf.to_container(cfg.cam_align, resolve=True), rect_mat

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def create_cam(intrinsic, c2w):
    fovx, fovy = intrinsic['fovx'], intrinsic['fovy']
    h, w = intrinsic['H'], intrinsic['W']
    K = np.eye(4)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrinsic['cx'], intrinsic['cy']
    cam = Camera(K=K, c2w=c2w, width=w, height=h,
                image=np.zeros((h, w, 3)), image_name='')
    return cam


def get_camera_c2w(cam_params, ego_pose, cam_name, cam_rect=None):
    params = cam_params[cam_name]
    if 'front2cam' in params:
        return ego_pose @ params['front2cam']

    if cam_rect is None:
        cam_rect = np.eye(4)
    v2front = cam_params['CAM_FRONT']['v2c']
    v2c = params['v2c']
    c2front = v2front @ np.linalg.inv(v2c) @ cam_rect
    return ego_pose @ c2front


def local_traj_to_world(plan_traj, ego_pose, ground_height_fn=None):
    points_world = []
    origin = ego_pose[:3, 3]
    right_dir = ego_pose[:3, 0]
    forward_dir = ego_pose[:3, 2]

    for right, forward in plan_traj:
        point = origin + right * right_dir + forward * forward_dir
        if ground_height_fn is not None:
            point[1] = ground_height_fn(point[0], point[2])
        points_world.append(point)

    if not points_world:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points_world, dtype=np.float32)


def local_plan_to_front_world(plan_traj, front_c2w, front_v2c, include_origin=False, forward_offset=0.0):
    """
    Convert a HUGSIM local plan [right, forward] into world points using the
    front-camera pose at plan-generation time.

    RAP/HUGSIM plans live in a ground-aligned local frame centered at the ego
    rear axle with axes [right, forward, up]. The rendered front camera is
    translated forward/up from that origin, so we reuse the same axis mapping as
    the RAP adapter and recover the camera offset from the front `v2c`
    extrinsic.
    """
    plan_traj = np.asarray(plan_traj, dtype=np.float32)
    if len(plan_traj) == 0 and not include_origin:
        return np.zeros((0, 3), dtype=np.float32)

    local_xyz = np.zeros((len(plan_traj), 3), dtype=np.float32)
    if len(plan_traj) > 0:
        local_xyz[:, :2] = plan_traj
        local_xyz[:, 1] += float(forward_offset)

    if include_origin:
        local_xyz = np.concatenate(
            [np.array([[0.0, float(forward_offset), 0.0]], dtype=np.float32), local_xyz],
            axis=0,
        )

    camera_in_vehicle = np.linalg.inv(np.asarray(front_v2c, dtype=np.float32))[:3, 3]
    # `v2c` translation is expressed in the vehicle frame [forward, left, up].
    # HUGSIM local plans use [right, forward, up].
    camera_in_local = np.array(
        [-camera_in_vehicle[1], camera_in_vehicle[0], camera_in_vehicle[2]],
        dtype=np.float32,
    )
    local_to_front_cam = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    points_cam = (local_to_front_cam @ (local_xyz - camera_in_local).T).T
    homogeneous = np.concatenate(
        [points_cam, np.ones((len(points_cam), 1), dtype=np.float32)],
        axis=1,
    )
    points_world = (np.asarray(front_c2w, dtype=np.float32) @ homogeneous.T).T[:, :3]
    return points_world.astype(np.float32)


def get_camera_matrix(intrinsic):
    K = np.eye(4, dtype=np.float32)
    K[0, 0] = fov2focal(intrinsic['fovx'], intrinsic['W'])
    K[1, 1] = fov2focal(intrinsic['fovy'], intrinsic['H'])
    K[0, 2] = intrinsic['cx']
    K[1, 2] = intrinsic['cy']
    return K


def world_points_to_ego_lidar(points_world, ego_pose):
    if len(points_world) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    homogeneous = np.concatenate(
        [np.asarray(points_world, dtype=np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    ego_inv = np.linalg.inv(ego_pose)
    ego_points = (ego_inv @ homogeneous.T).T[:, :3]
    lidar_points = np.stack(
        [ego_points[:, 0], ego_points[:, 2], ego_points[:, 1]],
        axis=1,
    )
    return lidar_points.astype(np.float32)


def project_world_points_to_image(points_world, intrinsic, c2w):
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    K = get_camera_matrix(intrinsic)
    homogeneous = np.concatenate(
        [np.asarray(points_world, dtype=np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]

    depth = camera_points[:, 2]
    valid = depth > 1e-4
    pixels = np.zeros((len(points_world), 2), dtype=np.int32)
    if np.any(valid):
        image_points = (K[:3, :3] @ camera_points[valid].T).T
        pixels_valid = image_points[:, :2] / np.clip(image_points[:, 2:3], 1e-3, None)
        pixels[valid] = np.round(pixels_valid).astype(np.int32)
    return pixels, valid


def draw_projected_polyline_camera_clipped(image, points_world, intrinsic, c2w, color, thickness=3, near=1e-3):
    """
    Draw a 3D polyline after clipping each segment against the camera near
    plane. This preserves segments that cross from behind the camera into the
    visible frustum instead of dropping them entirely.
    """
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) < 2:
        return image

    K = get_camera_matrix(intrinsic)[:3, :3]
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (w2c @ homogeneous.T).T[:, :3]

    h, w = image.shape[:2]
    rect = (0, 0, w, h)

    def project_point(point_cam):
        projected = K @ point_cam
        uv = projected[:2] / np.clip(projected[2], near, None)
        return tuple(np.round(uv).astype(np.int32))

    for p0, p1 in zip(camera_points[:-1], camera_points[1:]):
        z0, z1 = float(p0[2]), float(p1[2])
        if z0 <= near and z1 <= near:
            continue

        q0 = p0.copy()
        q1 = p1.copy()
        if z0 <= near:
            alpha = (near - z0) / max(z1 - z0, 1e-6)
            q0 = p0 + alpha * (p1 - p0)
            q0[2] = near
        if z1 <= near:
            alpha = (near - z1) / max(z0 - z1, 1e-6)
            q1 = p1 + alpha * (p0 - p1)
            q1[2] = near

        pixel0 = project_point(q0)
        pixel1 = project_point(q1)
        ok, clipped_p0, clipped_p1 = cv2.clipLine(rect, pixel0, pixel1)
        if not ok:
            continue
        cv2.line(image, clipped_p0, clipped_p1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return image


def project_world_points_renderer_uv(points_world, intrinsic, c2w):
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    K = get_camera_matrix(intrinsic)
    homogeneous = np.concatenate(
        [np.asarray(points_world, dtype=np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (np.linalg.inv(c2w) @ homogeneous.T).T[:, :3]

    depth = camera_points[:, 2]
    valid = depth > 1e-4
    pixels = np.zeros((len(points_world), 2), dtype=np.int32)
    if np.any(valid):
        image_points = (K[:3, :3] @ camera_points[valid].T).T
        renderer_uv = image_points[:, [1, 0]] / np.clip(image_points[:, 2:3], 1e-3, None)
        pixels[valid] = np.round(renderer_uv).astype(np.int32)
    return pixels, valid


def project_lidar_points_to_image(points_lidar, intrinsic, l2c):
    if len(points_lidar) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    K = get_camera_matrix(intrinsic)
    homogeneous = np.concatenate(
        [np.asarray(points_lidar, dtype=np.float32), np.ones((len(points_lidar), 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (np.asarray(l2c, dtype=np.float32) @ homogeneous.T).T[:, :3]

    depth = camera_points[:, 2]
    valid = depth > 1e-4
    pixels = np.zeros((len(points_lidar), 2), dtype=np.int32)
    if np.any(valid):
        image_points = (K[:3, :3] @ camera_points[valid].T).T
        pixels_valid = image_points[:, :2] / np.clip(image_points[:, 2:3], 1e-3, None)
        pixels[valid] = np.round(pixels_valid).astype(np.int32)
    return pixels, valid


def snap_lidar_points_to_image_ground(points_lidar, depth, semantic, intrinsic, l2c, sample_step=2, max_snap_dist=2.5):
    points_lidar = np.asarray(points_lidar, dtype=np.float32)
    if len(points_lidar) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    depth = np.asarray(depth, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.uint8)
    h, w = depth.shape

    ys = np.arange(0, h, sample_step, dtype=np.int32)
    xs = np.arange(0, w, sample_step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sampled_depth = depth[grid_y, grid_x]
    sampled_semantic = semantic[grid_y, grid_x]
    valid_mask = np.isfinite(sampled_depth) & (sampled_depth > 1e-3) & (sampled_semantic == 0)
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    fx = fov2focal(intrinsic['fovx'], intrinsic['W'])
    fy = fov2focal(intrinsic['fovy'], intrinsic['H'])
    cx = intrinsic['cx']
    cy = intrinsic['cy']

    px = grid_x[valid_mask].astype(np.float32)
    py = grid_y[valid_mask].astype(np.float32)
    z = sampled_depth[valid_mask]
    cam_x = (px - cx) / fx * z
    cam_y = (py - cy) / fy * z
    cam_points = np.stack([cam_x, cam_y, z], axis=1)
    homogeneous = np.concatenate([cam_points, np.ones((len(cam_points), 1), dtype=np.float32)], axis=1)
    lidar_points = (np.linalg.inv(np.asarray(l2c, dtype=np.float32)) @ homogeneous.T).T[:, :3]

    tree = cKDTree(lidar_points[:, :2])
    distances, indices = tree.query(points_lidar[:, :2], k=1)

    snapped_pixels = np.stack([px[indices], py[indices]], axis=1)
    snapped_valid = distances <= max_snap_dist
    return np.round(snapped_pixels).astype(np.int32), snapped_valid


def draw_projected_polyline(image, pixels, valid_mask, color, thickness=3):
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 2:
        return image

    h, w = image.shape[:2]
    rect = (0, 0, w, h)
    for start_idx, end_idx in zip(valid_indices[:-1], valid_indices[1:]):
        if end_idx != start_idx + 1:
            continue
        p0 = tuple(int(v) for v in pixels[start_idx])
        p1 = tuple(int(v) for v in pixels[end_idx])
        ok, clipped_p0, clipped_p1 = cv2.clipLine(rect, p0, p1)
        if not ok:
            continue
        cv2.line(image, clipped_p0, clipped_p1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return image


def resample_polyline(points_world, spacing=0.25):
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) < 2:
        return points_world

    segment_lengths = np.linalg.norm(np.diff(points_world[:, [0, 2]], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length < 1e-4:
        return points_world[:1]

    sample_distances = np.arange(0.0, total_length + spacing, spacing, dtype=np.float32)
    sample_distances[-1] = min(sample_distances[-1], total_length)

    resampled = []
    seg_idx = 0
    for dist in sample_distances:
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < dist:
            seg_idx += 1
        if seg_idx + 1 >= len(points_world):
            resampled.append(points_world[-1])
            continue
        seg_start = cumulative[seg_idx]
        seg_end = cumulative[seg_idx + 1]
        denom = max(seg_end - seg_start, 1e-6)
        alpha = float((dist - seg_start) / denom)
        point = (1.0 - alpha) * points_world[seg_idx] + alpha * points_world[seg_idx + 1]
        resampled.append(point)
    return np.asarray(resampled, dtype=np.float32)


def build_ground_ribbon(points_world, width):
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    left_points, right_points = [], []
    half_width = width * 0.5
    for idx in range(len(points_world)):
        prev_idx = max(0, idx - 1)
        next_idx = min(len(points_world) - 1, idx + 1)
        tangent = points_world[next_idx, [0, 2]] - points_world[prev_idx, [0, 2]]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            continue
        tangent = tangent / tangent_norm
        lateral = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        center = points_world[idx].copy()
        left = center.copy()
        right = center.copy()
        left[[0, 2]] += lateral * half_width
        right[[0, 2]] -= lateral * half_width
        left_points.append(left)
        right_points.append(right)

    if len(left_points) < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.asarray(left_points, dtype=np.float32), np.asarray(right_points, dtype=np.float32)


def draw_projected_ribbon(image, left_pixels, right_pixels, left_valid, right_valid, color, alpha=0.35):
    overlay = image.copy()
    for idx in range(min(len(left_pixels), len(right_pixels)) - 1):
        quad_valid = left_valid[idx] and right_valid[idx] and left_valid[idx + 1] and right_valid[idx + 1]
        if not quad_valid:
            continue
        quad = np.array(
            [
                left_pixels[idx],
                right_pixels[idx],
                right_pixels[idx + 1],
                left_pixels[idx + 1],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(overlay, quad, color=color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    return image


def snap_world_points_to_image_ground(points_world, depth, semantic, intrinsic, c2w, sample_step=2, max_snap_dist=2.5):
    points_world = np.asarray(points_world, dtype=np.float32)
    if len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    depth = np.asarray(depth, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.uint8)
    h, w = depth.shape

    ys = np.arange(0, h, sample_step, dtype=np.int32)
    xs = np.arange(0, w, sample_step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sampled_depth = depth[grid_y, grid_x]
    sampled_semantic = semantic[grid_y, grid_x]
    valid_mask = np.isfinite(sampled_depth) & (sampled_depth > 1e-3) & (sampled_semantic == 0)
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    fx = fov2focal(intrinsic['fovx'], intrinsic['W'])
    fy = fov2focal(intrinsic['fovy'], intrinsic['H'])
    cx = intrinsic['cx']
    cy = intrinsic['cy']

    px = grid_x[valid_mask].astype(np.float32)
    py = grid_y[valid_mask].astype(np.float32)
    z = sampled_depth[valid_mask]
    cam_x = (px - cx) / fx * z
    cam_y = (py - cy) / fy * z
    cam_points = np.stack([cam_x, cam_y, z], axis=1)
    homogeneous = np.concatenate([cam_points, np.ones((len(cam_points), 1), dtype=np.float32)], axis=1)
    world_points = (c2w @ homogeneous.T).T[:, :3]

    tree = cKDTree(world_points[:, [0, 2]])
    distances, indices = tree.query(points_world[:, [0, 2]], k=1)

    snapped_pixels = np.stack([px[indices], py[indices]], axis=1)
    snapped_valid = distances <= max_snap_dist
    return np.round(snapped_pixels).astype(np.int32), snapped_valid

def traj2control(plan_traj, info):
    """
        The input plan trajectory is under lidar coordinates
        x to right, y to forward and z to upward
    """
    plan_traj_stats = np.zeros((plan_traj.shape[0]+1, 5))
    plan_traj_stats[1:, :2] = plan_traj[:, [1,0]]
    prev_a, prev_b = 0.0, 0.0
    for i, (a, b) in enumerate(plan_traj):
        # plan2control expects heading in the swapped [forward, right] frame.
        rot = np.arctan2(a - prev_a, b - prev_b)
        rot = np.where(rot > np.pi/2, rot - np.pi, rot)
        rot = np.where(rot < -np.pi/2, rot + np.pi, rot)
        plan_traj_stats[i+1, 2] = rot
        prev_a, prev_b = a, b
    curr_stat = np.array(
        [0.0, 0.0, 0.0, info['ego_velo'], info['ego_steer']]
    )
    acc, steer_rate = plan2control(plan_traj_stats, curr_stat)
    return acc, steer_rate

def dense_cam_poses(cam_poses, cmds):
    
    for i in range(5):
        dense_poses = []
        dense_cmds = []
        for i in range(cam_poses.shape[0]-1):
            cam1 = cam_poses[i]
            cam2 = cam_poses[i+1]
            dense_poses.append(cam1)
            dense_cmds.append(cmds[i])
            if np.linalg.norm(cam1[:3, 3]-cam2[:3, 3]) > 0.1:
                euler1 = SCR.from_matrix(cam1[:3, :3]).as_euler("XYZ")
                euler2 = SCR.from_matrix(cam2[:3, :3]).as_euler("XYZ")
                interp_euler = (euler1 + euler2) / 2
                interp_trans = (cam1[:3, 3] + cam2[:3, 3]) / 2
                interp_pose = np.eye(4)
                interp_pose[:3, :3] = SCR.from_euler("XYZ", interp_euler).as_matrix()
                interp_pose[:3, 3] = interp_trans
                dense_poses.append(interp_pose)
                dense_cmds.append(cmds[i])
        dense_poses.append(cam_poses[-1])
        dense_poses = np.stack(dense_poses)
        cam_poses = dense_poses
        cmds = dense_cmds
        
    return cam_poses, cmds

def traj_transform_to_global(traj, ego_box):
        """
        Transform trajectory from ego-centeric frame to global frame
        """
        ego_x, ego_y, _, _, _, _, ego_yaw = ego_box
        global_points = [
            (
                ego_x
                + px * math.cos(ego_yaw)
                - py * math.sin(ego_yaw),
                ego_y
                + px * math.sin(ego_yaw)
                + py * math.cos(ego_yaw),
            )
            for px, py in traj
        ]
        global_trajs = []
        for i in range(1, len(global_points)):
            x1, y1 = global_points[i - 1]
            x2, y2 = global_points[i]
            dx, dy = x2 - x1, y2 - y1
            # distance = math.sqrt(dx**2 + dy**2)
            yaw = math.atan2(dy, dx)
            global_trajs.append((x1, y1, yaw))
        return global_trajs
        
