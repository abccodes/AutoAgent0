import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import hugsim_env
from argparse import ArgumentParser
from sim.utils.sim_utils import (
    draw_projected_polyline_camera_clipped,
    draw_projected_polyline,
    get_camera_c2w,
    local_plan_to_front_world,
    project_world_points_to_image,
    resample_polyline,
    rt2pose,
    traj2control,
    traj_transform_to_global,
)
import pickle
import json
import pickle
import struct
from sim.utils.launch_ad import launch, check_alive
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
import numpy as np
import cv2
from moviepy import ImageSequenceClip

VIDEO_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'],
]
FRONT_CAM_NAME = 'CAM_FRONT'
REFERENCE_COLOR = (255, 230, 0)
PLAN_COLOR = (0, 120, 255)
TOPK_BASE_COLORS = [
    (0, 120, 255),
    (0, 180, 255),
    (0, 220, 200),
    (60, 220, 120),
    (120, 220, 60),
    (200, 220, 40),
    (255, 200, 0),
    (255, 160, 0),
    (255, 120, 0),
    (255, 80, 0),
]
REFERENCE_FORWARD_OFFSET_M = 4.0
PLAN_VIS_FORWARD_OFFSET_M = 4.5
VIS_PLAN_MIN_PATH_M = 0.5
VIS_PLAN_HOLD_FRAMES = 10**9
PLAN_REPLAN_EVERY_STEPS = 2

def _resize_for_video(image, target_height):
    if image.shape[0] == target_height:
        return image
    width = max(1, int(round(image.shape[1] * (target_height / image.shape[0]))))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_LINEAR)


def _pad_row_for_video(row, target_width):
    if row.shape[1] == target_width:
        return row
    pad_width = target_width - row.shape[1]
    return np.pad(row, ((0, 0), (0, pad_width), (0, 0)), mode='constant')


def to_video(observations, rollout_frames, output_path):
    frames = []
    if not observations:
        return

    target_height = max(
        obs[cam_name].shape[0]
        for obs in observations
        for row in VIDEO_LAYOUT
        for cam_name in row
    )

    for frame_idx, obs in enumerate(observations):
        row1 = np.concatenate(
            [_resize_for_video(obs[cam_name], target_height) for cam_name in VIDEO_LAYOUT[0]],
            axis=1,
        )
        row2 = np.concatenate(
            [_resize_for_video(obs[cam_name], target_height) for cam_name in VIDEO_LAYOUT[1]],
            axis=1,
        )
        target_width = max(row1.shape[1], row2.shape[1])
        row1 = _pad_row_for_video(row1, target_width)
        row2 = _pad_row_for_video(row2, target_width)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


def write_pipe_message(pipe_path, payload_obj):
    payload = pickle.dumps(payload_obj, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pipe_path, "wb") as pipe:
        pipe.write(struct.pack("<Q", len(payload)))
        pipe.write(payload)


def read_pipe_message(pipe_path):
    with open(pipe_path, "rb") as pipe:
        header = pipe.read(8)
        if len(header) != 8:
            raise EOFError(f"Incomplete pipe header from {pipe_path}")
        payload_size = struct.unpack("<Q", header)[0]
        payload = bytearray()
        while len(payload) < payload_size:
            chunk = pipe.read(payload_size - len(payload))
            if not chunk:
                raise EOFError(f"Incomplete pipe payload from {pipe_path}")
            payload.extend(chunk)
    return pickle.loads(payload)


def _select_reference_segment(reference_poses, current_c2w, lookahead=25):
    if reference_poses is None or len(reference_poses) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    reference_xyz = reference_poses[:, :3, 3]
    current_xyz = current_c2w[:3, 3]
    nearest_idx = int(np.argmin(np.sum((reference_xyz - current_xyz[None]) ** 2, axis=1)))
    end_idx = min(len(reference_xyz), nearest_idx + lookahead)
    return np.asarray(reference_xyz[nearest_idx:end_idx], dtype=np.float32)


def _select_reference_pose_segment(reference_poses, current_c2w, lookahead=25):
    if reference_poses is None or len(reference_poses) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)

    reference_xyz = reference_poses[:, :3, 3]
    current_xyz = current_c2w[:3, 3]
    nearest_idx = int(np.argmin(np.sum((reference_xyz - current_xyz[None]) ** 2, axis=1)))
    end_idx = min(len(reference_xyz), nearest_idx + lookahead)
    return np.asarray(reference_poses[nearest_idx:end_idx], dtype=np.float32)


def _draw_projected_points(image, pixels, valid_mask, color, radius=4):
    h, w = image.shape[:2]
    valid_indices = np.flatnonzero(valid_mask)
    for idx in valid_indices:
        px, py = pixels[idx]
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(image, (int(px), int(py)), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_first_visible_segment_marker(image, pixels, valid_mask, color, radius=6):
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < 2:
        return

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
        anchor = clipped_p0
        cv2.circle(image, anchor, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        return


def _count_in_frame(pixels, valid_mask, image_shape):
    h, w = image_shape[:2]
    if len(pixels) == 0:
        return 0
    in_frame = (
        valid_mask
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    return int(np.count_nonzero(in_frame))


def _trajectory_path_length(points_xy):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points_xy, axis=0), axis=1).sum())


def _build_reference_worldline(reference_poses, ground_height_fn, forward_offset=REFERENCE_FORWARD_OFFSET_M):
    if reference_poses is None or len(reference_poses) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    reference_world = np.asarray(reference_poses[:, :3, 3], dtype=np.float32).copy()
    forward_dirs = np.asarray(reference_poses[:, :3, 2], dtype=np.float32).copy()
    forward_norm = np.linalg.norm(forward_dirs, axis=1, keepdims=True)
    forward_dirs = forward_dirs / np.clip(forward_norm, 1e-6, None)

    reference_world[:, 0] += forward_offset * forward_dirs[:, 0]
    reference_world[:, 2] += forward_offset * forward_dirs[:, 2]
    reference_world[:, 1] = np.array(
        [ground_height_fn(point[0], point[2]) for point in reference_world],
        dtype=np.float32,
    )
    return resample_polyline(reference_world, spacing=0.5)


def _render_front_overlay(front_image, current_obs, current_info, env, plan_traj, plan_origin_pose):
    ego_pose = rt2pose(np.asarray(current_info['ego_rot']), np.asarray(current_info['ego_pos']))
    cam_params = current_info['cam_params']
    intrinsic = cam_params[FRONT_CAM_NAME]['intrinsic']
    front_c2w = get_camera_c2w(cam_params, ego_pose, FRONT_CAM_NAME, env.unwrapped.cam_rect)

    overlay = front_image.copy()

    reference_poses = _select_reference_pose_segment(env.unwrapped.ground_model[0], front_c2w, lookahead=40)
    reference_world = _build_reference_worldline(
        reference_poses,
        env.unwrapped.scene_ground_height,
    )
    ref_pixels, ref_valid = project_world_points_to_image(reference_world, intrinsic, front_c2w)
    draw_projected_polyline(overlay, ref_pixels, ref_valid, color=REFERENCE_COLOR, thickness=4)
    _draw_projected_points(overlay, ref_pixels, ref_valid, color=REFERENCE_COLOR, radius=4)

    def draw_candidate(candidate_plan, color, thickness):
        if candidate_plan is None or len(candidate_plan) == 0 or plan_origin_pose is None:
            return
        plan_local = np.asarray(candidate_plan, dtype=np.float32)
        plan_origin_front_c2w = get_camera_c2w(
            cam_params,
            np.asarray(plan_origin_pose, dtype=np.float32),
            FRONT_CAM_NAME,
            env.unwrapped.cam_rect,
        )
        plan_world_line = local_plan_to_front_world(
            plan_local,
            plan_origin_front_c2w,
            cam_params[FRONT_CAM_NAME]['v2c'],
            include_origin=True,
            forward_offset=PLAN_VIS_FORWARD_OFFSET_M,
        )
        plan_world_dense = resample_polyline(plan_world_line, spacing=0.08)
        draw_projected_polyline_camera_clipped(
            overlay,
            plan_world_dense,
            intrinsic,
            front_c2w,
            color=color,
            thickness=thickness,
        )

    candidate_plans = current_info.get('overlay_candidate_plans')
    if candidate_plans:
        for rank, candidate_plan in enumerate(candidate_plans):
            base_color = TOPK_BASE_COLORS[min(rank, len(TOPK_BASE_COLORS) - 1)]
            if rank >= len(TOPK_BASE_COLORS):
                decay = min(rank - len(TOPK_BASE_COLORS) + 1, 8)
                fade = max(0.35, 1.0 - 0.08 * decay)
                base_color = tuple(int(channel * fade) for channel in base_color)
            draw_candidate(candidate_plan, base_color, 4 if rank == 0 else 2)
    elif plan_traj is not None and len(plan_traj) > 0:
        draw_candidate(plan_traj, PLAN_COLOR, 4)

    return overlay


def _select_overlay_plan(plan_traj, plan_path_length, last_valid_plan, stale_frames):
    if plan_traj is not None and len(plan_traj) > 0 and plan_path_length >= VIS_PLAN_MIN_PATH_M:
        return np.asarray(plan_traj, dtype=np.float32), 0, False

    if last_valid_plan is not None and stale_frames < VIS_PLAN_HOLD_FRAMES:
        return last_valid_plan.copy(), stale_frames + 1, True

    return None, stale_frames, False


def create_gym_env(cfg, output):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    obs, info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    for pipe_path in (obs_pipe, plan_pipe):
        if os.path.exists(pipe_path):
            os.remove(pipe_path)
        os.mkfifo(pipe_path)
    print('Ready for simulation')

    obs, info = None, None
    last_valid_overlay_plan = None
    last_valid_overlay_pose = None
    last_valid_overlay_plan_stale_frames = 0
    current_plan_traj = None
    current_plan_origin_pose = None
    current_topk_plans = None
    current_topk_scores = None
    current_selected_idx = None
    current_selected_source = None
    current_vlm_selected_idx = None
    current_vlm_confidence = None
    current_vlm_reasoning = None
    current_vlm_elapsed_sec = None
    current_vlm_error = None
    overlay_front_dir = os.path.join(output, 'overlay_front')
    os.makedirs(overlay_front_dir, exist_ok=True)
    for name in os.listdir(overlay_front_dir):
        if name.endswith('.jpg'):
            os.remove(os.path.join(overlay_front_dir, name))
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
        current_obs, current_info = obs, info
        infos_save.append(current_info)

        print('ego pose', current_info['ego_pos'])

        should_replan = (cnt % PLAN_REPLAN_EVERY_STEPS == 0) or (current_plan_traj is None)
        if should_replan:
            write_pipe_message(obs_pipe, (current_obs, current_info))
            plan_payload = read_pipe_message(plan_pipe)
            current_topk_plans = None
            current_topk_scores = None
            current_selected_idx = None
            current_selected_source = None
            current_vlm_selected_idx = None
            current_vlm_confidence = None
            current_vlm_reasoning = None
            current_vlm_elapsed_sec = None
            current_vlm_error = None
            if isinstance(plan_payload, dict):
                current_plan_traj = plan_payload.get('selected_plan')
                current_topk_plans = plan_payload.get('topk_plans')
                current_topk_scores = plan_payload.get('topk_scores')
                current_selected_idx = plan_payload.get('selected_idx')
                current_selected_source = plan_payload.get('selected_source')
                current_vlm_selected_idx = plan_payload.get('vlm_selected_idx')
                current_vlm_confidence = plan_payload.get('vlm_confidence')
                current_vlm_reasoning = plan_payload.get('vlm_reasoning')
                current_vlm_elapsed_sec = plan_payload.get('vlm_elapsed_sec')
                current_vlm_error = plan_payload.get('vlm_error')
            else:
                current_plan_traj = plan_payload
            current_plan_origin_pose = rt2pose(
                np.asarray(current_info['ego_rot']),
                np.asarray(current_info['ego_pos']),
            )
        plan_traj = current_plan_traj

        if plan_traj is not None:
            imu_plan_traj = plan_traj[:, [1, 0]]
            imu_plan_traj[:, 1] *= -1
            global_traj = traj_transform_to_global(imu_plan_traj, current_info['ego_box'])
            current_rc = current_info.get('rc', env.unwrapped.route_completion[0])
            local_plan = np.asarray(plan_traj, dtype=np.float32)
            plan_path_length = _trajectory_path_length(local_plan)
            plan_endpoint_distance = float(np.linalg.norm(local_plan[-1])) if len(local_plan) > 0 else 0.0
            plan_min_step = float(np.linalg.norm(np.diff(local_plan, axis=0), axis=1).min()) if len(local_plan) > 1 else 0.0
            plan_max_step = float(np.linalg.norm(np.diff(local_plan, axis=0), axis=1).max()) if len(local_plan) > 1 else 0.0
            overlay_plan, last_valid_overlay_plan_stale_frames, overlay_plan_held = _select_overlay_plan(
                local_plan,
                plan_path_length,
                last_valid_overlay_plan,
                last_valid_overlay_plan_stale_frames,
            )
            overlay_plan_origin_pose = None
            if plan_path_length >= VIS_PLAN_MIN_PATH_M:
                last_valid_overlay_plan = local_plan.copy()
                last_valid_overlay_pose = current_plan_origin_pose.copy() if current_plan_origin_pose is not None else None
                overlay_plan_origin_pose = last_valid_overlay_pose
            elif overlay_plan is not None:
                overlay_plan_origin_pose = last_valid_overlay_pose
            save_data['frames'].append({
                'time_stamp': current_info['timestamp'],
                'is_key_frame': True,
                'ego_box': current_info['ego_box'],
                'ego_pos': current_info['ego_pos'],
                'ego_rot': current_info['ego_rot'],
                'obj_boxes': current_info['obj_boxes'],
                'obj_names': ['car' for _ in current_info['obj_boxes']],
                'planned_traj': {
                    'traj': global_traj,
                    'timestep': 0.5
                },
                'planner_debug': {
                    'local_plan': local_plan.tolist(),
                    'path_length_m': plan_path_length,
                    'endpoint_distance_m': plan_endpoint_distance,
                    'min_step_m': plan_min_step,
                    'max_step_m': plan_max_step,
                    'selected_idx': current_selected_idx,
                    'selected_source': current_selected_source,
                    'topk_scores': current_topk_scores,
                    'topk_count': 0 if current_topk_plans is None else int(len(current_topk_plans)),
                    'vlm_selected_idx': current_vlm_selected_idx,
                    'vlm_confidence': current_vlm_confidence,
                    'vlm_reasoning': current_vlm_reasoning,
                    'vlm_elapsed_sec': current_vlm_elapsed_sec,
                    'vlm_error': current_vlm_error,
                    'overlay_plan_held': bool(overlay_plan_held),
                    'overlay_plan_stale_frames': int(last_valid_overlay_plan_stale_frames),
                },
                'collision': current_info.get('collision', False),
                'rc': current_rc
            })

            acc, steer_rate = traj2control(plan_traj, current_info)
            save_data['frames'][-1]['planner_debug']['acc_cmd'] = float(acc)
            save_data['frames'][-1]['planner_debug']['steer_rate_cmd'] = float(steer_rate)

            vis_rgb = {
                cam_name: image.copy()
                for cam_name, image in current_obs['rgb'].items()
            }
            overlay_info = dict(current_info)
            overlay_info['overlay_candidate_plans'] = current_topk_plans
            vis_rgb[FRONT_CAM_NAME] = _render_front_overlay(
                current_obs['rgb'][FRONT_CAM_NAME],
                current_obs,
                overlay_info,
                env,
                overlay_plan,
                overlay_plan_origin_pose,
            )
            observations_save.append(vis_rgb)
            cv2.imwrite(
                os.path.join(overlay_front_dir, f'{len(observations_save) - 1:04d}.jpg'),
                cv2.cvtColor(vis_rgb[FRONT_CAM_NAME], cv2.COLOR_RGB2BGR),
            )

            action = {'acc': acc, 'steer_rate': steer_rate}
            obs, reward, terminated, truncated, info = env.step(action)
            cnt += 1
            done = terminated or truncated or cnt > 400

        else:
            done = True
            break

    write_pipe_message(obs_pipe, 'Done')

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)
        
    try:
        to_video(
            observations_save,
            save_data['frames'],
            os.path.join(output, 'video.mp4'),
        )
    except Exception as exc:
        print(f"Skipping video export due to error: {exc}")
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)
    
    ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument("--planner_path", type=str, default="")
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    planner_path = args.planner_path
    if not planner_path:
        inferred_planner_path = os.path.join("configs", "planners", f"{args.ad}.yaml")
        if os.path.exists(inferred_planner_path):
            planner_path = inferred_planner_path
    planner_config = OmegaConf.load(planner_path) if planner_path else OmegaConf.create()
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config},
        {"planner": planner_config},
    )
    cfg.base.output_dir = cfg.base.output_dir + args.ad

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    if args.ad == 'uniad':
        ad_path = cfg.base.uniad_path
    elif args.ad == 'vad':
        ad_path = cfg.base.vad_path
    elif args.ad == 'ltf':
        ad_path = cfg.base.ltf_path
    elif args.ad == 'rap':
        ad_path = cfg.planner.rap.launch_path
    else:
        raise NotImplementedError

    extra_env = {}
    if args.ad == 'rap':
        extra_env = {
            'RAP_REPO_ROOT': cfg.planner.rap.get('repo_root', ''),
            'RAP_CHECKPOINT': cfg.planner.rap.get('checkpoint', ''),
            'RAP_PYTHON_BIN': cfg.planner.rap.get('python_bin', 'python'),
            'RAP_DEVICE': cfg.planner.rap.get('device', 'cuda'),
            'RAP_IMAGE_SCALE': cfg.planner.rap.get('image_scale', 0.4),
            'RAP_HF_HUB_OFFLINE': cfg.planner.rap.get('hf_hub_offline', True),
            'RAP_TRANSFORMERS_OFFLINE': cfg.planner.rap.get('transformers_offline', True),
            'RAP_VLM_ENABLED': cfg.planner.rap.vlm.get('enabled', False),
            'RAP_VLM_BACKEND': cfg.planner.rap.vlm.get('backend', 'qwen3_vl'),
            'RAP_VLM_MODEL_ID': cfg.planner.rap.vlm.get('model_id', 'Qwen/Qwen3-VL-8B-Instruct'),
            'RAP_VLM_DEVICE': cfg.planner.rap.vlm.get('device', 'auto'),
            'RAP_VLM_MAX_NEW_TOKENS': cfg.planner.rap.vlm.get('max_new_tokens', 300),
            'RAP_VLM_CANDIDATE_LIMIT': cfg.planner.rap.vlm.get('candidate_limit', 10),
            'RAP_VLM_TIMEOUT_SEC': cfg.planner.rap.vlm.get('timeout_sec', 10.0),
            'RAP_VLM_SAVE_DEBUG_ARTIFACTS': cfg.planner.rap.vlm.get('save_debug_artifacts', True),
            'RAP_VLM_DEBUG_DIR_NAME': cfg.planner.rap.vlm.get('debug_dir_name', 'vlm_debug'),
        }

    process = launch(ad_path, args.ad_cuda, output, extra_env=extra_env)
    try:
        create_gym_env(cfg, output)
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        process.kill()
    
    # create_gym_env(cfg, output)
