from sim.ilqr.lqr_solver import ILQRSolverParameters, ILQRWarmStartParameters, ILQRSolver
import os
import numpy as np

solver_params = ILQRSolverParameters(
    discretization_time=0.5,
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
    input_cost_diagonal_entries=[1.0, 10.0],
    state_trust_region_entries=[1.0] * 5,
    input_trust_region_entries=[1.0] * 2,
    max_ilqr_iterations=100,
    convergence_threshold=1e-6,
    max_solve_time=0.05,
    max_acceleration=3.0,
    max_steering_angle=np.pi / 3.0,
    max_steering_angle_rate=0.4,
    min_velocity_linearization=0.01,
    wheelbase=2.7
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=0.5,
    k_steering_angle_error_feedback=0.05,
    lookahead_distance_lateral_error=15.0,
    k_lateral_error=0.1,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

lqr = ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _trajectory_target_speed(plan_traj):
    """Estimate a speed target from the whole reference path."""
    xy = np.asarray(plan_traj[:, :2], dtype=np.float64)
    if len(xy) < 2:
        return 0.0

    dt = solver_params.discretization_time
    horizon = dt * float(len(xy) - 1)
    if horizon <= 0.0:
        return 0.0

    deltas = np.diff(xy, axis=0)
    arc_length = float(np.sum(np.linalg.norm(deltas, axis=1)))
    forward_distance = max(float(xy[-1, 0] - xy[0, 0]), 0.0)

    # Forward progress is stable on straight roads; arc length keeps curved
    # trajectories from looking artificially slow.
    speed_from_forward = forward_distance / horizon
    speed_from_arc = arc_length / horizon
    blend = float(np.clip(_env_float("AUTOAGENT0_LQR_SPEED_ARC_BLEND", 0.35), 0.0, 1.0))
    target_speed = (1.0 - blend) * speed_from_forward + blend * speed_from_arc

    gain = _env_float("AUTOAGENT0_LQR_SPEED_TARGET_GAIN", 1.15)
    max_speed = _env_float("AUTOAGENT0_LQR_SPEED_TARGET_MAX", 8.0)
    target_speed *= gain

    base_speed = _env_float("AUTOAGENT0_LQR_BASE_SPEED_MPS", 0.0)
    base_min_path = _env_float("AUTOAGENT0_LQR_BASE_SPEED_MIN_PATH_M", 5.0)
    if base_speed > 0.0 and max(forward_distance, arc_length) >= base_min_path:
        target_speed = max(target_speed, base_speed)

    return float(np.clip(target_speed, 0.0, max_speed))


def _speed_assisted_accel(plan_traj, init_state, accel_cmd):
    """Keep iLQR steering, but avoid overly timid longitudinal output."""
    if not _env_bool("AUTOAGENT0_LQR_SPEED_ASSIST", True):
        return accel_cmd

    current_speed = float(init_state[3])
    target_speed = _trajectory_target_speed(plan_traj)
    deadband = _env_float("AUTOAGENT0_LQR_SPEED_DEADBAND", 0.25)
    if target_speed <= current_speed + deadband:
        return accel_cmd

    response_sec = max(_env_float("AUTOAGENT0_LQR_SPEED_RESPONSE_SEC", 1.0), 1e-3)
    speed_accel = (target_speed - current_speed) / response_sec
    speed_accel = float(
        np.clip(
            speed_accel,
            -solver_params.max_acceleration,
            solver_params.max_acceleration,
        )
    )
    return float(max(float(accel_cmd), speed_accel))


def plan2control(plan_traj, init_state):
    current_state = init_state
    solutions = lqr.solve(current_state, plan_traj)
    optimal_inputs = solutions[-1].input_trajectory
    accel_cmd = optimal_inputs[0, 0]
    steering_rate_cmd = optimal_inputs[0, 1]
    accel_cmd = _speed_assisted_accel(plan_traj, current_state, accel_cmd)
    return accel_cmd, steering_rate_cmd

if __name__ == '__main__':
    # plan_traj = np.zeros((6,5))
    # plan_traj[:, 0] = 1
    # plan_traj[:, 1] = np.ones(6)
    # plan_traj = np.cumsum(plan_traj, axis=0)
    # print(plan_traj)
    plan_traj = np.array([[-0.18724936,  2.29100776,  0.,          0.,          0.,        ],
                        [-0.29260731,  2.2971828 ,  0.,          0.,          0.        ],
                        [-0.46831554,  2.55596018,  0.,          0.,          0.        ],
                        [-0.5859955 ,  2.73183298,  0.,          0.,          0.        ],
                        [-0.62684   ,  2.84659386,  0.,          0.,          0.        ],
                        [-0.67761713,  2.80647802,  0.,          0.,          0.        ]])
    plan_traj = plan_traj[:, [1,0,2,3,4]]
    init_state = np.array([0.00000000e+00, 3.46944695e-17, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    print(plan_traj.shape, init_state.shape)
    acc, steer = plan2control(plan_traj, init_state)
    print(acc, steer)
