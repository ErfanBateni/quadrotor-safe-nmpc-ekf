"""
Main Closed-Loop Simulation Pipeline
Features:
    - Obstacle-Aware Real-Time NMPC
    - Disturbance-Aware EKF
    - 6-DOF Physical Quadrotor Model
"""

import numpy as np
from quadrotor_model import QuadrotorModel
from ekf_estimator import ExtendedKalmanFilter
from nmpc_controller import NMPCController
from visualizer import Visualizer


def generate_reference_trajectory(t, horizon, dt):
    ref_window = np.zeros((12, horizon + 1))
    for k in range(horizon + 1):
        tk = t + k * dt
        ref_window[0, k] = 0.6 * tk
        ref_window[1, k] = 0.3 * np.sin(0.4 * tk)
        ref_window[2, k] = 1.4 + 0.2 * np.cos(0.4 * tk)
    return ref_window


def main():
    print("--- Running Obstacle-Aware NMPC Simulation ---")

    sim_time = 10.0
    dt = 0.05
    n_steps = int(sim_time / dt)

    robot = QuadrotorModel()
    ekf = ExtendedKalmanFilter(dt=dt)
    nmpc = NMPCController(dt=dt, horizon=12)
    vis = Visualizer()

    x_actual = np.zeros(12)
    x_actual[2] = 1.4

    obstacle_center = np.array([2.8, 0.2, 1.45])
    obstacle_radius = 0.50

    t_history = []
    state_history = []
    ref_history = []
    est_history = []
    dist_history = []
    ctrl_history = []
    cbf_flags = []

    for step in range(n_steps):
        t_current = step * dt
        t_history.append(t_current)

        # Wind disturbance
        if 2.0 <= t_current <= 7.0:
            true_wind = np.array([0.5, -0.2, 0.1])
        else:
            true_wind = np.zeros(3)
        dist_history.append(true_wind)

        # Sensor readings
        meas_noise = np.random.normal(0, 0.01, 9)
        z_meas = np.concatenate([x_actual[0:3], x_actual[6:9], x_actual[9:12]]) + meas_noise

        # EKF update
        u_prev = ctrl_history[-1] if step > 0 else np.array([robot.mass * robot.g, 0, 0, 0])
        ekf.predict(u_prev)
        x_est_aug = ekf.update(z_meas)

        state_est = x_est_aug[0:12]
        dist_est = x_est_aug[12:15]

        # Trajectory reference
        ref_traj = generate_reference_trajectory(t_current, nmpc.N, dt)

        # Solve Obstacle-Aware NMPC
        u_cmd = nmpc.solve(state_est, dist_est, ref_traj, obstacle_center, obstacle_radius)

        # Dynamic integration
        x_actual = robot.step_rk4(x_actual, u_cmd, dt, external_force=true_wind)

        # Check distance
        dist_to_obs = np.linalg.norm(x_actual[0:3] - obstacle_center)
        safety_active = dist_to_obs < (obstacle_radius + 0.3)

        state_history.append(x_actual.copy())
        ref_history.append(ref_traj[:, 0].copy())
        est_history.append(x_est_aug.copy())
        ctrl_history.append(u_cmd.copy())
        cbf_flags.append(safety_active)

        if step % 20 == 0:
            print(f"Time: {t_current:.2f}s | Dist: {dist_to_obs:.2f}m | Alt: {x_actual[2]:.2f}m")

    print("\n--- Simulation Finished. Rendering Plots & Animation ---")
    vis.plot_results(t_history, state_history, ref_history, est_history,
                     dist_history, ctrl_history, cbf_flags, obstacle_center, obstacle_radius)

    # Launch live 3D animation
    vis.animate_3d_flight(state_history, ref_history, obstacle_center, obstacle_radius)


if __name__ == "__main__":
    main()