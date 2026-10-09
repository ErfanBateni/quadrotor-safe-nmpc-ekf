"""
3D Trajectory Visualization, Performance Telemetry Plotter,
and Real-Time Flight Animation Engine.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Visualizer:
    def __init__(self):
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_results(self, t_vec, state_hist, ref_hist, est_hist, dist_hist,
                     ctrl_hist, cbf_flags, obs_center, obs_radius):
        """Generates publication-quality 2D telemetry subplots."""
        state_hist = np.array(state_hist)
        ref_hist = np.array(ref_hist)
        est_hist = np.array(est_hist)
        dist_hist = np.array(dist_hist)
        ctrl_hist = np.array(ctrl_hist)
        cbf_flags = np.array(cbf_flags)

        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle("Safe Trajectory Tracking & Disturbance Estimation Performance",
                     fontsize=14, fontweight='bold')

        # 1. 3D Position tracking
        axs[0, 0].plot(t_vec, ref_hist[:, 0], 'r--', label='Ref X')
        axs[0, 0].plot(t_vec, state_hist[:, 0], 'b-', label='Actual X')
        axs[0, 0].plot(t_vec, ref_hist[:, 1], 'm--', label='Ref Y')
        axs[0, 0].plot(t_vec, state_hist[:, 1], 'c-', label='Actual Y')
        axs[0, 0].plot(t_vec, ref_hist[:, 2], 'k--', label='Ref Z')
        axs[0, 0].plot(t_vec, state_hist[:, 2], 'g-', label='Actual Z')
        axs[0, 0].set_ylabel("Position [m]")
        axs[0, 0].set_title("Cartesian Tracking vs Reference")
        axs[0, 0].legend(loc='upper right', ncol=3, fontsize=8)
        axs[0, 0].grid(True, linestyle=':', alpha=0.6)

        # 2. Obstacle distance margin
        dists = np.linalg.norm(state_hist[:, 0:3] - obs_center, axis=1)
        axs[0, 1].plot(t_vec, dists, 'b-', label='Distance to Obstacle')
        axs[0, 1].axhline(y=obs_radius, color='r', linestyle='--', label='Safety Boundary')
        axs[0, 1].set_ylabel("Distance [m]")
        axs[0, 1].set_title("Safety Certificate Margin (CBF)")
        axs[0, 1].legend(loc='lower right', fontsize=8)
        axs[0, 1].grid(True, linestyle=':', alpha=0.6)

        # 3. Attitude tracking
        axs[1, 0].plot(t_vec, np.rad2deg(state_hist[:, 6]), label='Roll (phi)')
        axs[1, 0].plot(t_vec, np.rad2deg(state_hist[:, 7]), label='Pitch (theta)')
        axs[1, 0].plot(t_vec, np.rad2deg(state_hist[:, 8]), label='Yaw (psi)')
        axs[1, 0].set_ylabel("Attitude [deg]")
        axs[1, 0].set_title("Quadrotor Attitude Profile")
        axs[1, 0].legend(loc='upper right', fontsize=8)
        axs[1, 0].grid(True, linestyle=':', alpha=0.6)

        # 4. Wind disturbance estimation
        axs[1, 1].plot(t_vec, dist_hist[:, 0], 'r--', label='True Dist X')
        axs[1, 1].plot(t_vec, est_hist[:, 12], 'b-', label='EKF Est Dist X')
        axs[1, 1].plot(t_vec, dist_hist[:, 1], 'm--', label='True Dist Y')
        axs[1, 1].plot(t_vec, est_hist[:, 13], 'c-', label='EKF Est Dist Y')
        axs[1, 1].set_ylabel("Force [N]")
        axs[1, 1].set_title("External Wind Disturbance Force Estimation")
        axs[1, 1].legend(loc='upper right', fontsize=8)
        axs[1, 1].grid(True, linestyle=':', alpha=0.6)

        # 5. Control thrust input
        axs[2, 0].plot(t_vec, ctrl_hist[:, 0], 'k-', label='Thrust Command')
        axs[2, 0].set_ylabel("Thrust [N]")
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_title("Total Control Thrust")
        axs[2, 0].legend(loc='upper right', fontsize=8)
        axs[2, 0].grid(True, linestyle=':', alpha=0.6)

        # 6. Safety Trigger Profile
        axs[2, 1].step(t_vec, cbf_flags.astype(int), 'r-', label='CBF Intervened')
        axs[2, 1].set_ylabel("Active State")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylim(-0.2, 1.2)
        axs[2, 1].set_title("Safety Filter Trigger Profile")
        axs[2, 1].legend(loc='upper right', fontsize=8)
        axs[2, 1].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()

    def animate_3d_flight(self, state_hist, ref_hist, obs_center, obs_radius, arm_length=0.25):
        """Creates and renders a live, smooth 3D animation of the quadrotor maneuvering around the obstacle."""
        state_hist = np.array(state_hist)
        ref_hist = np.array(ref_hist)
        n_frames = len(state_hist)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Static reference path
        ax.plot(ref_hist[:, 0], ref_hist[:, 1], ref_hist[:, 2], 'r--', label='Reference Path', alpha=0.7)

        # Static Obstacle Sphere
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        x_sph = obs_center[0] + obs_radius * np.outer(np.cos(u), np.sin(v))
        y_sph = obs_center[1] + obs_radius * np.outer(np.sin(u), np.sin(v))
        z_sph = obs_center[2] + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sph, y_sph, z_sph, color='red', alpha=0.35, edgecolor='none')

        # Dynamic elements: trajectory line, quadrotor arms, and rotors
        traj_line, = ax.plot([], [], [], 'b-', linewidth=2, label='Flown Path')
        arm1_line, = ax.plot([], [], [], 'k-', linewidth=3)
        arm2_line, = ax.plot([], [], [], 'k-', linewidth=3)
        rotors_scatter = ax.scatter([], [], [], color='darkblue', s=25)

        # Camera and axis boundaries
        ax.set_xlim([0.0, 6.5])
        ax.set_ylim([-0.8, 1.0])
        ax.set_zlim([0.8, 2.0])
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Real-Time 3D Flight Animation with Collision Avoidance", fontweight='bold')
        ax.legend(loc='upper left')

        def rot_mat(phi, theta, psi):
            c_ph, s_ph = np.cos(phi), np.sin(phi)
            c_th, s_th = np.cos(theta), np.sin(theta)
            c_ps, s_ps = np.cos(psi), np.sin(psi)
            return np.array([
                [c_th * c_ps, s_ph * s_th * c_ps - c_ph * s_ps, c_ph * s_th * c_ps + s_ph * s_ps],
                [c_th * s_ps, s_ph * s_th * s_ps + c_ph * c_ps, c_ph * s_th * s_ps - s_ph * c_ps],
                [-s_th,       s_ph * c_th,                     c_ph * c_th]
            ])

        # Cross frame geometry in body coordinates
        d = arm_length
        p1 = np.array([d, 0.0, 0.0])
        p2 = np.array([-d, 0.0, 0.0])
        p3 = np.array([0.0, d, 0.0])
        p4 = np.array([0.0, -d, 0.0])

        def update(frame):
            # Update flown trajectory trail
            traj_line.set_data(state_hist[:frame, 0], state_hist[:frame, 1])
            traj_line.set_3d_properties(state_hist[:frame, 2])

            # Current state
            pos = state_hist[frame, 0:3]
            euler = state_hist[frame, 6:9]
            R = rot_mat(euler[0], euler[1], euler[2])

            # Transform arm endpoints to inertial space
            w_p1 = pos + R @ p1
            w_p2 = pos + R @ p2
            w_p3 = pos + R @ p3
            w_p4 = pos + R @ p4

            # Update arm lines
            arm1_line.set_data([w_p1[0], w_p2[0]], [w_p1[1], w_p2[1]])
            arm1_line.set_3d_properties([w_p1[2], w_p2[2]])

            arm2_line.set_data([w_p3[0], w_p4[0]], [w_p3[1], w_p4[1]])
            arm2_line.set_3d_properties([w_p3[2], w_p4[2]])

            # Update rotor tip points
            rotors = np.array([w_p1, w_p2, w_p3, w_p4])
            rotors_scatter._offsets3d = (rotors[:, 0], rotors[:, 1], rotors[:, 2])

            return traj_line, arm1_line, arm2_line, rotors_scatter

        # 50 ms per frame corresponds to dt = 0.05s
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False, repeat=True)

        # 50 ms per frame corresponds to dt = 0.05s
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False, repeat=True)

        # Save the live 3D flight as an animated GIF
        try:
            print("Exporting 3D Flight Animation to flight_simulation.gif (using Pillow) ...")
            # Using PillowWriter which is included in standard matplotlib setups
            from matplotlib.animation import PillowWriter
            anim.save('flight_simulation.gif', writer=PillowWriter(fps=20))
            print("Animation exported successfully as flight_simulation.gif")
        except Exception as e:
            print(f"Skipped GIF saving due to: {e}. Animation will display on screen.")

        plt.show()

        plt.show()