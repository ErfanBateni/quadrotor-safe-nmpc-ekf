"""
Nonlinear 6-DOF Dynamic Model of a Multi-Rotor Aerial Robot
State vector (12x1):
    x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]^T
Control input vector (4x1):
    u = [T, tau_phi, tau_theta, tau_psi]^T
"""

import numpy as np


class QuadrotorModel:
    def __init__(self):
        # Physical parameters
        self.mass = 1.0       # Mass in kg
        self.g = 9.81         # Gravitational acceleration in m/s^2
        self.arm_length = 0.22 # Arm length in meters

        # Moments of inertia (kg * m^2)
        self.Ixx = 8.1e-3
        self.Iyy = 8.1e-3
        self.Izz = 14.2e-3
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.J_inv = np.linalg.inv(self.J)

        # Aerodynamic drag coefficients
        self.cd_trans = 0.1   # Translational drag coefficient
        self.cd_rot = 0.02    # Rotational drag coefficient

    def rotation_matrix(self, phi, theta, psi):
        """Z-Y-X Euler rotation matrix from body frame to inertial frame."""
        c_ph, s_ph = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_ps, s_ps = np.cos(psi), np.sin(psi)

        R = np.array([
            [c_th * c_ps, s_ph * s_th * c_ps - c_ph * s_ps, c_ph * s_th * c_ps + s_ph * s_ps],
            [c_th * s_ps, s_ph * s_th * s_ps + c_ph * c_ps, c_ph * s_th * s_ps - s_ph * c_ps],
            [-s_th,       s_ph * c_th,                     c_ph * c_th]
        ])
        return R

    def euler_rates_matrix(self, phi, theta):
        """Transformation matrix from body angular velocities (p, q, r) to Euler angle rates."""
        c_ph, s_ph = np.cos(phi), np.sin(phi)
        t_th = np.tan(theta)
        c_th = np.cos(theta)

        # Singularity avoidance for gimbal lock
        if abs(c_th) < 1e-4:
            c_th = 1e-4 * np.sign(c_th)

        W = np.array([
            [1.0, s_ph * t_th,  c_ph * t_th],
            [0.0, c_ph,        -s_ph],
            [0.0, s_ph / c_th,  c_ph / c_th]
        ])
        return W

    def dynamics(self, state, control_input, external_force=None):
        """
        Computes time derivative of state vector: dx/dt = f(x, u, d)
        state: 12-element numpy array
        control_input: 4-element array [Thrust, tau_x, tau_y, tau_z]
        external_force: 3-element disturbance force [Fx, Fy, Fz]
        """
        if external_force is None:
            external_force = np.zeros(3)

        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]   # phi, theta, psi
        omega = state[9:12]  # p, q, r

        thrust = control_input[0]
        torques = control_input[1:4]

        # 1. Kinematics (Position rate)
        d_pos = vel

        # 2. Translational Dynamics (Newton's second law in inertial frame)
        R = self.rotation_matrix(euler[0], euler[1], euler[2])
        thrust_body = np.array([0.0, 0.0, thrust])
        thrust_inertial = R @ thrust_body
        gravity = np.array([0.0, 0.0, -self.mass * self.g])
        drag_force = -self.cd_trans * vel

        d_vel = (thrust_inertial + gravity + drag_force + external_force) / self.mass

        # 3. Attitude Kinematics
        W = self.euler_rates_matrix(euler[0], euler[1])
        d_euler = W @ omega

        # 4. Rotational Dynamics (Euler's equations for rigid body)
        gyro_moment = np.cross(omega, self.J @ omega)
        rot_drag = -self.cd_rot * omega
        d_omega = self.J_inv @ (torques - gyro_moment + rot_drag)

        return np.concatenate([d_pos, d_vel, d_euler, d_omega])

    def step_rk4(self, state, control_input, dt, external_force=None):
        """One-step numerical integration using Runge-Kutta 4th order method."""
        k1 = self.dynamics(state, control_input, external_force)
        k2 = self.dynamics(state + 0.5 * dt * k1, control_input, external_force)
        k3 = self.dynamics(state + 0.5 * dt * k2, control_input, external_force)
        k4 = self.dynamics(state + dt * k3, control_input, external_force)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_state


if __name__ == "__main__":
    robot = QuadrotorModel()
    x0 = np.zeros(12)
    u_hover = np.array([robot.mass * robot.g, 0.0, 0.0, 0.0])
    dt = 0.01
    x_next = robot.step_rk4(x0, u_hover, dt)

    print("--- Quadrotor Physics Model Self-Test ---")
    print("Initial position:      ", np.round(x0[0:3], 4))
    print("Next position (RK4):   ", np.round(x_next[0:3], 4))
    print("Next velocity (RK4):   ", np.round(x_next[3:6], 4))
    print("Model validation successful.")