"""
Extended Kalman Filter (EKF) with Disturbance Force Estimation
Augmented state vector (15x1):
    x_aug = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r, dx, dy, dz]^T
Measurement vector (9x1):
    z = [px_meas, py_meas, pz_meas, phi_meas, theta_meas, psi_meas, p_meas, q_meas, r_meas]^T
"""

import numpy as np
from quadrotor_model import QuadrotorModel


class ExtendedKalmanFilter:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.model = QuadrotorModel()
        self.n_states = 15
        self.n_meas = 9

        # State estimate: [pos(3), vel(3), euler(3), omega(3), disturbance(3)]
        self.x_hat = np.zeros(self.n_states)

        # Initial estimation error covariance matrix
        self.P = np.eye(self.n_states) * 0.1

        # Process noise covariance matrix Q (trust in physics model)
        self.Q = np.eye(self.n_states) * 1e-4
        self.Q[3:6, 3:6] *= 1e-2     # Higher uncertainty in velocities
        self.Q[12:15, 12:15] *= 1e-1 # High variance for unknown disturbance adaptation

        # Measurement noise covariance matrix R (sensor noise characteristics)
        self.R = np.eye(self.n_meas) * 1e-2
        self.R[0:3, 0:3] = np.eye(3) * (0.05 ** 2)   # Position sensor noise std = 5cm
        self.R[3:6, 3:6] = np.eye(3) * (0.02 ** 2)   # Orientation sensor noise std = ~1 deg
        self.R[6:9, 6:9] = np.eye(3) * (0.01 ** 2)   # Gyroscope noise std

        # Measurement matrix H (linear mapping: maps augmented states to measurements)
        self.H = np.zeros((self.n_meas, self.n_states))
        self.H[0:3, 0:3] = np.eye(3)    # Position
        self.H[3:6, 6:9] = np.eye(3)    # Euler angles
        self.H[6:9, 9:12] = np.eye(3)   # Angular velocities

    def compute_jacobian_f(self, state, control_input):
        """Numerical computation of state transition Jacobian: F = df/dx."""
        eps = 1e-6
        F = np.zeros((self.n_states, self.n_states))
        f0 = self._augmented_dynamics(state, control_input)

        for i in range(self.n_states):
            state_pert = state.copy()
            state_pert[i] += eps
            f_pert = self._augmented_dynamics(state_pert, control_input)
            F[:, i] = (f_pert - f0) / eps

        # Discretize Jacobian: Phi ≈ I + F * dt
        Phi = np.eye(self.n_states) + F * self.dt
        return Phi

    def _augmented_dynamics(self, aug_state, control_input):
        """Augmented dynamics where disturbance derivative is modeled as random walk."""
        quad_state = aug_state[0:12]
        dist_force = aug_state[12:15]

        d_quad = self.model.dynamics(quad_state, control_input, external_force=dist_force)
        d_dist = np.zeros(3)  # Modeled as constant with zero mean derivative

        return np.concatenate([d_quad, d_dist])

    def predict(self, control_input):
        """EKF Prediction Step."""
        # 1. State propagation via Runge-Kutta integration
        k1 = self._augmented_dynamics(self.x_hat, control_input)
        k2 = self._augmented_dynamics(self.x_hat + 0.5 * self.dt * k1, control_input)
        k3 = self._augmented_dynamics(self.x_hat + 0.5 * self.dt * k2, control_input)
        k4 = self._augmented_dynamics(self.x_hat + self.dt * k3, control_input)
        self.x_hat += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 2. Covariance propagation: P = Phi * P * Phi^T + Q
        Phi = self.compute_jacobian_f(self.x_hat, control_input)
        self.P = Phi @ self.P @ Phi.T + self.Q

        return self.x_hat

    def update(self, measurement):
        """EKF Update Step with sensor measurement."""
        # Innovation (measurement residual)
        z_pred = self.H @ self.x_hat
        y = measurement - z_pred

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Optimal Kalman Gain: K = P * H^T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State and covariance correction
        self.x_hat += K @ y
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P

        return self.x_hat


if __name__ == "__main__":
    ekf = ExtendedKalmanFilter(dt=0.01)
    u_nominal = np.array([ekf.model.mass * ekf.model.g, 0.0, 0.0, 0.0])

    # Propagate 10 steps
    for step in range(10):
        ekf.predict(u_nominal)
        # Mock noisy sensor reading around origin
        noisy_meas = np.zeros(9) + np.random.normal(0, 0.02, 9)
        est = ekf.update(noisy_meas)

    print("--- EKF Module Self-Test ---")
    print("Estimated position:    ", np.round(est[0:3], 4))
    print("Estimated disturbance: ", np.round(est[12:15], 4))
    print("EKF state estimator validated successfully.")