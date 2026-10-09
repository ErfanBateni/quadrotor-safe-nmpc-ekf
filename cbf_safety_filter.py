"""
Robust 3D Control Barrier Function Safety Filter (Hard Constraint CBF-QP)
With Symmetry-Breaking Tangential Navigation to eliminate chattering and deadlock.
"""

import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp
from quadrotor_model import QuadrotorModel


class CBFSafetyFilter:
    def __init__(self):
        self.model = QuadrotorModel()
        self.gamma1 = 1.8  # Position CBF gain
        self.gamma2 = 2.4  # Velocity CBF gain
        self.a_max = 12.0

    def filter_control(self, state, u_nominal, obstacle_center, obstacle_radius):
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        omega = state[9:12]

        rel_pos = pos - obstacle_center
        dist = float(np.linalg.norm(rel_pos))

        # Strict safety radius with certified geometric buffer
        r_safe = obstacle_radius + 0.20
        activation_dist = r_safe + 1.2

        if dist > activation_dist:
            return u_nominal, False

        # Current nominal translational acceleration
        R = self.model.rotation_matrix(euler[0], euler[1], euler[2])
        thrust_body_nom = np.array([0.0, 0.0, u_nominal[0]])
        thrust_inertial_nom = R @ thrust_body_nom
        gravity = np.array([0.0, 0.0, -self.model.mass * self.model.g])
        drag = -self.model.cd_trans * vel
        a_nom = (thrust_inertial_nom + gravity + drag) / self.model.mass

        # Symmetry-breaking circulating acceleration to circumvent the obstacle sideways
        # Tangential cross product force in horizontal plane
        tangent_dir = np.array([-rel_pos[1], rel_pos[0], 0.0])
        norm_tan = np.linalg.norm(tangent_dir)
        if norm_tan > 1e-3:
            tangent_dir /= norm_tan
        else:
            tangent_dir = np.array([0.0, 1.0, 0.0])

        circulating_bias = 2.5 * (1.0 - (dist / activation_dist)) * tangent_dir
        a_ref = a_nom + circulating_bias

        # Strict HOCBF inequality: 2 * rel_pos^T * a_safe >= -2*||vel||^2 - gamma2*h_dot - gamma1*h
        h = (dist ** 2) - (r_safe ** 2)
        h_dot = 2.0 * float(np.dot(rel_pos, vel))

        G_dense = np.zeros((1, 3))
        G_dense[0, :] = -2.0 * rel_pos
        G = sp.csc_matrix(G_dense)

        rhs_limit = 2.0 * float(np.dot(vel, vel)) + self.gamma2 * h_dot + self.gamma1 * h
        h_bound = np.array([rhs_limit])

        # Quadratic Program: min 0.5 * ||a_safe - a_ref||^2
        P = sp.eye(3, format='csc')
        q = -a_ref

        lb = np.array([-self.a_max, -self.a_max, -self.a_max])
        ub = np.array([self.a_max, self.a_max, self.a_max])

        sol = solve_qp(P=P, q=q, G=G, h=h_bound, lb=lb, ub=ub, solver="osqp")

        if sol is None:
            # Fallback to purely repulsive safe vector if QP fails
            a_safe = a_nom + (3.0 * rel_pos / (dist + 1e-3))
        else:
            a_safe = sol

        # Convert 3D acceleration back to physical control inputs
        f_total = self.model.mass * a_safe - gravity - drag
        thrust_safe = float(np.linalg.norm(f_total))
        thrust_safe = np.clip(thrust_safe, 0.0, 2.2 * self.model.mass * self.model.g)

        if thrust_safe > 1e-3:
            z_b_des = f_total / thrust_safe
        else:
            z_b_des = np.array([0.0, 0.0, 1.0])

        # Safe attitude targets
        theta_des = np.arcsin(np.clip(-z_b_des[0], -0.4, 0.4))
        phi_des = np.arctan2(z_b_des[1], z_b_des[2])

        # Tracking loops
        kp_att = 5.5
        kd_rate = 0.7
        tau_x = kp_att * (phi_des - euler[0]) - kd_rate * omega[0]
        tau_y = kp_att * (theta_des - euler[1]) - kd_rate * omega[1]
        tau_z = u_nominal[3]

        u_safe = np.array([
            thrust_safe,
            np.clip(tau_x, -0.8, 0.8),
            np.clip(tau_y, -0.8, 0.8),
            np.clip(tau_z, -0.8, 0.8)
        ])

        return u_safe, True