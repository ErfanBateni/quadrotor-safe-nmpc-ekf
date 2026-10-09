"""
Nonlinear Model Predictive Control (NMPC) for Obstacle-Aware Trajectory Tracking
Formulated via CasADi and solved using IPOPT with analytic obstacle constraints.
"""

import casadi as ca
import numpy as np


class NMPCController:
    def __init__(self, dt=0.05, horizon=15):
        self.dt = dt
        self.N = horizon

        self.mass = 1.0
        self.g = 9.81
        self.Ixx = 8.1e-3
        self.Iyy = 8.1e-3
        self.Izz = 14.2e-3
        self.cd_trans = 0.1

        self.n_states = 12
        self.n_controls = 4

        # State tracking weights
        self.Q = np.diag([
            15.0, 15.0, 18.0,  # Position
            1.5, 1.5, 1.5,     # Velocity
            3.0, 3.0, 12.0,     # Attitude
            0.1, 0.1, 1.0      # Body rates
        ])
        self.R = np.diag([0.05, 1.0, 1.0, 1.0])

        self.solver = None
        self._build_nlp()

    def _symbolic_dynamics(self, x, u, dist_force):
        pos = x[0:3]
        vel = x[3:6]
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]

        thrust = u[0]
        tau_x, tau_y, tau_z = u[1], u[2], u[3]

        c_ph, s_ph = ca.cos(phi), ca.sin(phi)
        c_th, s_th = ca.cos(theta), ca.sin(theta)
        c_ps, s_ps = ca.cos(psi), ca.sin(psi)

        R = ca.vertcat(
            ca.horzcat(c_th * c_ps, s_ph * s_th * c_ps - c_ph * s_ps, c_ph * s_th * c_ps + s_ph * s_ps),
            ca.horzcat(c_th * s_ps, s_ph * s_th * s_ps + c_ph * c_ps, c_ph * s_th * s_ps - s_ph * c_ps),
            ca.horzcat(-s_th, s_ph * c_th, c_ph * c_th)
        )

        thrust_inertial = ca.mtimes(R, ca.vertcat(0.0, 0.0, thrust))
        gravity = ca.vertcat(0.0, 0.0, -self.mass * self.g)
        drag = -self.cd_trans * vel

        d_pos = vel
        d_vel = (thrust_inertial + gravity + drag + dist_force) / self.mass

        t_th = ca.tan(theta)
        W = ca.vertcat(
            ca.horzcat(1.0, s_ph * t_th, c_ph * t_th),
            ca.horzcat(0.0, c_ph, -s_ph),
            ca.horzcat(0.0, s_ph / c_th, c_ph / c_th)
        )
        omega = ca.vertcat(p, q, r)
        d_euler = ca.mtimes(W, omega)

        dp = (tau_x - (self.Izz - self.Iyy) * q * r) / self.Ixx
        dq = (tau_y - (self.Ixx - self.Izz) * p * r) / self.Iyy
        dr = (tau_z - (self.Iyy - self.Ixx) * p * q) / self.Izz
        d_omega = ca.vertcat(dp, dq, dr)

        return ca.vertcat(d_pos, d_vel, d_euler, d_omega)

    def _rk4_integrator(self, x, u, dist_force):
        dt = self.dt
        k1 = self._symbolic_dynamics(x, u, dist_force)
        k2 = self._symbolic_dynamics(x + 0.5 * dt * k1, u, dist_force)
        k3 = self._symbolic_dynamics(x + 0.5 * dt * k2, u, dist_force)
        k4 = self._symbolic_dynamics(x + dt * k3, u, dist_force)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build_nlp(self):
        X = ca.SX.sym('X', self.n_states, self.N + 1)
        U = ca.SX.sym('U', self.n_controls, self.N)

        P_x0 = ca.SX.sym('P_x0', self.n_states)
        P_dist = ca.SX.sym('P_dist', 3)
        P_obs = ca.SX.sym('P_obs', 4)  # [ox, oy, oz, r_safe]
        P_ref = ca.SX.sym('P_ref', self.n_states, self.N + 1)
        P = ca.vertcat(P_x0, P_dist, P_obs, ca.reshape(P_ref, -1, 1))

        cost = 0
        g = []

        g.append(X[:, 0] - P_x0)

        obs_center = P_obs[0:3]
        obs_radius = P_obs[3]

        for k in range(self.N):
            state_err = X[:, k] - P_ref[:, k]
            u_hover = ca.vertcat(self.mass * self.g, 0.0, 0.0, 0.0)
            ctrl_err = U[:, k] - u_hover

            cost += ca.mtimes([state_err.T, self.Q, state_err])
            cost += ca.mtimes([ctrl_err.T, self.R, ctrl_err])

            # Forward RK4 dynamics
            x_next = self._rk4_integrator(X[:, k], U[:, k], P_dist)
            g.append(X[:, k + 1] - x_next)

            # Obstacle avoidance constraint with dynamic safety buffer (radius + 0.20m)
            pos_k = X[0:3, k + 1]
            dist_sq = ca.sumsqr(pos_k - obs_center)
            r_buffered = obs_radius + 0.20
            g.append(dist_sq - (r_buffered ** 2))

        term_err = X[:, self.N] - P_ref[:, self.N]
        cost += 2.0 * ca.mtimes([term_err.T, self.Q, term_err])

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        constraints = ca.vertcat(*g)

        nlp = {'x': opt_vars, 'f': cost, 'g': constraints, 'p': P}

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 60,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.tol': 1e-3
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def solve(self, current_state, disturbance_est, ref_trajectory, obs_center, obs_radius):
        obs_param = np.array([obs_center[0], obs_center[1], obs_center[2], obs_radius])
        p_vec = np.concatenate([
            current_state,
            disturbance_est,
            obs_param,
            ref_trajectory.flatten(order='F')
        ])

        n_x_vars = self.n_states * (self.N + 1)
        n_u_vars = self.n_controls * self.N
        n_total_vars = n_x_vars + n_u_vars

        lbx = -np.inf * np.ones(n_total_vars)
        ubx = np.inf * np.ones(n_total_vars)

        # Control saturations
        u_start_idx = n_x_vars
        for k in range(self.N):
            idx = u_start_idx + k * self.n_controls
            lbx[idx] = 0.0
            ubx[idx] = 2.0 * self.mass * self.g
            lbx[idx + 1: idx + 4] = -0.5
            ubx[idx + 1: idx + 4] = 0.5

        # Attitude angles limit (+/- 30 deg)
        for k in range(self.N + 1):
            idx = k * self.n_states
            lbx[idx + 6: idx + 8] = -np.pi / 6.0
            ubx[idx + 6: idx + 8] = np.pi / 6.0

        # Constraints bounds: dynamics = 0, obstacle >= 0
        lbg = []
        ubg = []
        # Initial state constraint (12)
        lbg.extend([0.0] * self.n_states)
        ubg.extend([0.0] * self.n_states)

        for _ in range(self.N):
            # Dynamic equality (12)
            lbg.extend([0.0] * self.n_states)
            ubg.extend([0.0] * self.n_states)
            # Obstacle inequality (1): dist_sq - r_safe^2 >= 0
            lbg.append(0.0)
            ubg.append(np.inf)

        lbg = np.array(lbg)
        ubg = np.array(ubg)

        x0_guess = np.zeros(n_total_vars)
        for k in range(self.N):
            x0_guess[u_start_idx + k * self.n_controls] = self.mass * self.g

        sol = self.solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_vec)
        u_opt = sol['x'][u_start_idx: u_start_idx + self.n_controls].full().flatten()

        return u_opt