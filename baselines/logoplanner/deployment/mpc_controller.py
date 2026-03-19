import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Mpc_controller:
    def __init__(self, global_planed_traj, N=20, desired_v=0.3, 
                 vx_max=0.4, vy_max=0.4, wz_max=0.4, ref_gap=4, 
                 is_omnidirectional=True):
        self.N = N
        self.desired_v = desired_v
        self.ref_gap = ref_gap
        self.T = 0.05
        self.is_omnidirectional = is_omnidirectional
        
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = N // ref_gap + 1

        opti = ca.Opti()
        opt_controls = opti.variable(N, 3)
        vx, vy, wz = opt_controls[:, 0], opt_controls[:, 1], opt_controls[:, 2]
        opt_states = opti.variable(N+1, 3)
        x, y, theta = opt_states[:, 0], opt_states[:, 1], opt_states[:, 2]

        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len)

        if self.is_omnidirectional:
            f = lambda x_, u_: ca.vertcat(u_[0], u_[1], u_[2])
        else:
            f = lambda x_, u_: ca.vertcat(u_[0], u_[1], u_[2])
            for i in range(N):
                opti.subject_to(vy[i] * ca.cos(theta[i]) == vx[i] * ca.sin(theta[i]))

        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(N):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T * self.T
            opti.subject_to(opt_states[i+1, :] == x_next)

        Q = np.diag([10.0, 10.0, 0.5])
        R = np.diag([0.05, 0.05, 10])
        obj = 0
        for i in range(N):
            obj += ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
            if i % ref_gap == 0:
                nn = i // ref_gap
                state_error = opt_states[i, :] - opt_xs[nn*3:nn*3+3].T
                obj += ca.mtimes([state_error, Q, state_error.T])
        opti.minimize(obj)

        opti.subject_to(opti.bounded(-vx_max, vx, vx_max))
        opti.subject_to(opti.bounded(-vy_max, vy, vy_max))
        opti.subject_to(opti.bounded(-wz_max, wz, wz_max))

        opts_setting = {
            'ipopt.max_iter': 150,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        opti.solver('ipopt', opts_setting)

        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    def make_ref_denser(self, ref_traj, ratio=50):
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj)-1, num=len(ref_traj)*ratio)
        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')
        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        return np.stack((uniform_x, uniform_y), axis=1)
    
    def update_ref_traj(self, global_planed_traj):
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = self.N // self.ref_gap + 1
        
    def solve(self, x0):
        ref_traj = self.find_reference_traj(x0, self.ref_traj)
        ref_theta = self.compute_ref_theta(ref_traj)
        ref_traj = np.concatenate((ref_traj, ref_theta.reshape(-1, 1)), axis=1).reshape(-1, 1)
        
        self.opti.set_value(self.opt_xs, ref_traj)
        u0 = np.zeros((self.N, 3)) if self.last_opt_u_controls is None else self.last_opt_u_controls
        x00 = np.zeros((self.N+1, 3)) if self.last_opt_x_states is None else self.last_opt_x_states

        self.opti.set_value(self.opt_x0, x0)
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x00)

        sol = self.opti.solve()

        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)

        return self.last_opt_u_controls, self.last_opt_x_states

    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        
    def find_reference_traj(self, x0, global_planed_traj):
        ref_traj_pts = []
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))
        desire_arc_length = self.desired_v * self.ref_gap * self.T
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        for i in range(nearest_idx, len(global_planed_traj)-1):
            if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)

    # Fixed: Add safety check for short trajectories
    def compute_ref_theta(self, ref_traj):
        if len(ref_traj) <= 1:
            return np.array([0.0])
        ref_theta = np.zeros(len(ref_traj))
        for i in range(len(ref_traj)-1):
            dx = ref_traj[i+1, 0] - ref_traj[i, 0]
            dy = ref_traj[i+1, 1] - ref_traj[i, 1]
            ref_theta[i] = np.arctan2(dy, dx)
        ref_theta[-1] = ref_theta[-2]
        return ref_theta

# Trajectory Generators
def generate_l_shaped_traj():
    waypoints = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    t = np.linspace(0, 1, 1000)
    x = np.interp(t, [0, 0.5, 1.0], waypoints[:, 0])
    y = np.interp(t, [0, 0.5, 1.0], waypoints[:, 1])
    return np.stack((x, y), axis=1)

def generate_sinusoidal_traj():
    x = np.linspace(0, 5, 1000)
    y = np.sin(x)
    return np.stack((x, y), axis=1)

# Fixed Test Function
def test_mpc_with_traj(traj_name, traj_generator, is_omnidirectional=True):
    global_planed_traj = traj_generator()
    print(f"Testing MPC with trajectory: {traj_name}, Points: {global_planed_traj.shape}")
    mpc = Mpc_controller(
        global_planed_traj,
        is_omnidirectional=is_omnidirectional,
        vx_max=0.4, vy_max=0.4, wz_max=0.6
    )
    mpc.reset()
    
    # Fixed: Take first 2 points to compute initial theta
    init_ref_traj = global_planed_traj[:2]
    init_theta = mpc.compute_ref_theta(init_ref_traj)[0]
    init_state = np.array([global_planed_traj[0, 0], global_planed_traj[0, 1], init_theta])
    
    x_history = [init_state]
    u_history = []
    max_steps = 150
    
    for i in range(max_steps):
        current_pos = x_history[-1][:2]
        end_pos = global_planed_traj[-1, :]
        if np.linalg.norm(current_pos - end_pos) < 0.05:
            print(f"Trajectory '{traj_name}' completed in {i+1} steps!")
            break
        
        opt_u, opt_x = mpc.solve(x_history[-1])
        x_history.append(opt_x[1, :])
        u_history.append(opt_u[0, :])
    
    x_history = np.array(x_history)
    u_history = np.array(u_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(global_planed_traj[:, 0], global_planed_traj[:, 1], '--', 
             linewidth=2, label=f'Reference ({traj_name})')
    ax1.plot(x_history[:, 0], x_history[:, 1], linewidth=2.5, 
             label=f'MPC Tracking', color='orange')
    ax1.scatter(global_planed_traj[0, 0], global_planed_traj[0, 1], 
                color='green', s=50, label='Start')
    ax1.scatter(global_planed_traj[-1, 0], global_planed_traj[-1, 1], 
                color='red', s=50, label='End')
    ax1.set_xlabel('X (m)'), ax1.set_ylabel('Y (m)')
    ax1.set_title(f'MPC Trajectory Tracking: {traj_name}')
    ax1.legend(), ax1.grid(True)
    
    ax2.plot(u_history[:, 0], linewidth=2, label='vx (x.vel)')
    ax2.plot(u_history[:, 1], linewidth=2, label='vy (y.vel)')
    ax2.plot(u_history[:, 2], linewidth=2, label='wz (theta.vel)')
    ax2.set_xlabel('Time Step'), ax2.set_ylabel('Velocity (m/s or rad/s)')
    ax2.set_title(f'Control Inputs: {traj_name}')
    ax2.legend(), ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Run Tests
if __name__ == '__main__':
    fig1 = test_mpc_with_traj(
        traj_name="L-Shaped",
        traj_generator=generate_l_shaped_traj,
        is_omnidirectional=True
    )
    
    fig2 = test_mpc_with_traj(
        traj_name="Sinusoidal Curve",
        traj_generator=generate_sinusoidal_traj,
        is_omnidirectional=True
    )
    
    plt.show()
