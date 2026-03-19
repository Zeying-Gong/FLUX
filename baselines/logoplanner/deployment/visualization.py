def plot_trajectory(global_planed_traj, x_history, u_history, traj_name='real'):
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    if len(u_history) == 0:
        print("No control inputs to plot.")
        u_history = np.array([[0.0, 0.0, 0.0]])
        x_history = np.array([global_planed_traj[0]])

    # Plot trajectory
    ax1.plot(global_planed_traj[:, 0], global_planed_traj[:, 1], '--', 
             linewidth=2, label=f'Reference ({traj_name})')
    ax1.plot(x_history[:, 0], x_history[:, 1], label='Executed Path', color='blue')
    ax1.set_title(f'Trajectory for {traj_name}')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.axis('equal')
    ax1.grid()
    ax1.legend()

    # Plot control inputs
    time_steps = np.arange(len(u_history))
    ax2.plot(time_steps, u_history[:, 0], label='Vx', color='red')
    ax2.plot(time_steps, u_history[:, 1], label='Vy', color='green')
    ax2.plot(time_steps, u_history[:, 2], label='Wz', color='orange')
    ax2.set_title('Control Inputs Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Control Input Value')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'trajectory_{traj_name}.png')
    return fig