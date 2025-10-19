import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_loss(loss_history, config, save_path):
    """Plots and saves the PDE loss history."""
    plt.figure(figsize=(10, 6))
    # 如果 loss_history 是字典，只繪製 PDE loss
    if isinstance(loss_history, dict):
        plt.plot(range(1, len(loss_history['pde']) + 1), loss_history['pde'], 
                linewidth=2, label='PDE Loss')
        plt.yscale('log')
        plt.title('PDE Loss Evolution During Training')
        plt.xlabel('Epoch')
        plt.ylabel('PDE Loss (Log Scale)')
        plt.legend()
    else:
        # 如果是列表，當作總 loss 繪製
        plt.plot(range(1, config['training']['epochs'] + 1), loss_history,
                linewidth=2)
        plt.yscale('log')
        plt.title('Training Loss Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"-> PDE Loss plot saved to {save_path}")


def plot_comparison(x_np, numerical_history, exact_history, config, save_path):
    """Plots and saves the comparison between initial and final states."""
    plt.figure(figsize=(10, 6))
    t_max = config['physics']['t_max']
    plt.plot(x_np, numerical_history[0], 'b-', label='Initial Condition (t=0)')
    plt.plot(x_np, numerical_history[-1], 'r--',
             label=f'Numerical Solution (t={t_max})')
    plt.plot(x_np, exact_history[-1], 'g:', lw=3,
             label=f'Exact Solution (t={t_max})')
    plt.title('Comparison of Initial and Final States')
    plt.xlabel('x')
    plt.ylabel(r'$\rho(x,t)$')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"-> State comparison plot saved to {save_path}")


def create_animation(x_np, numerical_history, exact_history, time_steps, config, save_path):
    """Creates and saves an animation of the solution's evolution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    line_numerical, = ax.plot([], [], 'r--', label='Numerical Solution')
    line_exact, = ax.plot([], [], 'g:', lw=3, label='Exact Solution')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.set_xlim(config['physics']['x_min'], config['physics']['x_max'])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\rho(x,t)$')
    ax.set_title('1D Diffusion: Numerical vs. Exact Solution')
    ax.legend()
    ax.grid(True)

    def update(frame):
        line_numerical.set_data(x_np, numerical_history[frame])
        line_exact.set_data(x_np, exact_history[frame])
        time_text.set_text(f' t = {time_steps[frame]:.2f} s')
        return line_numerical, line_exact, time_text

    anim = FuncAnimation(fig, update, frames=len(
        time_steps), interval=40, blit=True)
    writer = PillowWriter(fps=config['visualization']['animation_fps'])
    anim.save(save_path, writer=writer)
    print(f"-> Animation saved to {save_path}")

