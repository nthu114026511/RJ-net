import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def plot_loss(loss_history, config, save_path):
    """Plots and saves the training loss history."""
    plt.figure(figsize=(12, 8))
    
    # 如果 loss_history 是字典（包含多個損失項）
    if isinstance(loss_history, dict):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        axes[0, 0].plot(range(1, len(loss_history['total']) + 1), loss_history['total'])
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (Log Scale)')
        axes[0, 0].grid(True)
        
        # PDE residual
        axes[0, 1].plot(range(1, len(loss_history['pde']) + 1), loss_history['pde'], 'r-')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('PDE Residual Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (Log Scale)')
        axes[0, 1].grid(True)
        
        # Boundary condition loss
        axes[1, 0].plot(range(1, len(loss_history['bc']) + 1), loss_history['bc'], 'g-')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title('Boundary Condition Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (Log Scale)')
        axes[1, 0].grid(True)
        
        # Reaction loss
        if 'reaction' in loss_history:
            axes[1, 1].plot(range(1, len(loss_history['reaction']) + 1), loss_history['reaction'], 'm-')
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_title('Reaction Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss (Log Scale)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
    else:
        # 簡單版本（單一損失曲線）
        plt.plot(range(1, config['training']['epochs'] + 1), loss_history)
        plt.yscale('log')
        plt.title('Training Loss Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.grid(True)
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"-> Loss plot saved to {save_path}")


def plot_comparison(x_np, numerical_history, config, save_path, exact_history=None):
    """Plots and saves the comparison between initial and final states."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    t_max = config['physics']['t_max']
    
    # Left plot: Initial vs Final
    axes[0].plot(x_np, numerical_history[0], 'b-', label='Initial Condition (t=0)', linewidth=2)
    axes[0].plot(x_np, numerical_history[-1], 'r--', label=f'Numerical Solution (t={t_max})', linewidth=2)
    if exact_history is not None:
        axes[0].plot(x_np, exact_history[-1], 'g:', lw=3, label=f'Exact Solution (t={t_max})')
    axes[0].set_title('Comparison of Initial and Final States')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel(r'$\rho(x,t)$')
    axes[0].legend()
    axes[0].grid(True)
    
    # Right plot: Solution evolution over time
    time_indices = np.linspace(0, len(numerical_history)-1, 5, dtype=int)
    t_values = np.linspace(config['physics']['t_min'], t_max, len(numerical_history))
    
    for idx in time_indices:
        axes[1].plot(x_np, numerical_history[idx], label=f't = {t_values[idx]:.2f}')
    axes[1].set_title('Solution Evolution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel(r'$\rho(x,t)$')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"-> State comparison plot saved to {save_path}")


def plot_reaction_jacobian(x_np, R_history, J_history, config, save_path):
    """Plots the evolution of reaction accumulation R and Jacobian J."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot R (reaction accumulation)
    time_indices = np.linspace(0, len(R_history)-1, 5, dtype=int)
    t_max = config['physics']['t_max']
    t_values = np.linspace(config['physics']['t_min'], t_max, len(R_history))
    
    for idx in time_indices:
        axes[0].plot(x_np, R_history[idx], label=f't = {t_values[idx]:.2f}')
    axes[0].set_title('Reaction Accumulation R(x,t)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('R')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot J (Jacobian)
    for idx in time_indices:
        axes[1].plot(x_np, J_history[idx], label=f't = {t_values[idx]:.2f}')
    axes[1].set_title('Jacobian J(x,t)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('J')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"-> Reaction-Jacobian plot saved to {save_path}")


def create_animation(x_np, numerical_history, time_steps, config, save_path, 
                     exact_history=None, R_history=None, J_history=None):
    """Creates and saves an animation of the solution's evolution."""
    if R_history is not None and J_history is not None:
        # 創建多子圖動畫
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # ρ (density)
        line_numerical, = axes[0].plot([], [], 'r-', linewidth=2, label='ρ (Numerical)')
        if exact_history is not None:
            line_exact, = axes[0].plot([], [], 'g:', lw=3, label='ρ (Exact)')
        axes[0].set_xlim(config['physics']['x_min'], config['physics']['x_max'])
        rho_min = min([min(h) for h in numerical_history])
        rho_max = max([max(h) for h in numerical_history])
        axes[0].set_ylim(rho_min - 0.1, rho_max + 0.1)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel(r'$\rho(x,t)$')
        axes[0].set_title('Density')
        axes[0].legend()
        axes[0].grid(True)
        
        # R (reaction accumulation)
        line_R, = axes[1].plot([], [], 'b-', linewidth=2, label='R')
        axes[1].set_xlim(config['physics']['x_min'], config['physics']['x_max'])
        R_min = min([min(h) for h in R_history])
        R_max = max([max(h) for h in R_history])
        axes[1].set_ylim(R_min - 0.1, R_max + 0.1)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('R(x,t)')
        axes[1].set_title('Reaction Accumulation')
        axes[1].legend()
        axes[1].grid(True)
        
        # J (Jacobian)
        line_J, = axes[2].plot([], [], 'm-', linewidth=2, label='J')
        axes[2].set_xlim(config['physics']['x_min'], config['physics']['x_max'])
        J_min = min([min(h) for h in J_history])
        J_max = max([max(h) for h in J_history])
        axes[2].set_ylim(J_min - 0.1, J_max + 0.1)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('J(x,t)')
        axes[2].set_title('Jacobian')
        axes[2].legend()
        axes[2].grid(True)
        
        time_text = fig.suptitle('')
        
        def update(frame):
            line_numerical.set_data(x_np, numerical_history[frame])
            if exact_history is not None:
                line_exact.set_data(x_np, exact_history[frame])
            line_R.set_data(x_np, R_history[frame])
            line_J.set_data(x_np, J_history[frame])
            fig.suptitle(f't = {time_steps[frame]:.3f} s')
            return line_numerical, line_R, line_J
        
    else:
        # 簡單版本（只有密度）
        fig, ax = plt.subplots(figsize=(10, 6))
        line_numerical, = ax.plot([], [], 'r-', linewidth=2, label='Numerical Solution')
        if exact_history is not None:
            line_exact, = ax.plot([], [], 'g:', lw=3, label='Exact Solution')
        
        ax.set_xlim(config['physics']['x_min'], config['physics']['x_max'])
        rho_min = min([min(h) for h in numerical_history])
        rho_max = max([max(h) for h in numerical_history])
        ax.set_ylim(rho_min - 0.1, rho_max + 0.1)
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\rho(x,t)$')
        ax.set_title('RJ-net: Reaction-Diffusion-Convection')
        ax.legend()
        ax.grid(True)
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
        def update(frame):
            line_numerical.set_data(x_np, numerical_history[frame])
            if exact_history is not None:
                line_exact.set_data(x_np, exact_history[frame])
            time_text.set_text(f't = {time_steps[frame]:.3f} s')
            return line_numerical,
    
    anim = FuncAnimation(fig, update, frames=len(time_steps), interval=40, blit=False)
    writer = PillowWriter(fps=config['visualization']['animation_fps'])
    anim.save(save_path, writer=writer)
    plt.close()
    print(f"-> Animation saved to {save_path}")
