#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DR-PINN: Diffusion-Reaction Physics-Informed Neural Network

本程式實現了基於物理的神經網路（PINN）求解擴散-反應系統：

    ∂ρ/∂t = ν·∇²ρ + r(x, ρ)

核心方程（使用有限差分離散化）：
    1. 時間導數: ∂ρ/∂t = (ρ^{n+1} - ρ^n) / Δt
    2. 擴散項: ν·∇²ρ = ν·∂²ρ/∂x²（二階中心差分）
    3. 反應項: r(x, ρ) = k⁺ρ - k⁻ρ²
    4. 反應率網路: r_{NN}(x, ρ)（由 ReactionNet 學習）
    5. 時間推進: ρ^{n+1} = ρ^n + Δt·(ν·∂²ρ/∂x² + r_{NN})

測試問題: 擴散-反應系統
    ∂ρ/∂t = ν·∂²ρ/∂x² + k⁺ρ - k⁻ρ²
    其中 ν 是擴散係數，k⁺ρ - k⁻ρ² 是 logistic 反應項
    
特色：
    - 使用有限差分計算空間導數
    - PDE 殘差改為擴散-反應形式
    - 讓模型從物理殘差學習反應項的參數化
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Import custom plotter module
import plotter


# ===========================
# Neural Network Definitions
# ===========================

class VelocityNet(nn.Module):
    """Neural network to predict velocity u(x, ρ)."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x_rho):
        """Input: [x, ρ], Output: u"""
        return self.network(x_rho)


class ReactionNet(nn.Module):
    """Neural network to predict reaction rate r(x, ρ)."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x_rho):
        """Input: [x, ρ], Output: r"""
        return self.network(x_rho)


# ===========================
# Helper Functions
# ===========================

def exact_reaction(rho, k_plus, k_minus):
    """Exact reaction term: r = k⁺ρ - k⁻ρ²"""
    return k_plus * rho - k_minus * rho**2


def exact_traveling_wave(x, t, x0=0.0):
    """
    Fisher-KPP 精確旅行波解：
    
    ρ(x, t) = [1 + exp(-(x - st - x0)/√6)]^{-2}
    
    其中波速 s = 5/√6
    """
    sqrt6 = torch.sqrt(torch.tensor(6.0, device=x.device, dtype=x.dtype))
    s = 5.0 / sqrt6  # 波速
    xi = (x - s * t - x0) / sqrt6  # 行進波座標
    return (1.0 + torch.exp(-xi))**(-2)


def initial_condition(x, x0=0.0):
    """
    Initial condition: Fisher-KPP 旅行波 (Ablowitz-Zeppetella)
    
    ρ(x, 0) = [1 + exp(-(x - x0)/√6)]^{-2}
    
    其中：
    - 擴散係數 D = 1
    - 反應係數 k⁺ = 1, k⁻ = 1 (r(ρ) = ρ(1-ρ))
    - 波速 s = 5/√6
    - 特徵長度 L = √6
    - 平移參數 x0 ∈ ℝ
    """
    sqrt6 = torch.sqrt(torch.tensor(6.0, device=x.device, dtype=x.dtype))
    # 精確解的初始條件
    return (1.0 + torch.exp(-(x - x0) / sqrt6))**(-2)


# ===========================
# Training Function
# ===========================

def train_rj_net(config, output_dir):
    """Main training function for RJ-net."""
    
    # Get parameters from config
    p = config['physics']
    t = config['training']
    DT = (p['t_max'] - p['t_min']) / (p['nt'] - 1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create spatial grid
    x = torch.linspace(p['x_min'], p['x_max'], p['nx'], device=device).view(-1, 1)
    x.requires_grad = True
    
    # Time steps
    time_steps = torch.linspace(p['t_min'], p['t_max'], p['nt'], device=device)
    
    # Initialize networks
    velocity_net = VelocityNet(hidden_size=t['hidden_size']).to(device)
    reaction_net = ReactionNet(hidden_size=t['hidden_size']).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(velocity_net.parameters()) + list(reaction_net.parameters()), 
        lr=t['learning_rate']
    )
    
    # Loss history
    loss_history = {
        'total': [],
        'pde': [],
        'bc': [],
        'reaction': []
    }
    
    # Get diffusion coefficient
    nu = p.get('nu', 0.01)  # diffusion coefficient
    
    print(f"✅ Training setup complete.")
    print(f"   Spatial points: {p['nx']}")
    print(f"   Time steps: {p['nt']}")
    print(f"   dt = {DT:.6f}")
    print(f"   dx = {(p['x_max'] - p['x_min']) / (p['nx'] - 1):.6f}")
    print(f"   Network architecture: {t['hidden_size']} hidden units")
    print(f"   Reaction parameters: k⁺={p['k_plus']}, k⁻={p['k_minus']}")
    
    print("\nStarting training...\n")
    
    dx = (p['x_max'] - p['x_min']) / (p['nx'] - 1)
    
    for epoch in range(t['epochs']):
        optimizer.zero_grad()
        
        # Initialize variables according to: ρ = (ρ₀ + R) / J
        rho_0 = initial_condition(x).detach()  # Initial density ρ₀
        R_current = torch.zeros_like(rho_0)     # R(t=0) = 0
        J_current = torch.ones_like(rho_0)      # J(t=0) = 1
        rho_current = (rho_0 + R_current) / J_current
        
        total_loss = 0.0
        total_loss_pde = 0.0
        total_loss_bc = 0.0
        total_loss_reaction = 0.0
        
        # Time marching: For n=0,1,2,...
        for n in range(p['nt'] - 1):
            # Prepare network input: [x, ρⁿ]
            net_input = torch.cat([x, rho_current], dim=1)
            
            # === 1. Predict velocity uⁿ⁺¹ = u_NN(ρⁿ) ===
            u_next = velocity_net(net_input)
            
            # === 2. Predict reaction rate rⁱⁿ⁺¹ = r_NN(ρⁿ, x) ===
            r_next = reaction_net(net_input)
            
            # === 3. Update reaction integral: Rⁱⁿ⁺¹ = Rⁱⁿ + Δt·rⁱⁿ ===
            R_next = R_current + DT * r_next
            
            # === 4. Compute ∇·u for Jacobian update ===
            # ∂u/∂x using finite differences (中心差分)
            du_dx = torch.zeros_like(u_next)
            du_dx[1:-1] = (u_next[2:] - u_next[:-2]) / (2 * dx)
            # Boundary: assume zero divergence at boundaries
            du_dx[0] = du_dx[1]
            du_dx[-1] = du_dx[-2]
            
            # === 5. Update Jacobian: Jⁿ⁺¹ = Jⁿ + Δt·(∇·u)Jⁿ - u·∇J ===
            # For 1D: ∇J ≈ ∂J/∂x
            dJ_dx = torch.zeros_like(J_current)
            dJ_dx[1:-1] = (J_current[2:] - J_current[:-2]) / (2 * dx)
            dJ_dx[0] = dJ_dx[1]
            dJ_dx[-1] = dJ_dx[-2]
            
            J_next = J_current + DT * (du_dx * J_current - u_next * dJ_dx)
            
            # === 6. Update density: ρⁿ⁺¹ = (ρ₀ + Rⁿ⁺¹) / Jⁿ⁺¹ ===
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            J_next_safe = torch.clamp(J_next, min=eps)
            rho_next = (rho_0 + R_next) / J_next_safe
            
            # === 7. Compute spatial derivatives for PDE residual ===
            # Second derivative ∂²ρ/∂x² (for diffusion term)
            d2rho_dx2 = torch.zeros_like(rho_current)
            d2rho_dx2[1:-1] = (rho_current[2:] - 2*rho_current[1:-1] + rho_current[:-2]) / (dx**2)
            d2rho_dx2[0] = d2rho_dx2[1]
            d2rho_dx2[-1] = d2rho_dx2[-2]
            
            # First derivative ∂ρ/∂x (for boundary condition)
            drho_dx = torch.zeros_like(rho_current)
            drho_dx[1:-1] = (rho_current[2:] - rho_current[:-2]) / (2 * dx)
            drho_dx[0] = drho_dx[1]
            drho_dx[-1] = drho_dx[-2]
            
            # === 8. Compute time derivative ===
            rho_t = (rho_next - rho_current) / DT
            
            # === 9. Compute PDE residual ===
            # PDE: ∂ρ/∂t = ν·∂²ρ/∂x² + r(ρ)
            # Residual: ∂ρ/∂t - ν·∂²ρ/∂x² - r ≈ 0
            diffusion_term = nu * d2rho_dx2
            pde_residual = rho_t - diffusion_term - r_next.squeeze()
            loss_pde = torch.mean(pde_residual**2)
            
            # === 10. Boundary conditions ===
            # Neumann BC: 零通量邊界（梯度為零）
            loss_bc = 0.1 * (drho_dx[0]**2 + drho_dx[-1]**2)
            
            # === 11. Reaction consistency loss ===
            # 讓反應網路學到的 r 與 exact reaction 接近
            r_exact = exact_reaction(rho_current, p['k_plus'], p['k_minus'])
            loss_reaction = torch.mean((r_next.squeeze() - r_exact.detach())**2)
            
            # Accumulate losses
            total_loss_pde += loss_pde
            total_loss_bc += loss_bc
            total_loss_reaction += loss_reaction
            
            # Update for next iteration (detach to avoid backprop through time)
            rho_current = rho_next.detach()
            R_current = R_next.detach()
            J_current = J_next.detach()
        
        # === 8. Total loss ===
        total_loss = (
            t['weight_pde'] * total_loss_pde + 
            t['weight_bc'] * total_loss_bc + 
            t['weight_reaction'] * total_loss_reaction
        )
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        loss_history['total'].append(total_loss.item())
        loss_history['pde'].append(total_loss_pde.item())
        loss_history['bc'].append(total_loss_bc.item())
        loss_history['reaction'].append(total_loss_reaction.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{t['epochs']}]")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  PDE: {total_loss_pde.item():.6f}, BC: {total_loss_bc.item():.6f}, Reaction: {total_loss_reaction.item():.6f}")
    
    print("\n✅ Training complete!")
    
    return velocity_net, reaction_net, loss_history, x, time_steps, device


# ===========================
# Evaluation Function
# ===========================

def evaluate_diffusion_reaction(velocity_net, reaction_net, x, time_steps, config, device):
    """Generate results in evaluation mode."""
    
    p = config['physics']
    DT = (p['t_max'] - p['t_min']) / (p['nt'] - 1)
    dx = (p['x_max'] - p['x_min']) / (p['nx'] - 1)
    nu = p.get('nu', 0.01)
    
    print("\nGenerating results and creating visualizations...\n")
    
    velocity_net.eval()
    reaction_net.eval()
    
    # Storage for histories
    rho_numerical_history = []
    rho_exact_history = []
    u_history = []
    r_history = []
    J_history = []
    
    # Initialize
    rho_0 = initial_condition(x).detach()
    R_current = torch.zeros_like(rho_0)
    J_current = torch.ones_like(rho_0)
    rho_current = (rho_0 + R_current) / J_current
    
    # Store initial state
    rho_numerical_history.append(rho_current.cpu().numpy())
    rho_exact_history.append(exact_traveling_wave(x, torch.tensor([p['t_min']], device=device)[0]).cpu().numpy())
    J_history.append(J_current.cpu().numpy())
    
    # Time marching (evaluation mode)
    for n in range(p['nt'] - 1):
        t_current = time_steps[n+1]  # Current time for exact solution
        
        with torch.no_grad():
            net_input = torch.cat([x, rho_current], dim=1)
            
            # Predict velocity and reaction rate
            u_next = velocity_net(net_input)
            r_next = reaction_net(net_input)
            
            # Update R: Rⁿ⁺¹ = Rⁿ + Δt·r
            R_next = R_current + DT * r_next
            
            # Compute ∂u/∂x for Jacobian
            du_dx = torch.zeros_like(u_next)
            du_dx[1:-1] = (u_next[2:] - u_next[:-2]) / (2 * dx)
            du_dx[0] = du_dx[1]
            du_dx[-1] = du_dx[-2]
            
            # Compute ∂J/∂x for Jacobian
            dJ_dx = torch.zeros_like(J_current)
            dJ_dx[1:-1] = (J_current[2:] - J_current[:-2]) / (2 * dx)
            dJ_dx[0] = dJ_dx[1]
            dJ_dx[-1] = dJ_dx[-2]
            
            # Update Jacobian: Jⁿ⁺¹ = Jⁿ + Δt·(∇·u)Jⁿ - u·∇J
            J_next = J_current + DT * (du_dx * J_current - u_next * dJ_dx)
            
            # Update density: ρⁿ⁺¹ = (ρ₀ + Rⁿ⁺¹) / Jⁿ⁺¹
            eps = 1e-8
            J_next_safe = torch.clamp(J_next, min=eps)
            rho_next = (rho_0 + R_next) / J_next_safe
            
            # Store results
            rho_numerical_history.append(rho_next.cpu().numpy())
            rho_exact_history.append(exact_traveling_wave(x, t_current).cpu().numpy())
            u_history.append(u_next.cpu().numpy())
            r_history.append(r_next.cpu().numpy())
            J_history.append(J_next.cpu().numpy())
            
            # Update for next iteration
            rho_current = rho_next
            R_current = R_next
            J_current = J_next
    
    print("✅ Solution generated.")
    print(f"   Final ρ (numerical) range: [{rho_numerical_history[-1].min():.4f}, {rho_numerical_history[-1].max():.4f}]")
    print(f"   Final ρ (exact) range: [{rho_exact_history[-1].min():.4f}, {rho_exact_history[-1].max():.4f}]")
    
    return rho_numerical_history, rho_exact_history, u_history, r_history, J_history


# ===========================
# Additional Analysis
# ===========================

def perform_full_analysis(x_np, u_history, r_history, rho_numerical_history, rho_exact_history,
                         time_steps_np, reaction_net, config, device, output_dir):
    """Perform comprehensive analysis including reaction comparison and mass conservation."""
    
    p = config['physics']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select time indices for visualization
    time_indices = [0, len(u_history)//3, 2*len(u_history)//3, len(u_history)-1]
    
    for idx in time_indices:
        if idx < len(u_history):
            t_val = time_steps_np[idx+1]  # +1 because we didn't store initial u,r
            axes[0, 0].plot(x_np, u_history[idx], label=f't={t_val:.2f}')
            axes[0, 1].plot(x_np, r_history[idx], label=f't={t_val:.2f}')
    
    axes[0, 0].set_title('Velocity Field u(x,t)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Reaction Rate r(x,t)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('r')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Compare learned reaction with exact reaction
    rho_test = torch.linspace(0, 1, 100, device=device).view(-1, 1)
    x_test = 0.5 * torch.ones_like(rho_test)
    net_input_test = torch.cat([x_test, rho_test], dim=1)
    
    with torch.no_grad():
        r_learned = reaction_net(net_input_test).cpu().numpy()
        r_exact = exact_reaction(rho_test, p['k_plus'], p['k_minus']).cpu().numpy()
    
    rho_test_np = rho_test.cpu().numpy()
    axes[1, 0].plot(rho_test_np, r_exact, 'g-', linewidth=2, label='Exact: k⁺ρ - k⁻ρ²')
    axes[1, 0].plot(rho_test_np, r_learned, 'r--', linewidth=2, label='Learned r_NN(ρ)')
    axes[1, 0].set_title('Reaction Rate: Learned vs Exact')
    axes[1, 0].set_xlabel('ρ')
    axes[1, 0].set_ylabel('r(ρ)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 比較數值解與精確解的誤差
    x_flat = x_np.flatten()
    error_history = []
    for i in range(len(rho_numerical_history)):
        error = np.abs(rho_numerical_history[i].flatten() - rho_exact_history[i].flatten())
        # 使用 L2 誤差
        l2_error = np.sqrt(np.trapz(error**2, x_flat) if 'trapz' in dir(np) else np.mean(error**2))
        error_history.append(l2_error)
    
    axes[1, 1].plot(time_steps_np, error_history, 'b-', linewidth=2, label='L₂ error: ||ρ_num - ρ_exact||')
    axes[1, 1].set_title('Error Evolution: Numerical vs Exact Solution')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('L₂ Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis.png", dpi=150)
    plt.close()
    
    print("✅ Additional analysis complete!")
    print(f"   Initial error: {error_history[0]:.6e}")
    print(f"   Final error: {error_history[-1]:.6e}")
    print(f"   Max error: {max(error_history):.6e}")


# ===========================
# Main Function
# ===========================

def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RJ-net: Reaction-Jacobian Network')
    # Default config path is in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'config.yaml')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    print("="*60)
    print("RJ-net: Reaction-Jacobian Network")
    print("="*60)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = config['paths']['output_dir']
        # Modify for local execution (remove Google Drive path)
        if '/content/drive/MyDrive' in output_dir:
            output_dir = './results'
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n✅ Configuration loaded from: {args.config}")
    print(f"✅ Results will be saved to: {output_dir}\n")
    
    # Train the model
    velocity_net, reaction_net, loss_history, x, time_steps, device = train_rj_net(config, output_dir)
    
    # Evaluate the model
    rho_numerical_history, rho_exact_history, u_history, r_history, J_history = evaluate_diffusion_reaction(
        velocity_net, reaction_net, x, time_steps, config, device
    )
    
    # Convert to numpy for plotting
    x_np = x.cpu().detach().numpy()
    time_steps_np = time_steps.cpu().numpy()
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Plot loss history
    plotter.plot_loss(loss_history, config, f"{output_dir}/loss.png")
    
    # Create animation (now with exact solution for comparison)
    plotter.create_animation(
        x_np, rho_numerical_history, rho_exact_history, time_steps_np, config, 
        f"{output_dir}/animation.gif"
    )
    
    # Perform detailed analysis
    perform_full_analysis(
        x_np, u_history, r_history, rho_numerical_history, rho_exact_history,
        time_steps_np, reaction_net, config, device, output_dir
    )
    
    # Save configuration for reference
    with open(f"{output_dir}/config_used.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("\n" + "="*60)
    print("✅ All tasks complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("  - loss.png: Training loss evolution")
    print("  - animation.gif: Full time evolution")
    print("  - config_used.yaml: Configuration used for this run")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
