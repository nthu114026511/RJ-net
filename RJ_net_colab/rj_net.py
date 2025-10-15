#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RJ-net: Reaction-Jacobian Network for Reaction-Diffusion-Convection Systems

本程式實現了基於物理的神經網路（PINN）求解反應-擴散-對流系統：

    ρ_t + ∇·(ρu) = Δρ + Σr_i

核心方程：
    1. 密度更新: ρ^{n+1} = (ρ_0 + R^{n+1}) / J^{n+1}
    2. 速度場: u^{n+1} = u_NN(ρ^n)
    3. 反應率: r^{n+1} = r_NN(ρ^n)
    4. 反應累積: R^{n+1} = R^n + Δt·r^n
    5. Jacobian: J^{n+1} = J^n + Δt[(∇·u)J^n - u·∇J]

測試問題: Fisher-KPP 方程
    ρ_t = Δρ + k⁺ρ - k⁻ρ²
    其中 k⁺ρ - k⁻ρ² 是 logistic 反應項
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
    """Neural network to predict velocity field u(x, ρ)."""
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


def initial_condition(x):
    """Initial condition: Gaussian bump or sine wave."""
    # Option 1: Gaussian
    # return torch.exp(-50 * (x - 0.5)**2)
    
    # Option 2: Sine wave
    return 0.5 * (1 + torch.sin(2 * torch.pi * x))


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
    
    print(f"✅ Training setup complete.")
    print(f"   Spatial points: {p['nx']}")
    print(f"   Time steps: {p['nt']}")
    print(f"   dt = {DT:.6f}")
    print(f"   Network architecture: {t['hidden_size']} hidden units")
    print(f"   Reaction parameters: k⁺={p['k_plus']}, k⁻={p['k_minus']}")
    
    # ===========================
    # Training Loop
    # ===========================
    
    print("\nStarting training...\n")
    
    for epoch in range(t['epochs']):
        optimizer.zero_grad()
        
        # Initialize variables
        rho_0 = initial_condition(x).detach()  # Initial density (constant)
        rho_current = rho_0.clone()
        J_current = torch.ones_like(x, device=device)  # Initial Jacobian = 1
        R_current = torch.zeros_like(x, device=device)  # Initial reaction accumulation = 0
        
        total_loss = 0.0
        total_loss_pde = 0.0
        total_loss_bc = 0.0
        total_loss_reaction = 0.0
        
        # Time marching
        for n in range(p['nt'] - 1):
            # Prepare network input: [x, ρ]
            net_input = torch.cat([x, rho_current], dim=1)
            
            # === 1. Predict velocity and reaction rate ===
            u_current = velocity_net(net_input)
            r_current = reaction_net(net_input)
            
            # === 2. Update R (reaction accumulation) ===
            R_next = R_current + DT * r_current
            
            # === 3. Update J (Jacobian) ===
            # Compute ∂u/∂x
            du_dx = torch.autograd.grad(
                u_current, x, 
                grad_outputs=torch.ones_like(u_current), 
                create_graph=True
            )[0]
            
            # J^{n+1} = J^n * (1 + Δt * ∂u/∂x)
            # Simplified version (assuming u·∇J ≈ 0 for 1D)
            J_next = J_current * (1 + DT * du_dx)
            
            # === 4. Update ρ (density) ===
            # ρ^{n+1} = (ρ₀ + R^{n+1}) / J^{n+1}
            rho_next = (rho_0 + R_next) / J_next
            
            # === 5. Compute PDE residual ===
            # ρ_t ≈ (ρ^{n+1} - ρ^n) / Δt
            rho_t = (rho_next - rho_current) / DT
            
            # Compute Laplacian: Δρ = ∂²ρ/∂x²
            drho_dx = torch.autograd.grad(
                rho_next, x, 
                grad_outputs=torch.ones_like(rho_next), 
                create_graph=True
            )[0]
            
            d2rho_dx2 = torch.autograd.grad(
                drho_dx, x, 
                grad_outputs=torch.ones_like(drho_dx), 
                create_graph=True
            )[0]
            
            # Exact reaction term for comparison
            r_exact = exact_reaction(rho_next, p['k_plus'], p['k_minus'])
            
            # PDE residual: ρ_t = Δρ + r(ρ)
            pde_residual = rho_t - d2rho_dx2 - r_exact
            loss_pde = torch.mean(pde_residual**2)
            
            # === 6. Boundary conditions ===
            # Homogeneous Neumann BC: ∂ρ/∂x = 0 at boundaries
            loss_bc = drho_dx[0]**2 + drho_dx[-1]**2
            
            # === 7. Reaction loss ===
            # Encourage reaction network to learn the correct reaction term
            loss_reaction = torch.mean((r_current - r_exact.detach())**2)
            
            # Accumulate losses
            total_loss_pde += loss_pde
            total_loss_bc += loss_bc
            total_loss_reaction += loss_reaction
            
            # Update for next iteration (detach to avoid backprop through time)
            rho_current = rho_next.detach()
            rho_current.requires_grad = True
            J_current = J_next.detach()
            R_current = R_next.detach()
        
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

def evaluate_rj_net(velocity_net, reaction_net, x, time_steps, config, device):
    """Generate results in evaluation mode."""
    
    p = config['physics']
    DT = (p['t_max'] - p['t_min']) / (p['nt'] - 1)
    
    print("\nGenerating results and creating visualizations...\n")
    
    velocity_net.eval()
    reaction_net.eval()
    
    # Storage for histories
    rho_numerical_history = []
    R_history = []
    J_history = []
    u_history = []
    r_history = []
    
    # Initialize
    rho_0 = initial_condition(x).detach()
    rho_current = rho_0.clone()
    J_current = torch.ones_like(x, device=device)
    R_current = torch.zeros_like(x, device=device)
    
    # Store initial state
    rho_numerical_history.append(rho_current.cpu().numpy())
    R_history.append(R_current.cpu().numpy())
    J_history.append(J_current.cpu().numpy())
    
    # Time marching (evaluation mode)
    with torch.no_grad():
        for n in range(p['nt'] - 1):
            net_input = torch.cat([x, rho_current], dim=1)
            
            # Predict
            u_current = velocity_net(net_input)
            r_current = reaction_net(net_input)
            
            # Update R
            R_next = R_current + DT * r_current
            
            # Update J (需要計算梯度)
            x_temp = x.clone()
            x_temp.requires_grad = True
            rho_temp = rho_current.clone()
            net_input_temp = torch.cat([x_temp, rho_temp], dim=1)
            u_temp = velocity_net(net_input_temp)
            
            du_dx = torch.autograd.grad(
                u_temp, x_temp, 
                grad_outputs=torch.ones_like(u_temp), 
                create_graph=False
            )[0]
            
            J_next = J_current * (1 + DT * du_dx)
            
            # Update ρ
            rho_next = (rho_0 + R_next) / J_next
            
            # Store
            rho_numerical_history.append(rho_next.cpu().numpy())
            R_history.append(R_next.cpu().numpy())
            J_history.append(J_next.cpu().numpy())
            u_history.append(u_current.cpu().numpy())
            r_history.append(r_current.cpu().numpy())
            
            # Update for next iteration
            rho_current = rho_next
            J_current = J_next
            R_current = R_next
    
    print("✅ Solution generated.")
    print(f"   Final ρ range: [{rho_numerical_history[-1].min():.4f}, {rho_numerical_history[-1].max():.4f}]")
    print(f"   Final R range: [{R_history[-1].min():.4f}, {R_history[-1].max():.4f}]")
    print(f"   Final J range: [{J_history[-1].min():.4f}, {J_history[-1].max():.4f}]")
    
    return rho_numerical_history, R_history, J_history, u_history, r_history


# ===========================
# Additional Analysis
# ===========================

def perform_analysis(x_np, u_history, r_history, time_steps_np, config, device, output_dir):
    """Perform additional analysis on learned velocity and reaction fields."""
    
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
    
    # Need to access reaction_net - pass it as parameter or make it accessible
    # For now, we'll skip this part in the standalone function
    
    axes[1, 0].set_title('Reaction Rate: Learned vs Exact')
    axes[1, 0].set_xlabel('ρ')
    axes[1, 0].set_ylabel('r(ρ)')
    axes[1, 0].grid(True)
    
    # Mass conservation check: M(t) = ∫ρ dx
    # This requires rho_numerical_history which should be passed in
    axes[1, 1].set_title('Total Mass M(t) = ∫ρ dx')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('M(t)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis.png", dpi=150)
    plt.close()
    
    print("✅ Additional analysis complete!")


def perform_full_analysis(x_np, u_history, r_history, rho_numerical_history, 
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
    
    # Mass conservation check: M(t) = ∫ρ dx
    mass_history = [np.trapz(rho, x_np.flatten()) for rho in rho_numerical_history]
    axes[1, 1].plot(time_steps_np, mass_history, 'b-', linewidth=2)
    axes[1, 1].set_title('Total Mass M(t) = ∫ρ dx')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('M(t)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis.png", dpi=150)
    plt.close()
    
    print("✅ Additional analysis complete!")
    print(f"   Mass conservation: M(0)={mass_history[0]:.4f}, M(T)={mass_history[-1]:.4f}")
    print(f"   Relative change: {abs(mass_history[-1] - mass_history[0])/mass_history[0]*100:.2f}%")


# ===========================
# Main Function
# ===========================

def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RJ-net: Reaction-Jacobian Network')
    parser.add_argument('--config', type=str, default='config.yaml',
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
    rho_numerical_history, R_history, J_history, u_history, r_history = evaluate_rj_net(
        velocity_net, reaction_net, x, time_steps, config, device
    )
    
    # Convert to numpy for plotting
    x_np = x.cpu().detach().numpy()
    time_steps_np = time_steps.cpu().numpy()
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Plot loss history
    plotter.plot_loss(loss_history, config, f"{output_dir}/loss.png")
    
    # Plot comparison
    plotter.plot_comparison(x_np, rho_numerical_history, config, f"{output_dir}/comparison.png")
    
    # Plot R and J evolution
    plotter.plot_reaction_jacobian(x_np, R_history, J_history, config, f"{output_dir}/reaction_jacobian.png")
    
    # Create animation
    plotter.create_animation(
        x_np, rho_numerical_history, time_steps_np, config, 
        f"{output_dir}/animation.gif",
        R_history=R_history, J_history=J_history
    )
    
    # Perform additional analysis
    perform_full_analysis(
        x_np, u_history, r_history, rho_numerical_history,
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
    print("  - comparison.png: Initial vs final states")
    print("  - reaction_jacobian.png: R and J evolution")
    print("  - animation.gif: Full time evolution")
    print("  - analysis.png: Velocity, reaction, and mass analysis")
    print("  - config_used.yaml: Configuration used for this run")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
