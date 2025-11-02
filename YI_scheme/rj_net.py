#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本程式實現了基於物理的神經網路（PINN）求解擴散-反應系統：

    ∂ρ/∂t = ∇²ρ + r(x, ρ)

測試問題: 擴散-反應系統
    ∂ρ/∂t = ∂²ρ/∂x² + k⁺ρ - k⁻ρ²
    其中 k⁺ρ - k⁻ρ² 是反應項
    
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
import time
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

def exact_traveling_wave(x, t, x0=0.0):
    """
    Fisher-KPP 精確旅行波解：
    
    ρ(x, t) = [1 + exp(-(x - st - x0)/√6)]^{-2}
    
    其中波速 s = 5/√6
    
    註：當 t=0 時，此函數即為初始條件
    """
    sqrt6 = torch.sqrt(torch.tensor(6.0, device=x.device, dtype=x.dtype))
    s = 5.0 / sqrt6  # 波速
    xi = (x - s * t - x0) / (sqrt6)  # 行進波座標
    return (1.0 + torch.exp(-xi))**(-2)


# ===========================
# Training Function
# ===========================

def train_rj_net(config):
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
        'ic': []
    }
    
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
        
        # Initialize variables according to: ρ = (ρ₀ + R)*I
        rho_0 = exact_traveling_wave(x, torch.tensor(0.0, device=x.device)).detach()  # Initial density ρ₀
        R_current = torch.zeros_like(rho_0)     # R(t=0) = 0
        I_current = torch.ones_like(rho_0)      # I(t=0) = 1
        Y_current = x.clone().detach()          # Y(x, 0) = x
        rho_current = (rho_0 + R_current) * I_current

        total_loss = 0.0
        total_loss_pde = 0.0
        total_loss_ic = 0.0
        
        # === Initial condition loss ===
        # 確保初始條件精確
        rho_ic_exact = exact_traveling_wave(x, torch.tensor([p['t_min']], device=device)[0]).detach()
        loss_ic = torch.mean((rho_current - rho_ic_exact)**2)
        total_loss_ic += loss_ic
        
        # Time marching: For n=0,1,2,...
        for n in range(p['nt'] - 1):
            # Prepare network input: [x, ρⁿ]
            net_input = torch.cat([x, rho_current], dim=1)
            
            # === Predict velocity uⁿ⁺¹ = u_NN(ρⁿ) ===
            u_next = velocity_net(net_input)
            
            # === Predict reaction rate rⁱⁿ⁺¹ = r_NN(ρⁿ, x) ===
            r_next = reaction_net(net_input)
            
            # === Update reaction integral: Rⁱⁿ⁺¹ = Rⁱⁿ + Δt·rⁱⁿ ===
            R_next = R_current + DT * r_next

            # === Update I: Iⁿ⁺¹ = Iⁿ - Δt·∇·(Iⁿuⁿ) ===
            # Compute ∇·(I*u) using autograd (精確的自動微分)
            # This represents the divergence of the Jacobian flux
            Iu = I_current * u_next
            div_Iu = torch.autograd.grad(Iu.sum(), x, create_graph=True)[0]
            
            I_next = I_current - DT * div_Iu
            
            # === Compute ∇·u using autograd ===
            # ∂u/∂x using autograd (精確的自動微分)
            du_dx = torch.autograd.grad(u_next, x, grad_outputs=torch.ones_like(u_next), create_graph=True)[0]
            
            # === Compute ∇·(Y*u) using autograd ===
            # Method: Compute (Y*u) first, then take derivative
            Yu = Y_current * u_next
            div_Yu = torch.autograd.grad(Yu.sum(), x, create_graph=True)[0]
            

            # === Update Y according to: ∂Y/∂t + ∇·(Y*u) = Y(∇·u) ===
            # Rearranged: ∂Y/∂t = Y(∇·u) - ∇·(Y*u)
            # Time discretization: Y^{n+1} = Y^n - Δt·[∇·(Y*u) + Y*(∇·u)]
            # Where: ∇·(Y*u) = ∂(Y*u)/∂x (computed using autograd as div_Yu)
            #        ∇·u = ∂u/∂x (computed using autograd as du_dx)
            Y_next = Y_current - DT * (div_Yu + Y_current * du_dx)
            

            rho_0_Y = exact_traveling_wave(Y_next, time_steps[n+1]).detach()  # Update ρ₀(Y, t)

            # === Update density: ρⁿ⁺¹ = (ρ₀。Y + Rⁿ⁺¹) * Iⁿ⁺¹ ===
            rho_next = (rho_0_Y + R_next) * I_next

            # === Compute spatial derivatives for PDE residual ===
            # Second derivative ∂²ρ/∂x² (for diffusion term)
            drho_dx = torch.autograd.grad(rho_next, x, grad_outputs=torch.ones_like(rho_next), create_graph=True)[0]
            d2rho_dx2 = torch.autograd.grad(drho_dx, x, grad_outputs=torch.ones_like(drho_dx), create_graph=True)[0]
            
            # === Compute time derivative ===
            rho_t = (rho_next - rho_current) / DT
            
            # === Compute PDE residual ===
            # PDE: ∂ρ/∂t = ∂²ρ/∂x² + ρ - ρ²
            # Residual: ∂ρ/∂t - ∂²ρ/∂x² - (ρ - ρ²) ≈ 0
            diffusion_term = d2rho_dx2
            reaction_term = rho_next - rho_next**2  # Fisher-KPP reaction: ρ(1-ρ)
            pde_residual = rho_t - diffusion_term - reaction_term
            loss_pde = torch.mean(pde_residual**2)
            

            total_loss_pde += loss_pde

            rho_current = rho_next.detach()
            R_current = R_next.detach()
            I_current = I_next.detach()
        
        # === 8. Total loss ===
        total_loss = (
            t['weight_pde'] * total_loss_pde + 
            t['weight_ic'] * total_loss_ic
        )
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        loss_history['total'].append(total_loss.item())
        loss_history['pde'].append(total_loss_pde.item())
        loss_history['ic'].append(total_loss_ic.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{t['epochs']}]")
            print(f"  Total Loss: {total_loss.item():.6e}")
            print(f"  PDE: {total_loss_pde.item():.6e}, IC: {total_loss_ic.item():.6e}")
    
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
    
    print("\nGenerating results and creating visualizations...\n")
    
    velocity_net.eval()
    reaction_net.eval()
    
    # Storage for histories (only keep what we need for plotting)
    rho_numerical_history = []
    rho_exact_history = []
    
    # Initialize variables according to: ρ = (ρ₀ + R)*I
    rho_0 = exact_traveling_wave(x, torch.tensor(0.0, device=x.device)).detach()  # Initial density ρ₀
    R_current = torch.zeros_like(rho_0)     # R(t=0) = 0
    I_current = torch.ones_like(rho_0)      # I(t=0) = 1
    Y_current = x.clone().detach()          # Y(x, 0) = x
    rho_current = (rho_0 + R_current) * I_current
    
    # Store initial state
    rho_numerical_history.append(rho_current.cpu().numpy())
    rho_exact_history.append(exact_traveling_wave(x, torch.tensor([p['t_min']], device=device)[0]).detach().cpu().numpy())
    
    # Time marching (evaluation mode)
    for n in range(p['nt'] - 1):
        t_current = time_steps[n+1]  # Current time for exact solution
        
        net_input = torch.cat([x, rho_current], dim=1)
        
        # === 1. Predict velocity uⁿ⁺¹ = u_NN(ρⁿ) ===
        u_next = velocity_net(net_input)
        
        # === 2. Predict reaction rate rⁿ⁺¹ = r_NN(ρⁿ, x) ===
        r_next = reaction_net(net_input)
        
        # === 3. Update reaction integral: Rⁿ⁺¹ = Rⁿ + Δt·rⁿ ===
        R_next = R_current + DT * r_next
        
        # === 4. Update I: Iⁿ⁺¹ = Iⁿ - Δt·∇·(Iⁿuⁿ) ===
        # Compute ∇·(I*u) using finite difference
        Iu = I_current * u_next
        div_Iu = torch.zeros_like(I_current)
        div_Iu[1:-1] = (Iu[2:] - Iu[:-2]) / (2 * dx)
        div_Iu[0] = (Iu[1] - Iu[0]) / dx
        div_Iu[-1] = (Iu[-1] - Iu[-2]) / dx
        
        I_next = I_current - DT * div_Iu
        
        # === 5. Compute ∇·u and ∇·(Y*u) for Y update ===
        # ∂u/∂x using autograd
        du_dx = torch.autograd.grad(u_next, x, grad_outputs=torch.ones_like(u_next), create_graph=False, retain_graph=True)[0]
        
        # === Compute ∇·(Y*u) using autograd ===
        # Method: Compute (Y*u) first, then take derivative
        Yu = Y_current * u_next
        div_Yu = torch.autograd.grad(Yu.sum(), x, create_graph=False)[0]
        
        # === Update Y according to: ∂Y/∂t + ∇·(Y*u) = Y(∇·u) ===
        # Rearranged: ∂Y/∂t = Y(∇·u) - ∇·(Y*u)
        # Time discretization: Y^{n+1} = Y^n - Δt·[∇·(Y*u) + Y*(∇·u)]
        Y_next = Y_current - DT * (div_Yu + Y_current * du_dx)
        
        rho_0_Y = exact_traveling_wave(Y_next, time_steps[n+1]).detach()  # Update ρ₀ for current time
        
        # === 6. Update density: ρⁿ⁺¹ = (ρ₀ + Rⁿ⁺¹) * Iⁿ⁺¹ ===
        rho_next = (rho_0_Y + R_next) * I_next
        
        # Store results (only density for plotting)
        rho_numerical_history.append(rho_next.detach().cpu().numpy())
        rho_exact_history.append(exact_traveling_wave(x, t_current).detach().cpu().numpy())
        
        # Update for next iteration
        rho_current = rho_next.detach()
        R_current = R_next.detach()
        I_current = I_next.detach()
        Y_current = Y_next.detach()
    
    print("✅ Solution generated.")
    print(f"   Final ρ (numerical) range: [{rho_numerical_history[-1].min():.4f}, {rho_numerical_history[-1].max():.4f}]")
    print(f"   Final ρ (exact) range: [{rho_exact_history[-1].min():.4f}, {rho_exact_history[-1].max():.4f}]")
    
    return rho_numerical_history, rho_exact_history


# ===========================
# Main Function
# ===========================

def main():
    """Main execution function."""
    
    # Start overall timer
    overall_start_time = time.time()
    
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
    
    # Train the model with timer
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    training_start_time = time.time()
    velocity_net, reaction_net, loss_history, x, time_steps, device = train_rj_net(config)
    training_time = time.time() - training_start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    
    # Evaluate the model
    print("="*60)
    print("EVALUATION PHASE")
    print("="*60)
    evaluation_start_time = time.time()
    rho_numerical_history, rho_exact_history = evaluate_diffusion_reaction(
        velocity_net, reaction_net, x, time_steps, config, device
    )
    evaluation_time = time.time() - evaluation_start_time
    print(f"✅ Evaluation completed in {evaluation_time:.2f} seconds\n")
    
    # Convert to numpy for plotting
    x_np = x.cpu().detach().numpy()
    time_steps_np = time_steps.cpu().numpy()
    
    # Create visualizations
    print("="*60)
    print("VISUALIZATION PHASE")
    print("="*60)
    visualization_start_time = time.time()
    print("Creating visualizations...")
    
    # Plot loss history
    plotter.plot_loss(loss_history, config, f"{output_dir}/loss.png")
    
    # Create animation (now with exact solution for comparison)
    plotter.create_animation(
        x_np, rho_numerical_history, rho_exact_history, time_steps_np, config, 
        f"{output_dir}/animation.gif"
    )
    visualization_time = time.time() - visualization_start_time
    print(f"✅ Visualization completed in {visualization_time:.2f} seconds\n")
    
    # Save configuration for reference
    with open(f"{output_dir}/config_used.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Calculate and display total execution time
    overall_time = time.time() - overall_start_time
    
    print("\n" + "="*60)
    print("✅ All tasks complete!")
    print("="*60)
    print("\n⏱️  EXECUTION TIME SUMMARY:")
    print("-" * 60)
    print(f"  Training phase:      {training_time:8.2f} seconds ({training_time/overall_time*100:5.1f}%)")
    print(f"  Evaluation phase:    {evaluation_time:8.2f} seconds ({evaluation_time/overall_time*100:5.1f}%)")
    print(f"  Visualization phase: {visualization_time:8.2f} seconds ({visualization_time/overall_time*100:5.1f}%)")
    print("-" * 60)
    print(f"  TOTAL:               {overall_time:8.2f} seconds ({overall_time/60:5.2f} minutes)")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("  - loss.png: Training loss evolution")
    print("  - animation.gif: Full time evolution")
    print("  - config_used.yaml: Configuration used for this run")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
