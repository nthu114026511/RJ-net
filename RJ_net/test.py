#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RJ-net (Y, I, R 版): Neural Characteristics + Integrating Factor + PINN

原方程（無顯式對流）:
    ∂ρ/∂t = ν·∇²ρ + r(x, ρ)

本版改成以三個變量的「硬結構」來組合密度：
    Y_t + u·∇Y = 0                      （反向特徵標籤）
    I_t + ∇·(I u) = 0                   （積分因子 / 壓縮因子；保體式）
    R_t + u·∇R = r(x, ρ)/I              （反應沿特徵累積）
    ρ = (ρ0∘Y + R) * I

訓練殘差有兩種模式（由 config['physics']['hard_split'] 控制）：
  - hard_split=True:
        ρ 已硬式滿足對流 + 反應 ⇒ 殘差只需讓擴散對上：
        Residual = ρ_t - ν ρ_xx
  - hard_split=False:
        傳統 PINN 殘差（不吃掉反應）：
        Residual = ρ_t - ν ρ_xx - r(x, ρ)

其它新旗標（請在 config.yaml 加入或沿用預設）：
    physics:
      use_u: true/false         # 是否學習流場 u（若 PDE 無對流，建議 false）
      hard_split: true/false    # 是否使用硬拆分只留擴散殘差
      reaction: "nn" | "logistic"  # 反應來源：神經網路 或 顯式 logistic
      k_plus: 1.0               # logistic 參數（若 reaction="logistic" 才用）
      k_minus: 1.0
      nu: 1.0                   # 擴散係數 ν
    training:
      hidden_size: 64
      epochs: 2000
      learning_rate: 1e-3
      weight_pde: 1.0
      weight_bc: 0.0
      weight_ic: 1.0
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

# 自訂視覺化（沿用你的 module）
import plotter


# ===========================
# Networks
# ===========================

class VelocityNet(nn.Module):
    """u_NN(ρ, x) → u"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x_rho):
        return self.network(x_rho)

class ReactionNet(nn.Module):
    """r_NN(ρ, x) → r"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x_rho):
        return self.network(x_rho)


# ===========================
# AD helpers (1D)
# ===========================

def grad_x(f, x, create_graph=True):
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f),
        create_graph=create_graph, retain_graph=True, only_inputs=True
    )[0]

def laplace_x(f, x, create_graph=True):
    dfdx = grad_x(f, x, create_graph=create_graph)
    d2fdx2 = grad_x(dfdx, x, create_graph=create_graph)
    return d2fdx2

def div_flux_product(phi, u, theta, x, create_graph=True):
    # ∇·(phi u) = u · ∇phi + phi * theta    (1D: u*dphi/dx + phi*theta)
    dphidx = grad_x(phi, x, create_graph=create_graph)
    return u * dphidx + phi * theta


# ===========================
# Problem data
# ===========================

def exact_reaction_logistic(rho, k_plus, k_minus):
    return k_plus * rho - k_minus * rho**2

def exact_traveling_wave(x, t, x0=0.5):
    # Fisher-KPP Ablowitz–Zeppetella 解析波形（D=1, r=ρ(1-ρ)）
    sqrt6 = torch.sqrt(torch.tensor(6.0, device=x.device, dtype=x.dtype))
    s = 5.0 / sqrt6
    xi = (x - s * t - x0) / sqrt6
    return (1.0 + torch.exp(-xi))**(-2)

def initial_condition(x, x0=0.5):
    return exact_traveling_wave(x, torch.tensor(0.0, device=x.device, dtype=x.dtype), x0=x0)


# ===========================
# Training
# ===========================

def train_rj_net(config, output_dir):
    p = config['physics']; t = config['training']
    DT = (p['t_max'] - p['t_min']) / (p['nt'] - 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 空間網格
    x = torch.linspace(p['x_min'], p['x_max'], p['nx'], device=device).view(-1, 1)
    x.requires_grad_(True)
    dx = (p['x_max'] - p['x_min']) / (p['nx'] - 1)

    # 時間節點
    time_steps = torch.linspace(p['t_min'], p['t_max'], p['nt'], device=device)

    # 網路
    velocity_net = VelocityNet(hidden_size=t['hidden_size']).to(device)
    reaction_net = ReactionNet(hidden_size=t['hidden_size']).to(device)

    # 參數與優化器
    params = list(reaction_net.parameters())
    if p.get('use_u', False):
        params += list(velocity_net.parameters())
    optimizer = torch.optim.Adam(params, lr=t['learning_rate'])

    # loss 紀錄
    loss_history = {'total': [], 'pde': [], 'ic': []}

    print("✅ Training setup complete.")
    print(f"   nx={p['nx']}, nt={p['nt']}, dt={DT:.4e}, dx={dx:.4e}")
    print(f"   use_u={p.get('use_u', False)}, hard_split={p.get('hard_split', True)}, reaction={p.get('reaction','nn')}")
    print(f"   ν={p.get('nu',1.0)}, k+={p.get('k_plus',1.0)}, k-={p.get('k_minus',1.0)}")

    for epoch in range(t['epochs']):
        optimizer.zero_grad(set_to_none=True)

        # 初始條件
        rho0 = initial_condition(x).detach()
        Y = x.detach().clone().requires_grad_(True)        # Y^0 = x
        I = torch.ones_like(x, requires_grad=True)         # I^0 = 1
        R = torch.zeros_like(x, requires_grad=True)        # R^0 = 0

        # 初值殘差（強化 IC）
        rho_init = (initial_condition(Y) + R) * I          # t=0 的組合（理想應回到 rho0）
        loss_ic = torch.mean((rho_init - rho0)**2)

        rho = rho_init
        pde_loss = 0.0

        # 時間步進
        for n in range(p['nt'] - 1):
            # 取流場 u 與其散度 theta（1D：theta = du/dx）
            if p.get('use_u', False):
                u = velocity_net(torch.cat([x, rho.detach()], dim=1))  # 可視需要改不 detach
            else:
                u = torch.zeros_like(x)

            theta = grad_x(u, x, create_graph=True)

            # --- 更新 Y：Y^{n+1} = Y^n - dt * u * dY/dx
            dYdx = grad_x(Y, x, create_graph=True)
            Y_next = Y - DT * (u * dYdx)

            # --- 更新 I：I^{n+1} = I^n - dt * div(I u)
            div_Iu = div_flux_product(I, u, theta, x, create_graph=True)
            I_next = I - DT * div_Iu

            # --- 反應 r(ρ,x)
            if p.get('reaction', 'nn') == 'logistic':
                r_val = exact_reaction_logistic(rho, p.get('k_plus', 1.0), p.get('k_minus', 1.0))
            else:
                r_val = reaction_net(torch.cat([x, rho], dim=1))

            # --- 更新 R：R^{n+1} = R^n - dt * u * dR/dx + dt * r/I
            dRdx = grad_x(R, x, create_graph=True)
            R_next = R - DT * (u * dRdx) + DT * (r_val / (I + 1e-8))

            # --- 組合 ρ^{n+1}：ρ = (ρ0∘Y + R) * I
            Y_safe = torch.clamp(Y_next, p['x_min'], p['x_max'])
            rho_next = (initial_condition(Y_safe) + R_next) * I_next

            # --- 時間導數與拉普拉斯
            rho_t = (rho_next - rho) / DT
            rho_xx = laplace_x(rho_next, x, create_graph=True)

            # --- 殘差
            if p.get('hard_split', True):
                # 只留擴散：ρ_t - ν ρ_xx = 0
                res = rho_t - p.get('nu', 1.0) * rho_xx
            else:
                # 全殘差：ρ_t - ν ρ_xx - r = 0
                res = rho_t - p.get('nu', 1.0) * rho_xx - r_val

            pde_loss = pde_loss + torch.mean(res**2)

            # rollout
            Y, I, R, rho = Y_next, I_next, R_next, rho_next

        total_loss = t['weight_pde'] * pde_loss + t['weight_ic'] * loss_ic
        total_loss.backward()
        optimizer.step()

        loss_history['total'].append(float(total_loss.detach().cpu()))
        loss_history['pde'].append(float(pde_loss.detach().cpu()))
        loss_history['ic'].append(float(loss_ic.detach().cpu()))

        if (epoch + 1) % max(1, (t['epochs']//20)) == 0:
            print(f"[{epoch+1:04d}] total={total_loss.item():.3e}  pde={pde_loss.item():.3e}  ic={loss_ic.item():.3e}")

    print("\n✅ Training complete!")
    return velocity_net, reaction_net, loss_history, x, time_steps, device


# ===========================
# Evaluation
# ===========================

def evaluate_diffusion_reaction(velocity_net, reaction_net, x, time_steps, config, device):
    p = config['physics']
    DT = (p['t_max'] - p['t_min']) / (p['nt'] - 1)

    print("\nGenerating results and creating visualizations...\n")
    velocity_net.eval(); reaction_net.eval()

    rho_hist, rho_exact_hist, u_hist, r_hist, I_hist = [], [], [], [], []

    # 初始化
    rho0 = initial_condition(x).detach()
    Y = x.detach().clone()
    I = torch.ones_like(x)
    R = torch.zeros_like(x)
    rho = (initial_condition(Y) + R) * I

    rho_hist.append(rho.cpu().numpy())
    rho_exact_hist.append(exact_traveling_wave(x, time_steps[0]).detach().cpu().numpy())
    I_hist.append(I.cpu().numpy())

    for n in range(p['nt'] - 1):
        with torch.no_grad():
            # 流場
            if p.get('use_u', False):
                u = velocity_net(torch.cat([x, rho], dim=1))
            else:
                u = torch.zeros_like(x)
            theta = grad_x(u, x, create_graph=False)

            # 更新 Y, I, R
            dYdx = grad_x(Y, x, create_graph=False)
            Y = Y - DT * (u * dYdx)

            div_Iu = u * grad_x(I, x, create_graph=False) + I * theta
            I = I - DT * div_Iu

            if p.get('reaction', 'nn') == 'logistic':
                r_val = exact_reaction_logistic(rho, p.get('k_plus', 1.0), p.get('k_minus', 1.0))
            else:
                r_val = reaction_net(torch.cat([x, rho], dim=1))

            dRdx = grad_x(R, x, create_graph=False)
            R = R - DT * (u * dRdx) + DT * (r_val / (I + 1e-8))

            # 重建 ρ
            Y = torch.clamp(Y, p['x_min'], p['x_max'])
            rho = (initial_condition(Y) + R) * I

            # 存檔
            rho_hist.append(rho.cpu().numpy())
            rho_exact_hist.append(exact_traveling_wave(x, time_steps[n+1]).detach().cpu().numpy())
            u_hist.append(u.cpu().numpy())
            r_hist.append(r_val.cpu().numpy())
            I_hist.append(I.cpu().numpy())

    print("✅ Solution generated.")
    print(f"   Final ρ range: [{rho_hist[-1].min():.4f}, {rho_hist[-1].max():.4f}]")
    return rho_hist, rho_exact_hist, u_hist, r_hist, I_hist


# ===========================
# Analysis (同你的風格)
# ===========================

def perform_full_analysis(x_np, u_hist, r_hist, rho_hist, rho_exact_hist,
                          time_steps_np, reaction_net, config, device, output_dir):
    p = config['physics']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tidxs = [0, max(0, len(u_hist)//3-1), max(0, 2*len(u_hist)//3-1), max(0, len(u_hist)-1)]
    for idx in tidxs:
        if len(u_hist) == 0: break
        t_val = time_steps_np[idx+1]
        axes[0, 0].plot(x_np, u_hist[idx].squeeze(), label=f't={t_val:.2f}')
        axes[0, 1].plot(x_np, r_hist[idx].squeeze(), label=f't={t_val:.2f}')

    axes[0, 0].set_title('Velocity Field u(x,t)'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].set_title('Reaction Rate r(x,t)');  axes[0, 1].legend(); axes[0, 1].grid(True)

    # r(ρ) 曲線對照
    rho_test = torch.linspace(0, 1, 200, device=device).view(-1, 1)
    x_test = 0.5 * torch.ones_like(rho_test)
    with torch.no_grad():
        r_learn = reaction_net(torch.cat([x_test, rho_test], dim=1)).cpu().numpy()
        r_ex = exact_reaction_logistic(rho_test, p.get('k_plus',1.0), p.get('k_minus',1.0)).cpu().numpy()

    axes[1, 0].plot(rho_test.cpu().numpy(), r_ex, 'g-', lw=2, label='Exact k⁺ρ - k⁻ρ²')
    axes[1, 0].plot(rho_test.cpu().numpy(), r_learn, 'r--', lw=2, label='Learned r_NN')
    axes[1, 0].set_title('Reaction Rate: Learned vs Exact'); axes[1, 0].legend(); axes[1, 0].grid(True)

    # 誤差演化
    x_flat = x_np.flatten()
    errs = []
    for i in range(len(rho_hist)):
        e = np.abs(rho_hist[i].flatten() - rho_exact_hist[i].flatten())
        errs.append(np.sqrt(np.trapezoid(e**2, x_flat)))
    axes[1, 1].plot(time_steps_np, errs, 'b-', lw=2, label='L2 error'); axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log'); axes[1, 1].legend(); axes[1, 1].set_title('Error Evolution')

    plt.tight_layout(); plt.savefig(f"{output_dir}/analysis.png", dpi=150); plt.close()

    print("✅ Additional analysis complete!")
    print(f"   Initial error: {errs[0]:.6e}")
    print(f"   Final error:   {errs[-1]:.6e}")
    print(f"   Max error:     {max(errs):.6e}")


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser(description='RJ-net (Y, I, R 版)')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, 'config.yaml')
    parser.add_argument('--config', type=str, default=default_config)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("="*60); print("RJ-net (Y, I, R 版)"); print("="*60)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = args.output or config['paths']['output_dir']
    if '/content/drive/MyDrive' in output_dir:
        output_dir = './results'
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"run_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n✅ Configuration: {args.config}")
    print(f"✅ Results to:    {output_dir}\n")

    print("="*60); print("TRAINING PHASE"); print("="*60)
    velocity_net, reaction_net, loss_history, x, time_steps, device = train_rj_net(config, output_dir)

    print("="*60); print("EVALUATION PHASE"); print("="*60)
    rho_hist, rho_exact_hist, u_hist, r_hist, I_hist = evaluate_diffusion_reaction(
        velocity_net, reaction_net, x, time_steps, config, device
    )

    x_np = x.detach().cpu().numpy()
    t_np = time_steps.detach().cpu().numpy()

    print("="*60); print("VISUALIZATION PHASE"); print("="*60)
    plotter.plot_loss(loss_history, config, f"{output_dir}/loss.png")
    plotter.create_animation(x_np, rho_hist, rho_exact_hist, t_np, config, f"{output_dir}/animation.gif")
    perform_full_analysis(x_np, u_hist, r_hist, rho_hist, rho_exact_hist, t_np, reaction_net, config, device, output_dir)

    with open(f"{output_dir}/config_used.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("\n" + "="*60)
    print("✅ All tasks complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("  - loss.png")
    print("  - animation.gif")
    print("  - analysis.png")
    print("  - config_used.yaml\n")

if __name__ == "__main__":
    main()
