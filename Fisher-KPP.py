import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime

def exact_traveling_wave(x, t, x0=0.0):
    """
    Fisher-KPP 精確旅行波解：
    
    ρ(x, t) = [1 + exp(-(x - st - x0)/√6)]^{-2}
    
    其中波速 s = 5/√6
    """
    sqrt6 = torch.sqrt(torch.tensor(6.0, device=x.device, dtype=x.dtype))
    s = 5.0 / sqrt6  # 波速
    xi = (x - s * t - x0) / (sqrt6)  # 行進波座標
    return (1.0 + torch.exp(-xi))**(-2)


def create_animation_with_exact_traveling_wave():
    
    # 參數設置
    x_min, x_max = -10.0, 10.0
    t_min, t_max = 0.0, 15.0
    nx, nt = 101, 101  # 網格點數
    
    # 計算旅行波速度和合適的初始位置
    # 波速 s = 5/√6 ≈ 2.04
    # 設置 x0 使得波在 t=0 時位於 x_min，在 t=t_max 時位於 x_max
    sqrt6 = np.sqrt(6.0)
    s = 5.0 / sqrt6
    x0 = x_min - s * t_min  # 在 t=0 時波位於 x_min
    
    # 創建空間和時間網格
    device = torch.device("cpu")  # 改用 CPU
    x = torch.linspace(x_min, x_max, nx, device=device)
    t_steps = torch.linspace(t_min, t_max, nt, device=device)
    
    print(f"生成動畫: x ∈ [{x_min}, {x_max}], t ∈ [{t_min}, {t_max}]")
    print(f"網格點: nx={nx}, nt={nt}")
    print(f"使用設備: {device}")
    print(f"旅行波參數: x0={x0}\n")
    
    # 初始化存儲
    rho_history = []
    
    # 對每個時間步計算精確解
    print("計算精確旅行波解...")
    for n, t_val in enumerate(t_steps):
        rho_t = exact_traveling_wave(x, t_val, x0=x0)
        rho_np = rho_t.cpu().detach().numpy().astype(np.float32)  # 確保是浮點數類型
        rho_history.append(rho_np)
        
        if (n + 1) % 20 == 0:
            print(f"  時間步 {n+1}/{nt}: min={rho_t.min():.6f}, max={rho_t.max():.6f}")
        elif n == 0:
            print(f"  時間步 {n+1}/{nt}: min={rho_t.min():.6f}, max={rho_t.max():.6f}")  # 列印第一步用於調試
    
    print("\n創建 GIF 動畫...")
    # 檢查存儲的數據
    print(f"stored rho_history frames: {len(rho_history)}")
    print(f"frame 0: min={rho_history[0].min()}, max={rho_history[0].max()}")
    print(f"frame 1: min={rho_history[1].min()}, max={rho_history[1].max()}")
    print(f"frame 19: min={rho_history[19].min()}, max={rho_history[19].max()}")
    # 創建 GIF 動畫
    x_np = x.cpu().detach().numpy()
    create_gif(x_np, rho_history, t_steps.cpu().numpy(), x_min, x_max)


def create_gif(x_np, rho_history, time_steps, x_min, x_max):
    """
    創建並保存 GIF 動畫
    """
    # 創建輸出目錄
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"animation_exact_traveling_wave_{timestamp}.gif")
    
    print(f"創建動畫... (共 {len(rho_history)} 幀)")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], 'b-', linewidth=2)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('rho(x,t)', fontsize=12)
    ax.set_title('Fisher-KPP Exact Traveling Wave: t=0~101, x=-10~10', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    def update(frame):
        line.set_data(x_np, rho_history[frame])
        time_text.set_text(f't = {time_steps[frame]:.2f}')
        return line, time_text
    
    anim = FuncAnimation(fig, update, frames=len(rho_history), interval=50, blit=True)
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer)
    plt.close()
    
    print(f"✅ GIF animation saved to: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("Fisher-KPP Exact Traveling Wave Animation Generator")
    print("="*60)
    print()
    create_animation_with_exact_traveling_wave()
    print()
    print("="*60)
    print("Done!")
    print("="*60)

