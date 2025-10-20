# RJ-net: Reaction-Jacobian Network

## 📋 概述

**RJ-net** 是一個基於物理信息神經網路 (Physics-Informed Neural Network, PINN) 的框架，用於求解**擴散-反應** (Diffusion-Reaction) 偏微分方程系統。該方法採用了創新的 **Reaction-Jacobian** 分解方法，能夠精確捕捉複雜的非線性反應動力學。

## 🎯 核心問題

本框架主要求解以下形式的 Fisher-KPP 擴散-反應方程：

$$\frac{\partial \rho}{\partial t} = \nu \cdot \frac{\partial^2 \rho}{\partial x^2} + r(x, \rho)$$

其中：
- $\rho(x,t)$ 是密度場
- $\nu$ 是擴散係數
- $r(x, \rho) = k^+ \rho - k^- \rho^2$ 是 Logistic 反應項

**精確旅行波解** (Traveling Wave Solution):
$$\rho(x, t) = \left[1 + \exp\left(-\frac{x - st - x_0}{\sqrt{6}}\right)\right]^{-2}$$

其中波速 $s = \frac{5}{\sqrt{6}}$。

## 🏗️ 核心方法論

### 1. 反應-雅可比分解

關鍵創新是將密度分解為：

$$\rho = \frac{\rho_0 + R}{J}$$

其中：
- $\rho_0$ 是初始密度
- $R(t)$ 是反應積分：$R^{n+1} = R^n + \Delta t \cdot r^n$
- $J(t)$ 是雅可比行列式：$J^{n+1} = J^n + \Delta t(\nabla \cdot u)J^n - u \cdot \nabla J$

### 2. 神經網路架構

框架使用兩個獨立的神經網路：

#### VelocityNet
```
輸入: [x, ρ]
隱層: 32 個隱元單元 (Tanh 激活)
輸出: 速度場 u(x, ρ)
```

#### ReactionNet
```
輸入: [x, ρ]
隱層: 32 個隱元單元 (Tanh 激活)
輸出: 反應率 r(x, ρ)
```

### 3. 時間離散化與前進

對於 $n = 0, 1, 2, \ldots$：

$$u^{n+1} = u_{NN}(\rho^n)$$
$$r_i^{n+1} = r_{NN}(\rho^n, x)$$
$$R^{n+1} = R^n + \Delta t \cdot r^n$$
$$J^{n+1} = J^n + \Delta t \left[(\nabla \cdot u)J^n - u \cdot \nabla J\right]$$
$$\rho^{n+1} = \frac{\rho_0 + R^{n+1}}{J^{n+1}}$$

## 📊 損失函數

訓練過程結合多個損失項：

$$\mathcal{L}_{\text{total}} = w_{\text{pde}} \mathcal{L}_{\text{pde}} + w_{\text{bc}} \mathcal{L}_{\text{bc}} + w_{\text{ic}} \mathcal{L}_{\text{ic}} + w_{\text{reaction}} \mathcal{L}_{\text{reaction}}$$

其中：
- **PDE 殘差損失**: $\mathcal{L}_{\text{pde}} = \|\rho_t - \nu \frac{\partial^2 \rho}{\partial x^2} - r\|^2$
- **邊界條件損失**: $\mathcal{L}_{\text{bc}} = \left|\frac{\partial \rho}{\partial x}\right|_{\text{邊界}}^2$ (零通量)
- **初始條件損失**: $\mathcal{L}_{\text{ic}} = \|\rho(t=0) - \rho_0\|^2$
- **反應一致性損失**: $\mathcal{L}_{\text{reaction}} = \|r_{NN} - r_{\text{exact}}\|^2$

## 🚀 快速開始

### 環境要求
- Python 3.8+
- PyTorch (建議使用 GPU)
- NumPy, Matplotlib
- PyYAML

### 安裝依賴
```bash
pip install torch numpy matplotlib pyyaml
```

### 運行訓練

在 `RJ_net` 目錄下運行：

```bash
python rj_net.py --config config.yaml --output ./results
```

**參數說明**：
- `--config`: 配置文件路徑 (默認: `config.yaml`)
- `--output`: 輸出目錄 (覆蓋配置文件中的設定)

### 配置參數

在 `config.yaml` 中修改以下參數：

```yaml
physics:
  x_min: -5.0          # 空間左邊界
  x_max: 5.0           # 空間右邊界
  t_min: 0.0           # 時間起點
  t_max: 2.0           # 時間終點
  nx: 201              # 空間網格點數
  nt: 101              # 時間步數
  nu: 1.0              # 擴散係數
  k_plus: 1.0          # 反應係數 k⁺
  k_minus: 1.0         # 反應係數 k⁻

training:
  learning_rate: 0.001 # Adam 優化器學習率
  epochs: 2000         # 訓練迭代次數
  hidden_size: 32      # 神經網路隱層大小
  
  # 損失權重
  weight_pde: 1.0      
  weight_bc: 0.1       
  weight_ic: 10.0      
  weight_reaction: 5.0 
```

## 📁 項目結構

```
RJ-net/
├── RJ_net/
│   ├── rj_net.py              # 主程式
│   ├── config.yaml            # 配置文件
│   ├── plotter.py             # 可視化模組
│   └── __pycache__/
├── nc_diffusion_colab/        # Colab 版本
│   ├── nc_diffusion.ipynb
│   ├── config.yaml
│   └── plotter.py
├── results/                   # 結果輸出目錄
│   └── run_YYYYMMDD_HHMMSS/   # 帶時間戳的運行結果
│       ├── loss.png           # 損失函數演化
│       ├── animation.gif      # 動畫
│       ├── analysis.png       # 詳細分析
│       └── config_used.yaml   # 使用的配置
├── note_RJ.md                 # 理論筆記
├── note_NC.md                 # Colab 筆記
└── README.md                  # 本文件
```

## 📈 輸出結果

運行完成後，會在 `results/run_YYYYMMDD_HHMMSS/` 目錄下生成以下文件：

### 1. **loss.png**
訓練過程中各項損失函數的演化曲線：
- 總損失 (Total Loss)
- PDE 殘差損失
- 邊界條件損失
- 反應一致性損失

### 2. **animation.gif**
完整的時間演化動畫，展示：
- 數值解 (紅線)
- 精確解 (綠線)
- 誤差 (藍線)

### 3. **analysis.png**
四面板詳細分析圖：
- **左上**: 速度場 $u(x,t)$ 的時空分佈
- **右上**: 反應率 $r(x,t)$ 的時空分佈
- **左下**: 神經網路學習的反應與精確反應的對比
- **右下**: 數值解與精確解的 $L^2$ 誤差演化

### 4. **config_used.yaml**
本次運行使用的完整配置，便於重現結果。

## 🔬 物理背景

### Fisher-KPP 方程
Fisher-KPP (Fisher-Kolmogorov-Petrovsky-Piskunov) 方程是描述種群動力學的經典模型，也用於建模火焰傳播等物理現象。

**特點**：
- 存在行進波解，以恆定速度 $s = 2\sqrt{\nu \cdot \lambda}$ 傳播
- 對於 $\nu = 1, \lambda = 1$，波速為 $s = 2$；對於 logistic 反應 $r = \rho(1-\rho)$，波速為 $s = \frac{5}{\sqrt{6}} \approx 2.04$
- 初始條件決定了波的寬度和形狀

### 為什麼使用 PINN？
傳統數值方法需要：
- 細密的空間時間網格
- 複雜的邊界條件處理
- 高計算成本

PINN 的優勢：
- ✅ 將物理約束直接編碼到損失函數
- ✅ 無需標記訓練數據
- ✅ 自動滿足邊界條件
- ✅ 可平滑推廣到新的域或參數

## 💡 核心創新點

### Reaction-Jacobian 分解
這個方法的關鍵創新是將複雜的質量守恆與反應耦合問題分解為：

1. **反應部分** ($R$): 通過神經網路學習反應速率
2. **雅可比部分** ($J$): 捕捉由速度場引起的體積變化

這種分解使得：
- 更容易訓練神經網路
- 提高了對反應項的學習精度
- 增強了物理可解釋性

## 🎓 使用場景

1. **參數發現**: 從觀測數據反推反應係數 $k^+, k^-$
2. **預測**: 快速預測新初始條件下的演化
3. **不確定性量化**: 通過集合方法估計模型不確定性
4. **多物理耦合**: 擴展到多種反應物和耦合方程組

## 📚 相關文獻

- Fisher, R. A. (1937). "The wave of advance of advantageous genes"
- Kolmogorov, A. N., et al. (1937). "Study of the diffusion equation with growth"
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks"

## 🤝 貢獻與反饋

歡迎提交問題和建議！

---

**最後更新**: 2025 年 10 月  
**作者**: RJ-net 開發團隊  
**許可證**: MIT
