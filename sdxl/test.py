import torch
import matplotlib.pyplot as plt
import numpy as np

# パラメータを定義
beta_start = 0.0001
beta_end = 0.02
num_train_timesteps = 1000

# --- 1. betaスケジュールを計算 ---
# Linear
betas_linear = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

# Cosine
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    def alpha_bar(time_step):
        return np.cos((time_step / num_diffusion_timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

betas_cosine = betas_for_alpha_bar(num_train_timesteps)

# --- 2. alpha_t を計算 ---
alphas_linear = 1.0 - betas_linear
alphas_cosine = 1.0 - betas_cosine


# --- 3. グラフの描画 ---
plt.figure(figsize=(12, 7))
timesteps = torch.arange(num_train_timesteps)

# Linearスケジュールから得られたalpha_tをプロット
plt.plot(timesteps, alphas_linear, label='Alpha (αt) from Linear Schedule', color='blue')

# Cosineスケジュールから得られたalpha_tをプロット
plt.plot(timesteps, alphas_cosine, label='Alpha (αt) from Cosine Schedule', color='green')

# グラフのタイトルとラベルを設定
plt.title('Alpha (αt) の値の変化 (t=0 から 999)', fontsize=16)
plt.xlabel('Timestep (t)', fontsize=12)
plt.ylabel('Alpha (αt) の値', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(0.978, 1.002) # y軸の範囲を調整して見やすくする

# グラフを表示
plt.show()