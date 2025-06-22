import torch
from diffusers import DDIMScheduler
import math
T=1000
scheduler = DDIMScheduler(num_train_timesteps=T)
scheduler.set_timesteps(41)

#xt -> xt-k
k = scheduler.config.num_train_timesteps // scheduler.num_inference_steps

def _lmd(t,k):
    return -k/(T-1-t)

def step(noise_pred,t,xt:torch.tensor,x_gamma:torch.tensor,k=1):
    alpha_cumprod_t = scheduler.alphas_cumprod[t]
    one_alpha_bar_sqrt = math.sqrt(1-alpha_cumprod_t)

    alpha_cumprod_t_k = scheduler.alphas_cumprod[t-k]
    alpha_bar_sqrt_k = math.sqrt(alpha_cumprod_t_k)
    one_alpha_bar_sqrt_k =  math.sqrt(1-alpha_cumprod_t_k)

    
    alpha_t = scheduler.alphas[t]
    alpha_t_sqrt = math.sqrt(alpha_t)

    return ((1-_lmd(t,k))/alpha_t_sqrt)*(xt-one_alpha_bar_sqrt*noise_pred)+(alpha_bar_sqrt_k*_lmd(t,k)*x_gamma)+one_alpha_bar_sqrt_k*noise_pred

def step2(
    noise_pred: torch.Tensor, 
    t: int, 
    xt: torch.Tensor, 
    x_gamma: torch.Tensor,
    eta: float = 0.0 # 0なら決定的(DDIM)、1なら確率的(DDPM)
):
    # 0. まず、時間に関する変数を準備
    total_timesteps = scheduler.config.num_train_timesteps
    
    # DDIMの論文に従い、一つ前のタイムステップを計算
    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 1. 必要なαの値を取得
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod
    
    beta_prod_t = 1 - alpha_prod_t

    # 2. あなたのモデルの核心部分：x_0を推定する
    # 2a. まず、クリーンな混合画像 y_t を推定
    y_t_pred = (xt - math.sqrt(beta_prod_t) * noise_pred) / math.sqrt(alpha_prod_t)
    
    # 2b. 次に、y_t_pred から x_0 (B) を逆算
    lmd_t = t / (total_timesteps - 1)
    # lmd_tの形状を整形
    lmd_t_reshaped = torch.tensor(lmd_t, device=xt.device).float().reshape(-1, 1, 1, 1)
    
    x_0_pred = (y_t_pred - lmd_t_reshaped * x_gamma) / (1 - lmd_t_reshaped)

    # 3. DDIMの公式を使って、前のステップの画像を計算
    # 3a. ノイズの方向を計算
    pred_epsilon = noise_pred
    
    # 3b. x_t-k を計算
    # variance_noise はeta>0のときのみ使われる確率的ノイズ
    variance = scheduler._get_variance(t, prev_t)
    std_dev_t = eta * math.sqrt(variance)
    variance_noise = torch.randn_like(noise_pred)
    
    prev_sample = (
        math.sqrt(alpha_prod_t_prev) * x_0_pred
        + math.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon
        + std_dev_t * variance_noise
    )
    
    return prev_sample 
