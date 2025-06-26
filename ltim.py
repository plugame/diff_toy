import torch
import numpy as np

from diffusers import DDIMScheduler,AutoencoderKL
from utils.utils import load_image,encode_image,decode_latents

device = "cuda"
dtype = torch.float16

model_path = "E:\\lab\\program\\train_controlnet\\diffusers_model\\v1-5-pruned-emaonly"

vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)


class LTIMScheduler:
    def __init__(self, total_timesteps=100, timesteps=5):
        """
        スケジューラの初期化。
        total_timestepsとtime_stepsを引数で受け取るように変更。
        """
        self.total_timesteps = total_timesteps
        self.time_steps = timesteps

        # --- 効率化: ループ計算をNumpyのベクトル化演算に置き換え ---
        timesteps_np = np.arange(1, self.total_timesteps + 1)
        f_values = self.f(timesteps_np)
        total_sum_f = np.sum(f_values)
        alpha_t_np = f_values / total_sum_f
        alpha_cum_t_np = np.cumsum(alpha_t_np)

        self.alpha_cum_t = torch.tensor(np.concatenate(([0.0], alpha_cum_t_np)), dtype=torch.float32)
        self.final_sigma_alpha_value = self.alpha_cum_t[1]

    def set_timesteps(self,timesteps):
        self.time_steps = timesteps

    def f(self, t):
        return t**2

    def add_noise(self, x_0, d, t):
        sigma_alpha_values = self.alpha_cum_t[t]
        sigma_alpha = sigma_alpha_values.to(x_0.device)
        while len(sigma_alpha.shape) < len(x_0.shape):
            sigma_alpha = sigma_alpha.unsqueeze(-1)

        x_t = x_0 + sigma_alpha * d
        x_t = x_t.to(dtype=x_0.dtype)

        return x_t
    
    def step(self, x_t, d_pred, t):
        k = self.total_timesteps // self.time_steps

        t_prev = t - k
        sigma_alpha_values = self.alpha_cum_t[t] - self.alpha_cum_t[t_prev] if t_prev > 0 else self.alpha_cum_t[t]

        sigma_alpha = sigma_alpha_values.to(x_t.device)
        
        while len(sigma_alpha.shape) < len(x_t.shape):
            sigma_alpha = sigma_alpha.unsqueeze(-1)

        x_t_prev_pred = x_t - sigma_alpha * d_pred
        x_t_prev_pred = x_t_prev_pred.to(dtype=dtype)
        return x_t_prev_pred



if __name__ == "__main__":
    scheduler = LTIMScheduler()
    x_0 = r"E:\lab\program\border_pixel\B\B004.png"
    x_0 = load_image(x_0,(512,512))
    x_0 = encode_image(x_0,vae)
    x_T = r"E:\lab\program\border_pixel\output2\3.png"
    x_T = load_image(x_T,(512,512))
    x_T = encode_image(x_T,vae)
    
    for t in range(0,101,20):
        d = x_T - x_0
        x_t = scheduler.add_noise(x_0,d,t)
        x_t_prev = scheduler.step(x_t,d,t)
        image = decode_latents(x_t,vae)
        image.save(f"x_{t}.png")
        image = decode_latents(x_t_prev,vae)
        image.save(f"x_{t}_prev.png")










