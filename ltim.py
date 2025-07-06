import torch
import numpy as np

from diffusers import DDIMScheduler,AutoencoderKL
from utils.utils import load_image,encode_image,decode_latents

device = "cuda"
dtype = torch.float16

model_path = "E:\\lab\\program\\train_controlnet\\diffusers_model\\v1-5-pruned-emaonly"

vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)


class LTIMScheduler:
    def __init__(self, total_timesteps=100, timesteps=100):
        """
        スケジューラの初期化。
        total_timestepsとtime_stepsを引数で受け取るように変更。
        """
        self.total_timesteps = total_timesteps
        self.timesteps = np.linspace(self.total_timesteps-1, 0, timesteps, dtype=int).tolist()#[99,98,93,...,5,0]

        # --- 効率化: ループ計算をNumpyのベクトル化演算に置き換え ---
        timesteps_np = np.arange(0, self.total_timesteps)#0～99
        f_values = self.f(timesteps_np)
        total_sum_f = np.sum(f_values)
        alpha_t_np = f_values / total_sum_f
        alpha_cum_t_np = np.cumsum(alpha_t_np)

        self.alpha_t = torch.tensor(alpha_t_np, dtype=torch.float32)
        self.alpha_cum_t = torch.tensor(alpha_cum_t_np, dtype=torch.float32)
        self.final_sigma_alpha_value = self.alpha_cum_t[1]

    def set_timesteps(self,timesteps):
        self.timesteps = np.linspace(self.total_timesteps-1, 0, timesteps, dtype=int).tolist()

    def f(self, t):
        return (t-self.total_timesteps)

    def add_noise(self, x_0, d, t):
        alpha_cum = self.alpha_cum_t[t-1].to(x_0.device)
        while len(alpha_cum.shape) < len(x_0.shape):
            alpha_cum = alpha_cum.unsqueeze(-1)

        x_t = x_0 + alpha_cum * d
        x_t = x_t.to(dtype=x_0.dtype)

        return x_t
    
    def step(self, x_t, d_pred, t):
        t = int(t.cpu())
        try:
            current_index = self.timesteps.index(t)
        except ValueError:
            raise ValueError(f"Timestep {t} is not in the scheduler's defined timesteps.")

        alpha_t_prev = self.alpha_t[t].to(x_t.device)
        
        while len(alpha_t_prev.shape) < len(x_t.shape):
            alpha_t_prev = alpha_t_prev.unsqueeze(-1)

        x_t_prev_pred = x_t - alpha_t_prev * d_pred
        x_t_prev_pred = x_t_prev_pred.to(dtype=dtype)
        return x_t_prev_pred



class LTIMScheduler_OLD:
    def __init__(self, total_timesteps=100, timesteps=10):
        """
        スケジューラの初期化。
        total_timestepsとtime_stepsを引数で受け取るように変更。
        """
        self.total_timesteps = total_timesteps
        self.timesteps = np.linspace(self.total_timesteps-1, 0, timesteps + 1, dtype=int).tolist()#[100,98,93,...,5,1]

        # --- 効率化: ループ計算をNumpyのベクトル化演算に置き換え ---
        timesteps_np = np.arange(1, self.total_timesteps)#1～99
        f_values = self.f(timesteps_np)
        total_sum_f = np.sum(f_values)
        alpha_t_np = f_values / total_sum_f
        alpha_cum_t_np = np.cumsum(alpha_t_np)

        self.alpha_cum_t = torch.tensor(np.concatenate(([0.0], alpha_cum_t_np)), dtype=torch.float32)
        self.final_sigma_alpha_value = self.alpha_cum_t[1]

    def set_timesteps(self,timesteps):
        self.timesteps = np.linspace(self.total_timesteps, 1, timesteps + 1, dtype=int).tolist()#[100,98,93,...,5,1]

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
        t = int(t.cpu())
        try:
            current_index = self.timesteps.index(t)
        except ValueError:
            raise ValueError(f"Timestep {t} is not in the scheduler's defined timesteps.")

        t_prev = self.timesteps[current_index + 1] if current_index < len(self.timesteps) - 1 else 0

        sigma_alpha_values = self.alpha_cum_t[t] - self.alpha_cum_t[t_prev] if t_prev > 0 else self.alpha_cum_t[t]

        sigma_alpha = sigma_alpha_values.to(x_t.device)
        
        while len(sigma_alpha.shape) < len(x_t.shape):
            sigma_alpha = sigma_alpha.unsqueeze(-1)

        x_t_prev_pred = x_t - sigma_alpha * d_pred
        x_t_prev_pred = x_t_prev_pred.to(dtype=dtype)
        return x_t_prev_pred
    

class LTIMScheduler2:
    def __init__(self, total_timesteps=10,timesteps=10):
        self.timesteps = np.linspace(1, total_timesteps, timesteps, dtype=int).tolist()

    def set_timesteps(self,timesteps):
        pass

    def f(self, n):
        return (0.2)**n
    
    def g(self,n):
        return (1/100)*(n-10)**2

    def add_noise(self, x_0, d, t,p_0):
        threshold = 0.2
        magnitudes = torch.abs(d)
        mask = magnitudes > threshold
        #noise = torch.randn_like(d)
        x_t = x_0+(1-self.f(t))*d+self.g(t)*p_0*mask.to(d.dtype)

        return x_t
    
    def step(self, x_t, d_pred, t):

        x_t_prev_pred = x_t + d_pred

        return x_t_prev_pred



if __name__ == "__main__":
    scheduler = LTIMScheduler()
    p_T_path = r"E:\lab\siage\separate_cut\C360_A_line\0_C360_A003.png"
    p_0_path = r"E:\lab\siage\separate_cut\C360_A_image\0_C360_A003.png"
    x_T_path = r"E:\lab\siage\separate_cut\C360_A_line\0_C360_A004.png"
    x_0_path = r"E:\lab\siage\separate_cut\C360_A_image\0_C360_A004.png"
    p_T = encode_image(load_image(p_T_path),vae)
    p_0 = encode_image(load_image(p_0_path),vae)
    x_T = encode_image(load_image(x_T_path),vae)
    x_0 = encode_image(load_image(x_0_path),vae)
    d = x_T - x_0
    d_p = p_T - p_0


    for t in scheduler.timesteps:
        t = t + 1 #1～100
        print(t)
        x_t = scheduler.add_noise(x_0,d,t)
        img = decode_latents(x_t,vae)
        img.save(f"x_{t}.png")

        """
        x_t_prev = scheduler.step(x_t,d,t)
        img = decode_latents(x_t,vae)
        img.save(f"x_{t}_prev.png")
        """










