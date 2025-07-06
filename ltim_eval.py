import os
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import Adafactor
from tqdm import tqdm
from safetensors.torch import load_file
from PIL import Image
#original

from utils.utils import get_optimal_torch_dtype, show_model_param_status,encode_image,load_image,decode_latents
from ltim import LTIMScheduler
from unet.openaimodel import UNetModel_LTIM
import torch.nn.functional as F

dtype = torch.float16
device = "cuda:0"

# base_model
model_path = r"E:\lab\program\train_controlnet\diffusers_model\v1-5-pruned-emaonly"
ltim_model_path =  r"E:\lab\diff_toy\models\7_+ltim_model.safetensors"
pretrained_model_state_dict = load_file(ltim_model_path)

# output
output_dir = "ltim_gen"
os.makedirs(output_dir,exist_ok=True)

# other
image_size = 512

# モデル読み込み
vae = AutoencoderKL.from_pretrained(f"{model_path}/vae", torch_dtype=dtype).to(device)
unet = UNetModel_LTIM.from_pretrained(pretrained_model_state_dict,dtype=dtype,device=device)

p_T_path = r"E:\lab\siage\separate_cut\C360_A_line\0_C360_A003.png"
p_0_path = r"E:\lab\siage\separate_cut\C360_A_image\0_C360_A003.png"
x_T_path = r"E:\lab\siage\separate_cut\C360_A_line\0_C360_A004.png"
x_0_path = r"E:\lab\siage\separate_cut\C360_A_image\0_C360_A004.png"
p_T = encode_image(load_image(p_T_path),vae)
p_0 = encode_image(load_image(p_0_path),vae)
x_T = encode_image(load_image(x_T_path),vae)
x_0 = encode_image(load_image(x_0_path),vae)

x_t = x_T
d = x_T - x_0
d_p = p_T - p_0

unet.eval()
vae.eval()

def vec_test(d,d_pred):
    d_norm = torch.linalg.norm(d)
    d_pred_norm = torch.linalg.norm(d_pred)
    d_p_norm = torch.linalg.norm(d_p)
    print(f"実際のノルム{d_norm},予測したノルム{d_pred_norm},d_pノルム{d_p_norm}")
    d_flat = d.flatten()
    d_pred_flat = d_pred.flatten()
    cosine_sim = F.cosine_similarity(d_flat, d_pred_flat, dim=0)

    print(f"コサイン類似度 (単位ベクトルの一致度): {cosine_sim.item():.4f}")
    return cosine_sim

d_pred_list = []
scheduler = LTIMScheduler()
for i, t in enumerate(tqdm(scheduler.timesteps)):
    t = torch.tensor(t).to(dtype=dtype,device=device)
    with torch.no_grad():

        # add prompt vector
        input_latents = torch.cat([x_t,d_p],dim=1)
        
        d_pred = unet(
            input_latents,
            t, 
            )
        d_pred_list.append(d_pred)
        #vec_test(d,d_pred)
        #out = x_T - d_pred
        x_t = scheduler.step(x_t, d_pred, t)


    decode_x_t = decode_latents(x_t,vae)
    decode_x_t.save(f"{output_dir}/gen_{i}.png")