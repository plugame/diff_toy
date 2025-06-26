import os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from safetensors.torch import load_file
#original
from utils.utils import encode_prompt, prepare_empty_latent,decode_latents,load_image,encode_image,convert_injectable_dict_from_weight,inject_lora_from_pretrained

device = torch.device("cuda")
dtype = torch.float16

# base_model
model_path = r"diffusers_model\v1-5-pruned-emaonly"

lora_path = "orign_output/yt_lora_type.safetensors"
lora_scale = 0.8

# output
output_dir = "generate2"
os.makedirs(output_dir,exist_ok=True)

# parameter
sampling_steps=100
guidance_scale = 6
width,height= 512,512

init_image_path = r"dataset/x_gamma/C064_C009_000.png"
denoising_strength = 0.4


prompts =  "line_to_color,transforming,anime,little_witch"
negative_prompt = ""


def step(x_t,t,noise_pred,x_gamma):
    t = t.cpu()
    # 1つ前のタイムステップ（t=300, prev_t = 200)
    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 1. 必要なαの値を取得
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_prev_t = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t = alpha_prod_t.to(x_t.device)
    alpha_prod_prev_t = alpha_prod_prev_t.to(x_t.device)
    
    num_dims = x_t.ndim
    for _ in range(num_dims - 1): 
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        alpha_prod_prev_t = alpha_prod_prev_t.unsqueeze(-1)
    
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_prev_t = 1 - alpha_prod_prev_t

    x_0_pred = (1/torch.sqrt(alpha_prod_t)) * (x_t - torch.sqrt(beta_prod_t) * (noise_pred + x_gamma))
    x_prev_t_pred = torch.sqrt(alpha_prod_prev_t) * x_0_pred+torch.sqrt(beta_prod_prev_t) * (noise_pred+x_gamma)
    x_prev_t_pred = x_prev_t_pred.to(dtype=x_t.dtype)

    return x_prev_t_pred


def step2(x_t,t,noise_pred,x_gamma):
    t = t.cpu()
    # 1つ前のタイムステップ（t=300, prev_t = 200)
    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 1. 必要なαの値を取得
    alpha_t = scheduler.alphas[t]
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_prev_t = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t = alpha_prod_t.to(x_t.device)
    alpha_prod_prev_t = alpha_prod_prev_t.to(x_t.device)

    num_dims = x_t.ndim
    for _ in range(num_dims - 1): 
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        alpha_prod_prev_t = alpha_prod_prev_t.unsqueeze(-1)

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_prev_t = 1 - alpha_prod_prev_t

    y_t_pred=(1/torch.sqrt(alpha_prod_t))*(x_t-torch.sqrt(beta_prod_t)*noise_pred)
    y_prev_t_pred = (torch.sqrt(alpha_prod_prev_t)/torch.sqrt(alpha_prod_t))*(y_t_pred-torch.sqrt(beta_prod_t)*x_gamma)+torch.sqrt(beta_prod_prev_t)*x_gamma
    x_prev_t_pred = torch.sqrt(alpha_prod_prev_t) * y_prev_t_pred + torch.sqrt(beta_prod_prev_t) * noise_pred
    x_prev_t_pred = x_prev_t_pred.to(dtype=x_t.dtype)
    return x_prev_t_pred

def add_noise2(x_0, t, noise, x_gamma):
    t = t.cpu()
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t = alpha_prod_t.to(x_0.device) # x_0と同じデバイスに移動させる

    num_dims = x_0.ndim
    for _ in range(num_dims - 1): # x_0の次元数-1回（バッチ次元を除く）繰り返す
        alpha_prod_t = alpha_prod_t.unsqueeze(-1) # 最後の次元に1を追加

    beta_prod_t = 1 - alpha_prod_t

    y_t = torch.sqrt(alpha_prod_t) * x_0 + torch.sqrt(beta_prod_t) * x_gamma
    x_t = torch.sqrt(alpha_prod_t) * y_t + torch.sqrt(beta_prod_t) * noise

    y_t = y_t.to(dtype=x_0.dtype)
    x_t = x_t.to(dtype=x_0.dtype)
    return x_t

def first_noise(t, noise, x_gamma):
    t = t.cpu()
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t = alpha_prod_t.to(x_gamma.device) # x_0と同じデバイスに移動させる

    num_dims = x_gamma.ndim
    for _ in range(num_dims - 1): # x_0の次元数-1回（バッチ次元を除く）繰り返す
        alpha_prod_t = alpha_prod_t.unsqueeze(-1) # 最後の次元に1を追加

    beta_prod_t = 1 - alpha_prod_t
    x_t = torch.sqrt(alpha_prod_t) * x_gamma + torch.sqrt(beta_prod_t) * noise
    x_t = x_t.to(dtype=x_gamma.dtype)
    return x_t
#----------------------------------------

tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer",local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder",torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)

# load lora
weights = load_file(lora_path)
# kohya_ss系から
unet_lora_dict, te_lora_dict, network_alphas = convert_injectable_dict_from_weight(weights)
# weight scale lora 
for key, weight in network_alphas.items():
    network_alphas[key] = weight * lora_scale
inject_lora_from_pretrained(unet,unet_lora_dict,network_alphas)
inject_lora_from_pretrained(text_encoder,te_lora_dict,network_alphas)


#scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path,subfolder="scheduler")
#scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path,subfolder="scheduler",use_karras_sigmas=True)
#scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path,subfolder="scheduler",algorithm_type="sde-dpmsolver++",use_karras_sigmas = True)
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
scheduler.set_timesteps(sampling_steps)


start_time = int(sampling_steps * (1-denoising_strength))
timesteps = scheduler.timesteps[start_time:]

x_gamma_image = load_image(init_image_path)
x_gamma = encode_image(x_gamma_image,vae)

noise = torch.randn_like(x_gamma)

first_flag = True

#------------------------------------------

positive_embeds, negative_embeds = encode_prompt(prompts, tokenizer, text_encoder,negative_prompt=negative_prompt)
prompt_embeds = torch.cat([negative_embeds, positive_embeds])

for i, t in enumerate(tqdm(timesteps.to(device))):
    with torch.no_grad():
        if first_flag:
            latents = first_noise(t,noise,x_gamma)
            latent = decode_latents(latents,vae)
            latent.save(f"{output_dir}/gen_.png")
            first_flag=False

        latent_input = torch.cat([latents]*2)
        
        #latent_input = scheduler.scale_model_input(latent_input, t)

        # 2回UNetに通す
        noise_pred = unet(
            latent_input,
            t, 
            encoder_hidden_states=prompt_embeds
            ).sample

        # CFGによる調整
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = step2(latents,t,noise_pred,x_gamma)




    latent = decode_latents(latents,vae)
    latent.save(f"{output_dir}/gen_{i}.png")