import os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler,EulerAncestralDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from safetensors.torch import load_file
#original
from utils.utils import encode_prompt, prepare_empty_latent, decode_latents,inject_lora_from_pretrained, convert_injectable_dict_from_weight
from utils.diffusers_utils import convert_injectable_dict_from_khoya_weight


device = torch.device("cuda:0")
dtype = torch.float16

# base_model
model_path = "E:\\lab\\program\\train_controlnet\\diffusers_model\\1450_v9"

# output
output_dir = "generate"
os.makedirs(output_dir,exist_ok=True)

# lora
lora_path = r"lora_output\lora_weights.safetensors"
lora_scale = 0.8

# parameter
guidance_scale = 8
sampling_steps=20
width = 512
height = 768

prompts =  " bocchi style, gotou hitori, 1girl, blue eyes, cube hair ornament, hair between eyes, hair ornament, hair over eyes, jacket, long hair, one side up, open mouth, pink hair, pink jacket, solo, track jacket, zipper, indoors, masterpiece"
negative_prompt = "low quality, worst quality, blurry, extra limbs"

"""
prompts = "base_color,only_color,anime,little_witch"
negative_prompt = None
"""

# load models
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer",local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder",torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
#scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
scheduler.set_timesteps(sampling_steps) 


# load lora
weights = load_file(lora_path)

# kohya_ss系から
# unet_lora_dict, te_lora_dict, network_alphas = convert_injectable_dict_from_khoya_weight(weights)
# originalから
unet_lora_dict, te_lora_dict, network_alphas = convert_injectable_dict_from_weight(weights)

# weight scale lora 
for key, weight in network_alphas.items():
    network_alphas[key] = weight * lora_scale

# inject lora
inject_lora_from_pretrained(unet,unet_lora_dict,network_alphas)
inject_lora_from_pretrained(text_encoder,te_lora_dict,network_alphas)




# generate
positive_embeds, negative_embeds = encode_prompt(prompts, tokenizer, text_encoder,negative_prompt=negative_prompt)
prompt_embeds = torch.cat([negative_embeds, positive_embeds])
latents = prepare_empty_latent(width,height,scheduler,device,dtype)

for i, t in enumerate(tqdm(scheduler.timesteps.to(device))):
    with torch.no_grad():
        latent_input = scheduler.scale_model_input(latents, t)
        #latent_input = latents
        noise_pred = unet(
            latent_input.repeat(2,1,1,1),
            t, 
            encoder_hidden_states=prompt_embeds
            ).sample

        # CFGによる調整
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # いわゆる残差？
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latent = decode_latents(latents,vae)
    latent.save(f"{output_dir}/gen_{i}.png")




