import os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
#original
from safetensors.torch import load_file
from utils.utils import encode_prompt, prepare_empty_latent,decode_latents,convert_injectable_dict_from_weight,inject_lora_from_pretrained
from utils.diffusers_utils import convert_injectable_dict_from_khoya_weight
device = torch.device("cuda")
dtype = torch.float16

# base_model
model_path = "/nfs2/sandbox/mori/train_controlnet/diffusers_model/v1-5-pruned-emaonly"

#lora_path = "/nfs2/sandbox/mori/diff_toy/lora_output/test_lora.safetensors"
lora_path = "/nfs2/sandbox/mori/ComfyUI/models/loras/only_color_v1.safetensors"
lora_scale = 0.8
original_lora = False


# output
output_dir = "generate"
os.makedirs(output_dir,exist_ok=True)

# parameter
sampling_steps=20 
guidance_scale = 8
width,height= 512,512


prompts = "base_color,only_color,anime,little_witch"
negative_prompt = None


#----------------------------------------

tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer",local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder",torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
#scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", beta_schedule='scaled_linear',beta_start=0.00085,beta_end=0.0120,eta=0.0, torch_dtype=dtype)
#scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path,subfolder="scheduler")
#scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path,subfolder="scheduler",use_karras_sigmas=True)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_path,
    subfolder="scheduler",
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas = True
)
scheduler.set_timesteps(sampling_steps)  

# load lora
weights = load_file(lora_path)

# kohya_ss系から
unet_lora_dict, te_lora_dict, network_alphas = convert_injectable_dict_from_weight(weights) if original_lora else convert_injectable_dict_from_khoya_weight(weights)


# weight scale lora 
for key, weight in network_alphas.items():
    network_alphas[key] = weight * lora_scale


inject_lora_from_pretrained(unet,unet_lora_dict,network_alphas)
inject_lora_from_pretrained(text_encoder,te_lora_dict,network_alphas)

#------------------------------------------

positive_embeds, negative_embeds = encode_prompt(prompts, tokenizer, text_encoder,negative_prompt=negative_prompt)
prompt_embeds = torch.cat([negative_embeds, positive_embeds])

latents = prepare_empty_latent(width,height,scheduler,device,dtype)

for i, t in enumerate(tqdm(scheduler.timesteps.to(device))):
    with torch.no_grad():
        latent_input = torch.cat([latents]*2)
        latent_input = scheduler.scale_model_input(latent_input, t)

        # 2回UNetに通す
        noise_pred = unet(
            latent_input,
            t, 
            encoder_hidden_states=prompt_embeds
            ).sample

        # CFGによる調整
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latent = decode_latents(latents,vae)
    latent.save(f"{output_dir}/gen_{i}.png")
