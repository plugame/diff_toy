import os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
#original
from utils.utils import encode_prompt, prepare_empty_latent,decode_latents

device = torch.device("cuda")
dtype = torch.float16

# base_model
model_path = r"E:\lab\program\train_controlnet\diffusers_model\1450_v9"

# output
output_dir = "generate"
os.makedirs(output_dir,exist_ok=True)

# parameter
sampling_steps=20 
guidance_scale = 10
width = 512
height = 768



prompts =  "1girl, virtual youtuber, blue hair, yellow eyes, hair flower, heart ahoge, white thighhighs, blue skirt, beret, frilled thighhighs, half updo, streaked hair, white headwear, colored tips, cleavage cutout, hair between eyes, blue coat, white shirt, side braid, elf, sleeveless shirt, corset, center frills, fur-trimmed boots, wariza, off shoulder, fur-trimmed coat, bare shoulders, belt, white flower, high heel boots, very long hair, pleated skirt, blue bowtie , solo, smile, open mouth, looking at viewer, standing, cowboy shot, straight-on, black background, simple background, source anime, absurdres, masterpiece, best quality, very aesthetic "
negative_prompt = "low quality, worst quality, blurry, extra limbs"


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


#------------------------------------------

positive_embeds, negative_embeds = encode_prompt(prompts, tokenizer, text_encoder,negative_prompt=negative_prompt)
prompt_embeds = torch.cat([negative_embeds, positive_embeds])

latents = prepare_empty_latent(width,height,scheduler,device,dtype)

for i, t in enumerate(tqdm(scheduler.timesteps.to(device))):
    with torch.no_grad():
        latent_input = scheduler.scale_model_input(latents, t)
        #latent_input = latents

        # 2回UNetに通す
        noise_pred = unet(
            latent_input.repeat(2,1,1,1),
            t, 
            encoder_hidden_states=prompt_embeds
            ).sample

        # CFGによる調整
        # いわゆる残差？
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latent = decode_latents(latents,vae)
    latent.save(f"{output_dir}/gen_{i}.png")




