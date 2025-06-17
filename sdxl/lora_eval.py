import os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from tqdm import tqdm
from typing import List, Optional, Tuple,Union

from safetensors.torch import load_file
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from utils.utils import prepare_empty_latent,decode_latents,inject_lora_from_pretrained
from utils.diffusers_utils import convert_injectable_dict_from_khoya_weight_sdxl

lora_path = r"E:\sd\Lora\Bocchi-20.safetensors"
weights = load_file(lora_path)

unet_lora_dict, te_lora_dict, network_alphas = convert_injectable_dict_from_khoya_weight_sdxl(weights)


device = torch.device("cuda")
dtype = torch.bfloat16

model_path = r"E:\sd\diffusers_model\stable-diffusion-xl-base-1.0"
# SDXLのVAEはfp16での不具合を修正したものが推奨される
vae_id = "madebyollin/sdxl-vae-fp16-fix"

output_dir="generate"
os.makedirs(output_dir,exist_ok=True)
width,height = 1024,1024
sampling_steps=20

prompts =  "An astronaut riding a green horse"
negative_prompt = "low quality, worst quality, blurry, extra limbs"

guidance_scale = 8

def _encode_one_prompt(
    prompt: List[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # tokenizerを通す
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    # text_encoderを通す
    outputs = text_encoder(text_input_ids, output_hidden_states=True)
    
    prompt_embeds = outputs.hidden_states[-2]
    pooled_prompt_embeds = outputs[0]

    return prompt_embeds, pooled_prompt_embeds


# 2. リファクタリングされたメインのencode_prompt関数
def encode_prompt_sdxl_simple(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    tokenizer_2: CLIPTokenizer,
    text_encoder_2: CLIPTextModelWithProjection,
    prompt: str,
    device: Optional[torch.device] = None,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = "",
):
    """
    SDXL向けにリファクタリングされた、シンプルで分かりやすいプロンプトエンコード関数。
    """
    # プロンプトをリスト形式に統一
    prompt_list = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt_list)
    
    # --- ポジティブプロンプトのエンコード ---
    
    # 1番目のエンコーダで処理
    prompt_embeds_1, _ = _encode_one_prompt(prompt_list, tokenizer, text_encoder, device)
    
    # 2番目のエンコーダで処理
    prompt_embeds_2, pooled_prompt_embeds = _encode_one_prompt(prompt_list, tokenizer_2, text_encoder_2, device)

    # 2つのエンコーダからの出力を連結
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    # --- ネガティブプロンプトのエンコード (CFGが有効な場合) ---
    if do_classifier_free_guidance:
        uncond_tokens = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
        
        # 1番目のエンコーダで処理
        negative_prompt_embeds_1, _ = _encode_one_prompt(uncond_tokens, tokenizer, text_encoder, device)
        
        # 2番目のエンコーダで処理
        negative_prompt_embeds_2, negative_pooled_prompt_embeds = _encode_one_prompt(uncond_tokens, tokenizer_2, text_encoder_2, device)
        
        # 2つのエンコーダからの出力を連結
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1)

    else: # CFGを使わない場合は、ゼロベクトルで埋める
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)


    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def _get_add_time_ids(
    original_size: Tuple[int, int],
    crops_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    dtype: torch.dtype,
    text_encoder_projection_dim: int = None,
) -> torch.Tensor:
    add_time_ids_list = list(original_size + crops_coords_top_left + target_size)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    passed_add_embed_dim = (
        unet.config.addition_time_embed_dim * len(add_time_ids_list)
        + text_encoder_projection_dim
    )
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"U-Netが期待する追加情報の次元数 ({expected_add_embed_dim}) と、"
            f"実際に生成された次元数 ({passed_add_embed_dim}) が一致しません。"
            "モデルのコンフィグ（unet.config.addition_time_embed_dimや"
            "text_encoder_2.config.projection_dim）が間違っている可能性があります。"
        )
    add_time_ids_tensor = torch.tensor([add_time_ids_list], dtype=dtype)
    return add_time_ids_tensor

vae = AutoencoderKL.from_pretrained(vae_id,torch_dtype=dtype).to(device)
text_encoder = CLIPTextModel.from_pretrained(model_path,subfolder="text_encoder",torch_dtype=dtype).to(device)
text_encoder_2 =CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder_2",torch_dtype=dtype).to(device)

tokenizer = CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer_2")
unet = UNet2DConditionModel.from_pretrained(model_path,subfolder="unet",torch_dtype=dtype).to(device)

inject_lora_from_pretrained(unet,unet_lora_dict,network_alphas)

for i in unet.state_dict().keys():
    print(i)

exit()
inject_lora_from_pretrained(text_encoder,te_lora_dict["text_encoder"],network_alphas)
inject_lora_from_pretrained(text_encoder_2,te_lora_dict["text_encoder_2"],network_alphas)






#scheduler = DDPMScheduler.from_pretrained(model_id,subfolder="scheduler",torch_dtype=dtype)
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler", algorithm_type="sde-dpmsolver++", use_karras_sigmas = True)
scheduler.set_timesteps(sampling_steps) 

# 最適化1: xFormersを有効化
try:
    unet.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not instaled")

print("model loaded")

(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds
) = encode_prompt_sdxl_simple(tokenizer,
                              text_encoder,
                              tokenizer_2,
                              text_encoder_2,
                              prompt=prompts,
                              negative_prompt=negative_prompt,do_classifier_free_guidance=True,
                              device=device
                              )

print("encode prompt")

add_text_embeds = pooled_prompt_embeds
text_encoder_projection_dim = text_encoder_2.config.projection_dim

original_size = target_size = (height,width)
crops_coords_top_left = (0,0)

add_time_ids = _get_add_time_ids(
    original_size,
    crops_coords_top_left,
    target_size,
    dtype=prompt_embeds.dtype,
    text_encoder_projection_dim=text_encoder_projection_dim,
)

negative_add_time_ids = add_time_ids


prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)


prompt_embeds = prompt_embeds.to(device)
add_text_embeds = add_text_embeds.to(device)
add_time_ids = add_time_ids.to(device).repeat(1, 1)


latents = prepare_empty_latent(width,height,scheduler,device,dtype)

print("start generate")

unet.eval()
vae.eval()
with torch.inference_mode():
    for i, t in enumerate(tqdm(scheduler.timesteps.to(device))):
        with torch.no_grad():
            latent_input = torch.cat([latents]*2)
            print("0")
            latent_input = scheduler.scale_model_input(latent_input, t)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            print("a")
            noise_pred = unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
                ).sample
            print("i")
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            print("u")
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            print("e")

image = decode_latents(latents,vae)
image.save(f"{output_dir}/gen_{i}.png")










