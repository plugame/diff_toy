import os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from tqdm import tqdm
from typing import List, Optional, Tuple,Union

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from utils.utils import prepare_empty_latent,decode_latents


device = torch.device("cuda")
dtype = torch.bfloat16

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
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
    """指定されたTokenizerとText Encoderでプロンプトをエンコードし、
    hidden_statesとpooled_outputの両方を返す。
    """
    # トークン化
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    # エンコード実行
    # output_hidden_statesはSDv1.5のclip_skip用。SDXLではpooled_outputを重視する。
    # text_encoderの返り値は (last_hidden_state, pooled_output, hidden_states) のタプル。
    outputs = text_encoder(text_input_ids, output_hidden_states=True)
    
    # U-Netの入力に使う、トークンごとの詳細な埋め込み
    # sdxlではclipskip 2
    prompt_embeds = outputs.hidden_states[-2]
    
    # 全体的なスタイルを決定する、プロンプト全体の要約ベクトル

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
text_encoder = CLIPTextModel.from_pretrained(model_id,subfolder="text_encoder",torch_dtype=dtype).to(device)
text_encoder_2 =CLIPTextModelWithProjection.from_pretrained(model_id,subfolder="text_encoder_2",torch_dtype=dtype).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id,subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id,subfolder="tokenizer_2")
unet = UNet2DConditionModel.from_pretrained(model_id,subfolder="unet",torch_dtype=dtype).to(device)
#scheduler = DDPMScheduler.from_pretrained(model_id,subfolder="scheduler",torch_dtype=dtype)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas = True
)
scheduler.set_timesteps(sampling_steps) 

# 最適化1: xFormersを有効化
try:
    unet.enable_xformers_memory_efficient_attention()
except Exception:
    print("xformers not instaled")




print("model loaded")


(
    prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds
) = encode_prompt_sdxl_simple(tokenizer,text_encoder,tokenizer_2,text_encoder_2,prompt=prompts,negative_prompt=negative_prompt,do_classifier_free_guidance=True,device=device)

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

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    指定された形状、デバイス、データ型で正規分布に従う乱数テンソルを生成する。
    バッチ処理のためにジェネレータのリストも受け付ける。
    """
    # ジェネレータがリストでない場合、単一のジェネレータをリストにラップする
    if generator is None or isinstance(generator, torch.Generator):
        generator = [generator] * shape[0]
    
    # バッチの各要素に対して個別のジェネレータで乱数を生成し、結合する
    latents = [
        torch.randn(shape[1:], generator=g, device=device, dtype=dtype)
        for g in generator
    ]
    latents = torch.stack(latents, dim=0)
    
    return latents

vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) 
print(vae_scale_factor)
# 2. `randn_tensor` を使うようにリファクタリングされた `prepare_latents` 関数
def prepare_latents(
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    latents: Optional[torch.Tensor] = None,
):
    """
    推論の開始点となる潜像（Latent）を準備する。
    `latents`が与えられない場合は、`randn_tensor`を使ってノイズを生成する。
    """
    # ステップ1: 生成するLatentの形状を計算する
    # VAEのスケールファクタで画像の解像度を縮小する
    shape = (
        batch_size,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )

    # ステップ2: `latents`が外部から与えられていないかチェック
    if latents is None:
        # `latents`がNoneの場合、新しく実装した`randn_tensor`でノイズを生成
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        # `latents`が与えられている場合は、デバイスを合わせてそのまま使用
        if latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
        latents = latents.to(device)

    # ステップ3: スケジューラの初期ノイズスケールに合わせてLatentをスケーリング
    # これにより、使用するスケジューラに関わらず、一貫したノイズレベルから開始できる
    latents = latents * scheduler.init_noise_sigma
    
    return latents

num_channels_latents = unet.config.in_channels
latents = prepare_latents(
    1,
    num_channels_latents,
    height,
    width,
    prompt_embeds.dtype,
    device,
    None,
    None,
    )
#latents = prepare_empty_latent(width,height,scheduler,device,dtype)

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










