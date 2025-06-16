import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import os

# --- 1. 設定 ---

# モデルとVAEのID
# SDXLのベースモデルと、高品質なVAEを指定します
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
vae_id = "madebyollin/sdxl-vae-fp16-fix"

# プロンプト（生成したい画像の内容）
prompt = "masterpiece, best quality, a majestic white dragon flying through a stormy sky, lightning, epic, cinematic lighting"
negative_prompt = "low quality, worst quality, blurry, ugly, watermark, text, signature"

# 出力設定
output_dir = "sdxl_images"
output_filename = "dragon_storm.png"
os.makedirs(output_dir, exist_ok=True)

# 生成パラメータ
num_inference_steps = 30  # 推論ステップ数 (25-40が一般的)
guidance_scale = 7.5      # プロンプトへの忠実度 (7-8が一般的)
width = 1024              # SDXLの標準的な解像度
height = 1024

# --- 2. モデルとパイプラインの準備 ---

# 使用するデバイスとデータ型を設定
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")

# 高品質なVAEを個別に読み込む
vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch_dtype)

# StableDiffusionXLPipelineを構築
# U-NetとText Encoder 1&2は自動で読み込まれます
# VAEは先ほど読み込んだ高品質なものに差し替えます
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch_dtype,
    variant="fp16",
    use_safetensors=True
)

# xformersを有効にして、メモリ効率と速度を向上させる（インストールされている場合）
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers enabled.")
except ImportError:
    print("xformers not installed. For faster generation, install it with 'pip install xformers'.")

# パイプラインをGPUに送る
pipe.to(device)

print("Pipeline setup complete.")

# --- 3. 画像の生成 ---

print("Generating image...")
# 再現性のためにシード値を固定
generator = torch.Generator(device=device).manual_seed(42)

# パイプラインを実行して画像を生成
# with torch.inference_mode()で囲むと、不要な勾配計算を無効化し、さらに高速になります
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

# --- 4. 画像の保存 ---

output_path = os.path.join(output_dir, output_filename)
image.save(output_path)

print(f"Image saved to: {output_path}")