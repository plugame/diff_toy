import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DDIMScheduler, ControlNetModel,EulerAncestralDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel, Adafactor
from transformers.optimization import AdafactorSchedule
from tqdm import tqdm
from accelerate import Accelerator
#original
from utils.utils import get_optimal_torch_dtype
from utils.dataset_utils import ControlNetDataset

# model
model_path = "E:\\lab\\program\\train_controlnet\\diffusers_model\\v1-5-pruned-emaonly"

#dataset
dataset_path = "dataset"
repeat = 1

# output
output_dir = "controlnet_output"
output_name = "test_only_color"

# train prameter
batch_size = 1
lr = 0.0001
num_epochs = 5
save_every_n_epochs = 10
image_size = 512



# accelerator, dtype, device
accelerator = Accelerator()
device = accelerator.device
dtype, train_model_dtype = get_optimal_torch_dtype(accelerator.mixed_precision)

# load model
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer",local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder",torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

# create controlnet
controlnet = ControlNetModel.from_unet(unet).to(device=device,dtype=train_model_dtype)

# train settings
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

# prepare dataset
dataset = ControlNetDataset(dataset_path, vae, tokenizer, text_encoder, image_size,repeat=repeat)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

"""
optimizer = torch.optim.AdamW(controlnet.parameters(),lr=lr)
# (step_size)エポックごとに学習率を(gamma)倍にする
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
"""

optimizer = Adafactor(
    controlnet.parameters(),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,  # 自動調整
)

lr_scheduler = AdafactorSchedule(
    optimizer,
    initial_lr=lr
    )

# prepare (acceleratorに渡して wrap する)
controlnet, unet, text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(
    controlnet, unet, text_encoder, optimizer, lr_scheduler,  dataloader
)

def _save_weight(output_name):
    save_dir = os.path.join(output_dir,output_name)
    os.makedirs(save_dir,exist_ok=True)

    controlnet.save_pretrained(
        save_directory=save_dir,
        safe_serialization=True  # safetensors形式で保存
        )


# train
controlnet.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for image_latents, positive_embeds, cond_tensors in tqdm(dataloader):

        # ノイズとタイムステップをサンプリング
        noise = torch.randn_like(image_latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (image_latents.size(0),), device=device).long()

        # ノイズ付加
        noisy_image_latents = scheduler.add_noise(image_latents, noise, t)
        

        down_block_res_samples,mid_block_res_sample = controlnet(
            noisy_image_latents,
            t,
            encoder_hidden_states=positive_embeds,
            controlnet_cond=cond_tensors,
            conditioning_scale=1.0,
            return_dict=False,
        )

        noise_pred = unet(
            noisy_image_latents, 
            t, 
            encoder_hidden_states=positive_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            ).sample
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        loss = loss.mean()

        accelerator.backward(loss)
        lr_scheduler.step()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()

    if save_every_n_epochs is not None:
        if (epoch+1) % save_every_n_epochs == 0 and epoch+1 < num_epochs:
            _save_weight(f"{epoch}_{output_name}")

    accelerator.print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")


_save_weight(output_name)