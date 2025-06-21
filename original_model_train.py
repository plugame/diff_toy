import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, Adafactor
from transformers.optimization import AdafactorSchedule
from tqdm import tqdm
from safetensors.torch import save_file
from accelerate import Accelerator
from PIL import Image
#original
from utils.dataset_utils import LoRADataset
from utils.utils import inject_initial_lora, get_model_prefix, get_optimal_torch_dtype, show_model_param_status,encode_image,encode_prompt

# base_model
model_path = "E:\\lab\\program\\train_controlnet\\diffusers_model\\v1-5-pruned-emaonly"

# train_data
dataset_path = "dataset"
repeat = 10
rank = 128
alpha = 64

# output
output_dir = "lora_output"
output_name = "test_lora"

# other
batch_size = 5
lr = 1e-3
num_epochs = 40
save_every_n_epochs = 10
image_size = 512


# accelerator, dtype, device
accelerator = Accelerator()
device = accelerator.device
dtype, train_model_dtype = get_optimal_torch_dtype(accelerator.mixed_precision) # dtype = load and default, train_dtype = use train model

class OriginalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, vae, tokenizer, text_encoder,size=256, repeat=1):
        self.x_0_dir = os.path.join(root_dir, "x_0")
        self.x_gamma_dir = os.path.join(root_dir, "x_gamma")
        self.caption_dir = os.path.join(root_dir, "captions")
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.size = (size,size)
        self.repeat=repeat

        self.image_filenames = sorted([
            f for f in os.listdir(self.x_0_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.x_0_latents = {}
        self.x_gamma_latents = {}
        self.positive_embeds = {}
        # latent 変換
        for img_filename in tqdm(self.image_filenames):
            #x_0
            x_0_path = os.path.join(self.x_0_dir, img_filename)
            x_0 = Image.open(x_0_path).convert("RGB").resize(self.size, resample=Image.NEAREST)
            x_0_latent = encode_image(x_0,vae)
            self.x_0_latents[img_filename] = x_0_latent.squeeze(0)
            #x_gamma
            x_gamma_path = os.path.join(self.x_gamma_dir, img_filename)
            x_gamma = Image.open(x_gamma_path).convert("RGB").resize(self.size, resample=Image.NEAREST)
            x_gamma_latent = encode_image(x_gamma,vae)
            self.x_gamma_latents[img_filename] = x_gamma_latent.squeeze(0)
            #text_embeds
            txt_path = os.path.join(self.caption_dir, os.path.splitext(img_filename)[0] + ".txt")
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            positive_embeds, _ = encode_prompt(caption,self.tokenizer,self.text_encoder)
            self.positive_embeds[img_filename] = positive_embeds.squeeze(0)

    def __len__(self):
        return len(self.image_filenames) * self.repeat

    def __getitem__(self, idx):
        true_idx = idx % len(self.image_filenames)
        img_filename = self.image_filenames[true_idx]

        x_0_latent = self.x_0_latents[img_filename]
        x_gamma_latent = self.x_gamma_latents[img_filename]
        positive_embeds = self.positive_embeds[img_filename]
        
        return x_0_latent, x_gamma_latent, positive_embeds
    
# モデル読み込み
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer")
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder", torch_dtype=dtype).to(device)
vae = AutoencoderKL.from_pretrained(f"{model_path}/vae", torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(f"{model_path}/unet", torch_dtype=dtype).to(device)
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
#scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")


# unetとtext_encorderからrequires_grad=Trueのparamのdictを返す
def get_trainable_dict(unet,text_encoder):
    trainable_dict={}
    for model in [unet,text_encoder]:
        prefix = get_model_prefix(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_dict[prefix+"."+name] = param

    return trainable_dict

# 元パラメータの凍結
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)

# lora注入前にデータセットを作る
dataset = OriginalDataset(dataset_path, vae, tokenizer, text_encoder, image_size,repeat=repeat)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# loraを注入
unet_alphas= inject_initial_lora(unet,rank,alpha,dtype=train_model_dtype)
te_alphas = inject_initial_lora(text_encoder,rank,alpha,dtype=train_model_dtype)
network_alphas = {**unet_alphas,**te_alphas}

# trainするparamだけをoptimizerに渡す
trainable_params = list(get_trainable_dict(unet,text_encoder).values())

"""
optimizer = torch.optim.AdamW(trainable_params,lr=lr)
# (step_size)エポックごとに学習率を(gamma)倍にする
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
"""
optimizer = Adafactor(
    trainable_params,
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
unet, text_encoder, optimizer, lr_scheduler, dataloader = accelerator.prepare(
    unet, text_encoder, optimizer, lr_scheduler, dataloader
)

def _save_weight(output_name,unet,text_encoder,network_alphas):
    os.makedirs(output_dir,exist_ok=True)
    unet_to_save = accelerator.unwrap_model(unet)
    text_encoder_to_save = accelerator.unwrap_model(text_encoder)
    trained_dict = {
        name: param.detach()
        for name, param in get_trainable_dict(unet_to_save,text_encoder_to_save).items()
    }
    # alpha 値も追加
    lora_state_dict = {**trained_dict, **network_alphas}
    # LoRAの重み保存
    save_file(lora_state_dict, os.path.join(output_dir, output_name+".safetensors"))

total_steps = len(dataloader) * num_epochs

# 学習ループ
unet.train()
text_encoder.train()
with tqdm(total=total_steps) as pgbar:
    for epoch in range(num_epochs):
        epoch_loss = 0
        pgbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for i, (x_0_latents, x_gamma_latents, positive_embeds) in enumerate(dataloader):
            noise = torch.randn_like(x_0_latents)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (x_0_latents.shape[0],), device=device).long()
            
            #tはlongなのでfloatにキャスト
            lmd_t = t.float() / (scheduler.config.num_train_timesteps - 1)
            # バッチ対応
            lmd_t = lmd_t.reshape(-1, 1, 1, 1)

            latent =(1 - lmd_t) * x_0_latents + lmd_t * x_gamma_latents

            noisy_latents = scheduler.add_noise(latent, noise, t)

            with accelerator.autocast():
                noise_pred = unet(
                    noisy_latents,
                    t, 
                    encoder_hidden_states=positive_embeds
                    ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
            loss = loss.mean()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            pgbar.update(1)
            pgbar.set_postfix(loss=f"{loss.item():.4f}")

        if save_every_n_epochs is not None:
            if (epoch+1) % save_every_n_epochs == 0 and epoch+1 < num_epochs:
                _save_weight(f"{epoch}_{output_name}",unet,text_encoder,network_alphas)

        accelerator.print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")


_save_weight(output_name,unet,text_encoder,network_alphas)