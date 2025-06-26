import os
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule
from tqdm import tqdm
from safetensors.torch import save_file
from accelerate import Accelerator
#original
from utils.dataset_utils import LoRADataset
from utils.utils import get_optimal_torch_dtype, show_model_param_status,encode_image,load_image
from ltim import LTIMScheduler
from unet.openaimodel import UNetModel_LTIM

# base_model
model_path = "E:\\lab\\program\\train_controlnet\\diffusers_model\\v1-5-pruned-emaonly"

# train_data
dataset_path = "dataset"
repeat = 20

# output
output_dir = "ltim_output"
output_name = "ltim_model"

# other
batch_size = 5
lr = 1e-3
num_epochs = 100
save_every_n_epochs = 10
image_size = 512

# accelerator, dtype, device
accelerator = Accelerator()
device = accelerator.device
dtype, train_model_dtype = get_optimal_torch_dtype(accelerator.mixed_precision) # dtype = load and default, train_dtype = use train model

# モデル読み込み
vae = AutoencoderKL.from_pretrained(f"{model_path}/vae", torch_dtype=dtype).to(device)
#unet = UNet2DConditionModel.from_pretrained(f"{model_path}/unet", torch_dtype=dtype).to(device)

unet = UNetModel_LTIM().to(dtype=dtype,device=device)

scheduler = LTIMScheduler()

class LTIMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, vae, size=512, repeat=1):
        self.size = (size, size)
        self.vae = vae
        self.repeat = repeat
        self.root_dir = os.path.abspath(root_dir)
        sub_dirs = [os.path.join(self.root_dir, d[:-6]) for d in os.listdir(self.root_dir) if d.endswith("_image")]

        self.path_latent_dict = {}
        for path in tqdm(self.get_all_image_paths(self.root_dir), desc="Encoding images"):
            self.path_latent_dict[path] = encode_image(load_image(path, self.size), vae)

        self.dataset = []
        for sub_dir in tqdm(sub_dirs, desc="Building dataset"):
            image_list = [[], []]
            for fname in os.listdir(sub_dir + "_image"):
                full_path = os.path.join(sub_dir + "_image", fname)
                if fname.startswith("0"):
                    image_list[0].append(full_path)
                else:
                    image_list[1].append(full_path)

            for _list in image_list:
                use_list = sorted(_list)
                for i in range(len(use_list) - 1):
                    self.dataset.append(self.path_list_to_dataset_shape(i, use_list, sub_dir))

    def path_list_to_dataset_shape(i,path_list,sub_dir):
        p_0 = os.path.join(sub_dir+"_image",path_list[i])
        p_T = os.path.join(sub_dir+"_line",path_list[i])
        x_0 = os.path.join(sub_dir+"_image",path_list[i+1])
        x_T = os.path.join(sub_dir+"_line",path_list[i+1])
        cut_diff = int(x_0[-7:-4])-int(p_0[-7:-4])

        return (p_0,p_T,x_0,x_T),cut_diff


    def get_all_image_paths(root_dir, extensions={".png", ".jpg", ".jpeg", ".bmp", ".webp"}):
        image_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in extensions:
                    image_paths.append(os.path.join(dirpath, filename))
        return image_paths

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        true_idx = idx % len(self.dataset)
        (p_0, p_T, x_0, x_T), cut_diff = self.dataset[true_idx]
        return (
            self.path_latent_dict[p_0],
            self.path_latent_dict[p_T],
            self.path_latent_dict[x_0],
            self.path_latent_dict[x_T]
        ), cut_diff

# 元パラメータの凍結
vae.requires_grad_(False)
unet.requires_grad_(True)

dataset = LTIMDataset(dataset_path,vae,repeat=repeat)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = Adafactor(
    unet.parameters(),
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
unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(
    unet, optimizer, lr_scheduler, dataloader
)

def _save_weight(output_name,unet):
    os.makedirs(output_dir,exist_ok=True)
    save_file(unet.state_dict(), output_name+".safetensors")

total_steps = len(dataloader) * num_epochs

# 学習ループ
unet.train()
with tqdm(total=total_steps) as pgbar:
    for epoch in range(num_epochs):
        epoch_loss = 0
        pgbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for i, (x_0, x_T, p_0, p_T), _ in enumerate(dataloader):
            
            t = torch.randint(0, scheduler.total_timesteps, (x_0.shape[0],), device=device).long()
            #true vector
            d = x_T - x_0
            transforming_latents = scheduler.add_noise(x_0, d, t)

            #prompt vector
            d_p = p_T - p_0

            # add prompt vector
            input_latents = torch.cat([transforming_latents,d_p],dim=1)# (1,8,64,64)
            
            with accelerator.autocast():
                d_pred = unet(
                    input_latents,
                    t
                    )

            loss = torch.nn.functional.mse_loss(d_pred.float(), d.float(), reduction="none")
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
                _save_weight(f"{epoch+1}_+{output_name}",unet)

        accelerator.print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")

_save_weight(output_name,unet)
