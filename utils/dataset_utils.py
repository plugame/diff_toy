import torch
from tqdm import tqdm
from utils.utils import encode_prompt,encode_image,image_to_tensor
from PIL import Image
import os

# データセットの準備（例：画像とキャプション）
class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, vae, tokenizer, text_encoder, size=512, repeat=1):
        self.image_dir = os.path.join(root_dir, "images")
        self.caption_dir = os.path.join(root_dir, "captions")
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.size = (size,size)
        self.repeat=repeat

        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.latents = {}
        self.positive_embeds = {}
        # latent 変換
        for img_filename in tqdm(self.image_filenames):
            #image
            img_path = os.path.join(self.image_dir, img_filename)
            image = Image.open(img_path).convert("RGB").resize(self.size, resample=Image.NEAREST)
            latent = encode_image(image,vae)
            self.latents[img_filename] = latent.squeeze(0)
            #text_embeds
            txt_path = os.path.join(self.caption_dir, os.path.splitext(img_filename)[0] + ".txt")
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            # enbedsを取得

            positive_embeds, _ = encode_prompt(caption,self.tokenizer,self.text_encoder)
            self.positive_embeds[img_filename] = positive_embeds.squeeze(0)

    def __len__(self):
        return len(self.image_filenames) * self.repeat

    def __getitem__(self, idx):
        true_idx = idx % len(self.image_filenames)
        img_filename = self.image_filenames[true_idx]

        latent = self.latents[img_filename]
        positive_embeds = self.positive_embeds[img_filename]

        return latent, positive_embeds
    
    

class ControlNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, vae, tokenizer, text_encoder,size=256, repeat=1):
        self.image_dir = os.path.join(root_dir, "images")
        self.cond_dir = os.path.join(root_dir, "conditionings")
        self.caption_dir = os.path.join(root_dir, "captions")
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.size = (size,size)
        self.repeat=repeat

        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.image_latents = {}
        self.cond_tensors = {}
        self.positive_embeds = {}
        # latent 変換
        for img_filename in tqdm(self.image_filenames):
            #image
            img_path = os.path.join(self.image_dir, img_filename)
            image = Image.open(img_path).convert("RGB").resize(self.size, resample=Image.NEAREST)
            image_latent = encode_image(image,vae)
            self.image_latents[img_filename] = image_latent.squeeze(0)
            #cond
            cond_img_path = os.path.join(self.cond_dir, img_filename)
            cond_image = Image.open(cond_img_path).convert("RGB").resize(self.size, resample=Image.NEAREST)
            cond_img_tensor = image_to_tensor(cond_image).squeeze(0).to(device=image_latent.device,dtype=image_latent.dtype)
            self.cond_tensors[img_filename] = cond_img_tensor
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

        image_latent = self.image_latents[img_filename]
        cond_tensor = self.cond_tensors[img_filename]
        positive_embeds = self.positive_embeds[img_filename]
        
        return image_latent, positive_embeds, cond_tensor