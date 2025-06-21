import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL,UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import torch.nn as nn
import math

def get_optimal_torch_dtype(dtype_name:str):
    if dtype_name == "fp16":
        return torch.float16,torch.float32
    elif dtype_name == "bf16":
        return torch.bfloat16,torch.bfloat16
    else:
        return torch.float32,torch.float32
    
def load_image(img_path,size:tuple[int,int]=None,resample=Image.NEAREST):
    if size is not None:
        return Image.open(img_path).convert("RGB").resize(size, resample=resample)
    return Image.open(img_path).convert("RGB")

def image_to_tensor(image: Image.Image, resize:tuple[int,int]=None)->torch.Tensor:
    if resize is None:
        width, height = image.size
    else:
        width, height = resize

    transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor() # これにより [0, 1] になる
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def encode_image(image: Image.Image, vae: AutoencoderKL) -> torch.Tensor:
    # 元のサイズを取得し、8の倍数に丸める
    orig_w, orig_h = image.size
    width = max(8, int(round(orig_w / 8) * 8))
    height = max(8, int(round(orig_h / 8) * 8))
    size = (width,height)

    image_tensor = image_to_tensor(image,size).to(vae.device, dtype=vae.dtype)
    image_tensor = image_tensor * 2 - 1 # [0, 1] -> [-1, 1] に変換
    with torch.no_grad():
        encoded = vae.encode(image_tensor)
        latents = encoded.latent_dist.sample() * vae.config.scaling_factor
    return latents # shape: (1, 4, 64, 64)

def decode_latents(latents: torch.Tensor, vae: AutoencoderKL) -> Image.Image:
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1) # ここは [-1, 1] -> [0, 1]
    image_np = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image_uint8 = (image_np * 255).round().astype("uint8")
    return Image.fromarray(image_uint8[0])


def encode_prompt(
        prompt, 
        tokenizer, 
        text_encoder, 
        negative_prompt=None
        ):
    
    def get_embeds(prompts):
        inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.to(device=text_encoder.device)

        with torch.no_grad():
            embeds = text_encoder(input_ids)[0]  # (batch, seq_len, dim)
        return embeds
    
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # Positive prompt
    
    positive_embeds = get_embeds(prompt)

    # Negative prompt（空文字列で埋める）
    if negative_prompt is None:
        negative_prompt = [""] * len(prompt)
    negative_embeds = get_embeds(negative_prompt)
    
    return positive_embeds, negative_embeds


def prepare_empty_latent(
        width,
        height,
        scheduler,
        device,
        dtype=torch.float16,
        batch_size = 1
        ):
    latents = torch.randn((batch_size, 4, height//8, width//8), device=device, dtype=dtype)
    return latents * scheduler.init_noise_sigma


def get_model_prefix(model):
    if isinstance(model,UNet2DConditionModel):
        return "unet"
    elif isinstance(model,CLIPTextModel):
        return "text_encoder"
    else:
        return None

def get_module_by_key(model, key):
    parts = key.split('.')
    module = model
    for p in parts[:-1]:
        if p.isdigit():
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module, parts[-1]

def convert_injectable_dict_from_weight(lora_weight):
    unet_lora_dict = {k.removeprefix("unet."): v for k, v in lora_weight.items() if k.startswith("unet.") and not k.endswith(".alpha")}
    te_lora_dict = {k.removeprefix("text_encoder."): v for k, v in lora_weight.items() if k.startswith("text_encoder.") and not k.endswith(".alpha")}
    network_alphas = {k: float(v) for k, v in lora_weight.items() if k.endswith(".alpha")}
    return unet_lora_dict, te_lora_dict, network_alphas

# 既存LoRAを注入, todo doropout
def inject_lora_from_pretrained(model, lora_state_dict, network_alphas):
    for key, value in lora_state_dict.items():
        if not key.endswith('lora_A.weight'):
            continue

        adapter_name = "default_0"
        copy_base_layer = False

        base_key = key.replace('.lora_A.weight', '')
        lora_A = value
        lora_B = lora_state_dict.get(base_key + '.lora_B.weight')
        rank = torch.tensor(lora_A.shape[0])
        #複数alphaに非対応
        alpha = torch.tensor(list(network_alphas.values())[0])

        # 対象モジュールの取得
        parent_module, attr_name = get_module_by_key(model, base_key)
        base_layer = getattr(parent_module, attr_name)
        # LoRA ラッパーで置換
        if isinstance(base_layer, nn.Linear):
            copy_base_layer=True
            lora_layer = LoRALinear(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                bias=base_layer.bias is not None
            )
        elif isinstance(base_layer, nn.Conv2d):
            copy_base_layer=True
            lora_layer = LoRAConv2d(
                in_channels=base_layer.in_channels,
                out_channels=base_layer.out_channels,
                kernel_size=base_layer.kernel_size,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
                bias=base_layer.bias is not None
            )
        elif isinstance(base_layer, (LoRALinear, LoRAConv2d)):
            lora_layer = base_layer
            idx=0
            while f"default_{idx}" in base_layer.lora_A:
                idx+=1
            adapter_name=f"default_{idx}"
        else:
            raise TypeError(f"Unsupported module type at {base_key}: {type(base_layer)}")
        
        # LoRALiner等に初めて置き換えたとき元の重みをコピー
        if copy_base_layer:
            lora_layer.base_layer.weight.data = base_layer.weight.data.clone()
            if base_layer.bias is not None:
                lora_layer.base_layer.bias.data = base_layer.bias.data.clone()

        # LoRA重みを注入
        lora_layer.add_lora(adapter_name,rank,alpha)
        lora_layer.lora_A[adapter_name].weight.data = lora_A
        lora_layer.lora_B[adapter_name].weight.data = lora_B
    
        # モジュールの置換
        setattr(parent_module, attr_name.replace('.weight', ''), lora_layer)
    # これ結構重要
    model = model.to(device=model.device, dtype=model.dtype)



# train用に初期化されたLoRAを注入
def inject_initial_lora(model, rank=4, alpha=1.0, dropout=0.0,dtype=torch.float32):
    network_alphas = {}

    #loraを注入する層か判定(unetはattention,teはmlpかattnに注入)
    def needs_lora_injection(module_name):
        if isinstance(model,UNet2DConditionModel):
            if "attention"  in module_name:
                return True
        elif isinstance(model,CLIPTextModel):
            if "mlp" in module_name or "self_attn" in module_name:
                return True
        return False

    for module_name, module in model.named_modules():
        if not needs_lora_injection(module_name):
            continue

        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prefix = get_model_prefix(model)
            network_alphas[prefix+"."+module_name+".alpha"] = torch.tensor(alpha)

            parent_module = model
            path = module_name.split(".")

            # 親モジュールを取得
            for p in path[:-1]:
                parent_module = getattr(parent_module, p)

            last_name = path[-1]
            base_layer = getattr(parent_module, last_name)
    
            # Linear の場合
            if isinstance(base_layer, nn.Linear):
                #lora_layerの空箱を作るイメージ
                lora_layer = LoRALinear(
                    in_features=base_layer.in_features,
                    out_features=base_layer.out_features,
                    bias=base_layer.bias is not None,
                    init=True
                )

            # Conv2d の場合
            elif isinstance(base_layer, nn.Conv2d):
                #lora_layerの空箱を作るイメージ
                lora_layer = LoRAConv2d(
                    in_channels=base_layer.in_channels,
                    out_channels=base_layer.out_channels,
                    kernel_size=base_layer.kernel_size,
                    stride=base_layer.stride,
                    padding=base_layer.padding,
                    dilation=base_layer.dilation,
                    groups=base_layer.groups,
                    bias=base_layer.bias is not None,
                    init=True
                )

            # もとの重みのコピー
            lora_layer.base_layer.weight.data = base_layer.weight.data.clone()
            if base_layer.bias is not None:
                lora_layer.base_layer.bias.data = base_layer.bias.data.clone()
            
            lora_layer.add_lora("",rank,alpha,dropout)
            lora_layer = lora_layer.to(dtype)

            # モジュールを置換
            setattr(parent_module, last_name, lora_layer)
    # これ結構重要
    model = model.to(device=model.device)

    return network_alphas




class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 元のLinear層
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        for param in self.base_layer.parameters():
            param.requires_grad = False  # 基本はfreeze

        # LoRAモジュール群（動的に追加）
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.lora_scaling = {}  # dict: name -> scale（float）
        self.init = init


    def add_lora(self, name, rank=4, alpha=1.0, dropout=0.0):
        """新しいLoRAを追加する"""
        if self.init:
            self.lora_A = nn.Linear(self.in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, self.out_features, bias=False)
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.lora_scaling = alpha / rank if rank > 0 else 1.0

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A[name] = nn.Linear(self.in_features, rank, bias=False)
            self.lora_B[name] = nn.Linear(rank, self.out_features, bias=False)
            self.lora_dropout[name] = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.lora_scaling[name] = alpha / rank if rank > 0 else 1.0
        


    def forward(self, x, active_loras=None):
        """active_loras: 使用するLoRAの名前リスト（Noneなら全て使う）"""
        result = self.base_layer(x)

        if self.init:
            lora_output = self.lora_B(self.lora_dropout(self.lora_A(x))) * self.lora_scaling
            result += lora_output
        else:
            active_loras = active_loras or self.lora_A.keys()
            for name in active_loras:
                A = self.lora_A[name]
                B = self.lora_B[name]
                dropout = self.lora_dropout[name]
                scale = self.lora_scaling[name]
                result += B(dropout(A(x))) * scale

        return result
    

    
class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,init=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 元のConv2D層（通常はfreeze）
        self.base_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias
        )
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA部分：複数管理
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.lora_scaling = {}  # name -> scale (float)

        self.init=init

    def add_lora(self, name, rank=4, alpha=1.0, dropout=0.0):
        if self.init:
            self.lora_A = nn.Conv2d(self.in_channels, rank, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(rank, self.out_channels, kernel_size=1, bias=False)
            self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
            self.lora_scaling = alpha / rank if rank > 0 else 1.0

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A[name] = nn.Conv2d(self.in_channels, rank, kernel_size=1, bias=False)
            self.lora_B[name] = nn.Conv2d(rank, self.out_channels, kernel_size=1, bias=False)
            self.lora_dropout[name] = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
            self.lora_scaling[name] = alpha / rank if rank > 0 else 1.0

        

    def forward(self, x, active_loras=None):
        """active_loras: 使用するLoRAの名前リスト（Noneならすべて使う）"""
        result = self.base_layer(x)
        if self.init:
            lora_output = self.lora_B(self.lora_dropout(self.lora_A(x))) * self.lora_scaling
            result += lora_output
        else:
            active_loras = active_loras or self.lora_A.keys()
            for name in active_loras:
                A = self.lora_A[name]
                B = self.lora_B[name]
                dropout = self.lora_dropout[name]
                scale = self.lora_scaling[name]
                result += scale * B(A(dropout(x)))

        return result
    
def show_model_param_status(model_or_dict,name_only=False):
    if isinstance(model_or_dict,nn.Module): 
        for name, param in model_or_dict.named_parameters():
            if name_only:
                print(name)
            else:
                print(f"{param.dtype}, {param.device}, {param.requires_grad}: {name}")
                print(param.shape)
        l = len(model_or_dict.state_dict())

    if isinstance(model_or_dict,dict): 
        for k, v in model_or_dict.items():
            if name_only:
                print(k)
            else:
                print(f"{v.dtype}, {v.device}, {v.requires_grad} : {k}")
                print(v.shape)
        
        l = len(model_or_dict)
    print("dtype, device, requires_grad : name , shape")
    print(f"length: {l}")
    

