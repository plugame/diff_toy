from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers, convert_state_dict_to_peft,convert_unet_state_dict_to_peft
import collections
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
import re
import torch

UNET_SAFETENSORS_TO_DIFFUSERS_PREFIX={
    "input_blocks.0.0": "conv_in",
    "input_blocks.1.0.": "down_blocks.0.resnets.0.",
    "input_blocks.2.0.": "down_blocks.0.resnets.1.",
    "input_blocks.3.0.": "down_blocks.0.downsamplers.0.",
    "input_blocks.4.0.": "down_blocks.1.resnets.0.",
    "input_blocks.4.1.": "down_blocks.1.attentions.0.",
    "input_blocks.5.0.": "down_blocks.1.resnets.1.",
    "input_blocks.5.1.": "down_blocks.1.attentions.1.",
    "input_blocks.6.0.": "down_blocks.1.downsamplers.0.",
    "input_blocks.7.0.": "down_blocks.2.resnets.0.",
    "input_blocks.7.1.": "down_blocks.2.attentions.0.",
    "input_blocks.8.0.": "down_blocks.2.resnets.1.",
    "input_blocks.8.1.": "down_blocks.2.attentions.1.",
    "middle_block.0.": "mid_block.resnets.0.",
    "middle_block.1.": "mid_block.attentions.0.",
    "middle_block.2.": "mid_block.resnets.1.",
    "output_blocks.0.0.": "up_blocks.0.resnets.0.",
    "output_blocks.0.1.": "up_blocks.0.attentions.0.",
    "output_blocks.1.0.": "up_blocks.0.resnets.1.",
    "output_blocks.1.1.": "up_blocks.0.attentions.1.",
    "output_blocks.2.0.": "up_blocks.0.resnets.2.",
    "output_blocks.2.1.": "up_blocks.0.attentions.2.",
    "output_blocks.2.2.": "up_blocks.0.upsamplers.0.",
    "output_blocks.3.0.": "up_blocks.1.resnets.0.",
    "output_blocks.3.1.": "up_blocks.1.attentions.0.",
    "output_blocks.4.0.": "up_blocks.1.resnets.1.",
    "output_blocks.4.1.": "up_blocks.1.attentions.1.",
    "output_blocks.5.0.": "up_blocks.1.resnets.2.",
    "output_blocks.5.1.": "up_blocks.1.attentions.2.",
    "output_blocks.5.2.": "up_blocks.1.upsamplers.0.",
    "output_blocks.6.0.": "up_blocks.2.resnets.0.",
    "output_blocks.7.0.": "up_blocks.2.resnets.1.",
    "output_blocks.8.0.": "up_blocks.2.resnets.2.",
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    "out.0": "conv_norm_out",
    "out.2": "conv_out",
    "label_emb.0.0": "add_embedding.linear_1",
    "label_emb.0.2": "add_embedding.linear_2",
}

UNET_SAFETENSORS_TO_DIFFUSERS_MODULE = {
    "skip_connection":"conv_shortcut",
    "emb_layers.1": "time_emb_proj",
    "in_layers.0": "norm1",
    "in_layers.2": "conv1",
    "out_layers.0": "norm2",
    "out_layers.3": "conv2",
    "op": "conv",
}

TEXT_ENCODER_2_SAFETENSORS_TO_DIFFUSERS={
    "ln_1" :	"layer_norm1",
    "ln_2" :	"layer_norm2",
    "mlp.c_fc" : "mlp.fc1",
    "mlp.c_proj" : "mlp.fc2",
    "attn.out_proj" : "self_attn.out_proj",
    "ln_final" : "text_model.final_layer_norm",
    "positional_embedding" : "text_model.embeddings.position_embedding.weight",
    "text_projection" : "text_projection.weight",
    "token_embedding" : "text_model.embeddings.token_embedding",
    "logit_scale" : None

}

def replace_key(key:str,pattern:dict):
    for k,v in pattern.items():
        if k in key:
            if v is None: return None
            new_key = key.replace(k,v)
            return new_key
    return key

def _convert_diffusers_unet_dict(dict):
    new_state_dict={}
    for key,value in dict.items():
        new_key = replace_key(key,UNET_SAFETENSORS_TO_DIFFUSERS_PREFIX)
        new_key = replace_key(new_key,UNET_SAFETENSORS_TO_DIFFUSERS_MODULE)
        new_state_dict[new_key] = value

    return new_state_dict

def _convert_diffusers_te2_dict(dict):
    new_state_dict={}
    for key,value in dict.items():
        new_key = key.replace("transformer.resblocks.","text_model.encoder.layers.")
        new_key = replace_key(new_key,TEXT_ENCODER_2_SAFETENSORS_TO_DIFFUSERS)

        if new_key is None: continue

        if "attn.in_proj" in key:
            q_proj, k_proj, v_proj = torch.chunk(value, 3)
                
            # 新しいキー名で辞書に追加
            layer_num = new_key.split(".")[3] # layers.{i} の番号を取得
            ext = 'weight' if 'weight' in key else 'bias'
            new_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.{ext}"] = q_proj
            new_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.{ext}"] = k_proj
            new_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.{ext}"] = v_proj
        else:
            new_state_dict[new_key] = value

    return new_state_dict
    

def weights_test(weights):
    UNET_PREFIX = "model.diffusion_model."
    TE1_PREFIX = "conditioner.embedders.0.transformer."
    TE2_PREFIX = "conditioner.embedders.1.model."
    unet_state_dict = {}
    te1_state_dict = {}
    te2_state_dict = {}
    other_dict = {}
    for key, value in weights.items():
        if key.startswith(UNET_PREFIX):
            # プレフィックスを削除してキーを整形
            new_key = key[len(UNET_PREFIX):]
            unet_state_dict[new_key] = value
        elif key.startswith(TE1_PREFIX):
            new_key = key[len(TE1_PREFIX):]
            te1_state_dict[new_key] = value
        elif key.startswith(TE2_PREFIX):
            new_key = key[len(TE2_PREFIX):]
            te2_state_dict[new_key] = value
        else:
            other_dict[key] = value


    unet_state_dict = _convert_diffusers_unet_dict(unet_state_dict)
    te2_state_dict = _convert_diffusers_te2_dict(te2_state_dict)

    return unet_state_dict, te1_state_dict, te2_state_dict
    

def convert_injectable_dict_from_khoya_weight(lora_weight):
    state_dict, network_alphas = _convert_non_diffusers_lora_to_diffusers(lora_weight)
    unet_lora_dict = {k.removeprefix(f"unet."): v for k, v in state_dict.items() if k.startswith(f"{"unet"}.")}
    unet_lora_dict = convert_unet_state_dict_to_peft(unet_lora_dict)

    te_lora_dict = {k.removeprefix(f"text_encoder."): v for k, v in state_dict.items() if k.startswith(f"{"text_encoder"}.")}
    te_lora_dict = convert_state_dict_to_diffusers(te_lora_dict)
    te_lora_dict = convert_state_dict_to_peft(te_lora_dict)



    return unet_lora_dict, te_lora_dict, network_alphas

def convert_injectable_dict_from_khoya_weight_sdxl(lora_weight):
    state_dict, network_alphas = _convert_non_diffusers_lora_to_diffusers(lora_weight)

    
    unet_lora_dict = {k.removeprefix(f"unet."): v for k, v in state_dict.items() if k.startswith(f"{"unet"}.")}
    #unet_lora_dict = convert_state_dict_to_diffusers(unet_lora_dict)
    unet_lora_dict = convert_unet_state_dict_to_peft(unet_lora_dict)
    unet_lora_dict = transform_state_dict(unet_lora_dict)

    te_lora_dict = {k.removeprefix(f"text_encoder."): v for k, v in state_dict.items() if k.startswith(f"{"text_encoder"}.")}
    te_lora_dict = convert_state_dict_to_diffusers(te_lora_dict)
    te_lora_dict = convert_state_dict_to_peft(te_lora_dict)

    te2_lora_dict = {k.removeprefix(f"text_encoder_2."): v for k, v in state_dict.items() if k.startswith(f"{"text_encoder_2"}.")}
    if te2_lora_dict:
        te2_lora_dict = convert_state_dict_to_diffusers(te2_lora_dict)
        te2_lora_dict = convert_state_dict_to_peft(te2_lora_dict)
        te_lora_dict = {"text_encoder": te_lora_dict, "text_encoder_2": te2_lora_dict}

    return unet_lora_dict, te_lora_dict, network_alphas


def _convert_unet_lora_key_sdxl(key):
    parts = key.split('.')

    if parts[0] == "down_blocks" and len(parts) >2:
        block_idx = parts[1]

        new_parts=None
        if block_idx == '4':
            new_parts = ['1', 'attentions', '0']
        elif block_idx == '5':
            new_parts = ['1', 'attentions', '1']
        elif block_idx == '7':
            new_parts = ['2', 'attentions', '0']
        elif block_idx == '8':
            new_parts = ['2', 'attentions', '1']

        if new_parts is not None:
            remaining_parts = parts[3:]
            new_key = [parts[0]] + new_parts + remaining_parts
            return '.'.join(new_key)
        
    elif parts[0] == "up_blocks" and len(parts) >2:
        block_idx = parts[1]

        new_parts=None
        if block_idx == '0':
            new_parts = ['0', 'attentions', '0']
        elif block_idx == '1':
            new_parts = ['0', 'attentions', '1']
        elif block_idx == '2':
            new_parts = ['0', 'attentions', '2']
        elif block_idx == '3':
            new_parts = ['1', 'attentions', '0']
        elif block_idx == '4':
            new_parts = ['1', 'attentions', '1']
        elif block_idx == '5':
            new_parts = ['1', 'attentions', '2']

        if new_parts is not None:
            remaining_parts = parts[3:]
            new_key = [parts[0]] + new_parts + remaining_parts
            return '.'.join(new_key)
    elif parts[0] == "mid_block" and len(parts) >1:
        block_idx = parts[1]

        new_parts=None
        if block_idx=='1':
            new_parts = ['attentions', '0']

        if new_parts is not None:
            remaining_parts = parts[2:]
            new_key = [parts[0]] + new_parts + remaining_parts
            return '.'.join(new_key)

    return key

def transform_state_dict(original_state_dict):
    # 変換後のキーと値を格納するための新しい空の辞書を作成
    new_state_dict = {}
    
    # 元のstate_dictの各要素（キーと値のペア）に対してループ処理
    for key, value in original_state_dict.items():
        # ヘルパー関数を呼び出して、キーを変換する
        new_key = _convert_unet_lora_key_sdxl(key)
        
        # 新しいキーと元の値のペアを、新しい辞書に追加する
        new_state_dict[new_key] = value
        
    # 全てのキーの変換が終わったら、新しい辞書を返す
    return new_state_dict



def _convert_unet_lora_key(key):
    """
    Converts a U-Net LoRA key to a Diffusers compatible key.
    """
    diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

    # Replace common U-Net naming patterns.
    diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
    diffusers_name = diffusers_name.replace("middle.block", "mid_block")
    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
    diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
    diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("proj.in", "proj_in")
    diffusers_name = diffusers_name.replace("proj.out", "proj_out")
    diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")

    # SDXL specific conversions.
    if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
        pattern = r"\.\d+(?=\D*$)"
        diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
    if ".in." in diffusers_name:
        diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
    if ".out." in diffusers_name:
        diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
    if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
        diffusers_name = diffusers_name.replace("op", "conv")
    if "skip" in diffusers_name:
        diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

    # LyCORIS specific conversions.
    if "time.emb.proj" in diffusers_name:
        diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
    if "conv.shortcut" in diffusers_name:
        diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")

    # General conversions.
    if "transformer_blocks" in diffusers_name:
        if "attn1" in diffusers_name or "attn2" in diffusers_name:
            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
        elif "ff" in diffusers_name:
            pass
    elif any(key in diffusers_name for key in ("proj_in", "proj_out")):
        pass
    else:
        pass

    return diffusers_name

def _convert_text_encoder_lora_key(key, lora_name):
    """
    Converts a text encoder LoRA key to a Diffusers compatible key.
    """
    if lora_name.startswith(("lora_te_", "lora_te1_")):
        key_to_replace = "lora_te_" if lora_name.startswith("lora_te_") else "lora_te1_"
    else:
        key_to_replace = "lora_te2_"

    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")

    if "self_attn" in diffusers_name or "text_projection" in diffusers_name:
        pass
    elif "mlp" in diffusers_name:
        # Be aware that this is the new diffusers convention and the rest of the code might
        # not utilize it yet.
        diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")

    return diffusers_name

def _get_alpha_name(lora_name_alpha, diffusers_name, alpha):
    """
    Gets the correct alpha name for the Diffusers model.
    """
    if lora_name_alpha.startswith("lora_unet_"):
        prefix = "unet."
    elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
        prefix = "text_encoder."
    else:
        prefix = "text_encoder_2."
    new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
    return {new_name: alpha}

def _convert_non_diffusers_lora_to_diffusers(state_dict, unet_name="unet", text_encoder_name="text_encoder"):
    """
    Converts a non-Diffusers LoRA state dict to a Diffusers compatible state dict.

    Args:
        state_dict (`dict`): The state dict to convert.
        unet_name (`str`, optional): The name of the U-Net module in the Diffusers model. Defaults to "unet".
        text_encoder_name (`str`, optional): The name of the text encoder module in the Diffusers model. Defaults to
            "text_encoder".

    Returns:
        `tuple`: A tuple containing the converted state dict and a dictionary of alphas.
    """
    unet_state_dict = {}
    te_state_dict = {}
    te2_state_dict = {}
    network_alphas = {}

    # Check for DoRA-enabled LoRAs.
    dora_present_in_unet = any("dora_scale" in k and "lora_unet_" in k for k in state_dict)
    dora_present_in_te = any("dora_scale" in k and ("lora_te_" in k or "lora_te1_" in k) for k in state_dict)
    dora_present_in_te2 = any("dora_scale" in k and "lora_te2_" in k for k in state_dict)


    # Iterate over all LoRA weights.
    all_lora_keys = list(state_dict.keys())
    for key in all_lora_keys:
        if not key.endswith("lora_down.weight"):
            continue

        # Extract LoRA name.
        lora_name = key.split(".")[0]

        # Find corresponding up weight and alpha.
        lora_name_up = lora_name + ".lora_up.weight"
        lora_name_alpha = lora_name + ".alpha"

        # Handle U-Net LoRAs.
        if lora_name.startswith("lora_unet_"):
            diffusers_name = _convert_unet_lora_key(key)

            # Store down and up weights.
            unet_state_dict[diffusers_name] = state_dict.pop(key)
            unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # Store DoRA scale if present.
            if dora_present_in_unet:
                dora_scale_key_to_replace = "_lora.down." if "_lora.down." in diffusers_name else ".lora.down."
                unet_state_dict[diffusers_name.replace(dora_scale_key_to_replace, ".lora_magnitude_vector.")] = (
                    state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                )

        # Handle text encoder LoRAs.
        elif lora_name.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
            diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

            # Store down and up weights for te or te2.
            if lora_name.startswith(("lora_te_", "lora_te1_")):
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
            else:
                te2_state_dict[diffusers_name] = state_dict.pop(key)
                te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # Store DoRA scale if present.
            if dora_present_in_te or dora_present_in_te2:
                dora_scale_key_to_replace_te = (
                    "_lora.down." if "_lora.down." in diffusers_name else ".lora_linear_layer."
                )
                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    te_state_dict[diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")] = (
                        state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                    )
                elif lora_name.startswith("lora_te2_"):
                    te2_state_dict[diffusers_name.replace(dora_scale_key_to_replace_te, ".lora_magnitude_vector.")] = (
                        state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                    )

        # Store alpha if present.
        if lora_name_alpha in state_dict:
            alpha = state_dict.pop(lora_name_alpha).item()
            network_alphas.update(_get_alpha_name(lora_name_alpha, diffusers_name, alpha))

    # Check if any keys remain.
    if len(state_dict) > 0:
        raise ValueError(f"The following keys have not been correctly renamed: \n\n {', '.join(state_dict.keys())}")


    # Construct final state dict.
    unet_state_dict = {f"{unet_name}.{module_name}": params for module_name, params in unet_state_dict.items()}
    te_state_dict = {f"{text_encoder_name}.{module_name}": params for module_name, params in te_state_dict.items()}
    te2_state_dict = (
        {f"text_encoder_2.{module_name}": params for module_name, params in te2_state_dict.items()}
        if len(te2_state_dict) > 0
        else None
    )
    if te2_state_dict is not None:
        te_state_dict.update(te2_state_dict)

    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict, network_alphas


def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True):
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]

    if len(set(rank_dict.values())) > 1: # 1. rank にバラつきがあるか確認
    # 2. 最もよく使われている rank（=標準）を取得
        most_common_rank = collections.Counter(rank_dict.values()).most_common(1)[0][0]
    # 3. 標準 rank ではないものだけ抽出（例外）
        non_standard_ranks = {
            key: rank for key, rank in rank_dict.items() if rank != most_common_rank
        }
    # 4. `.lora_B.` より前の部分をキーとして記録（レイヤー名抽出）
        rank_pattern = {
        key.split(".lora_B.")[0]: rank
        for key, rank in non_standard_ranks.items()
    }
    else:
        rank_pattern = {}

    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        if len(set(network_alpha_dict.values())) > 1:
            # get the alpha occurring the most number of times
            lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]

            # for modules with alpha different from the most occurring alpha, add it to the `alpha_pattern`
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
            if is_unet:
                alpha_pattern = {
                    ".".join(k.split(".lora_A.")[0].split(".")).replace(".alpha", ""): v
                    for k, v in alpha_pattern.items()
                }
            else:
                alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}
        else:
            lora_alpha = set(network_alpha_dict.values()).pop()

    # layer names without the Diffusers specific
    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})

    use_dora = any("lora_magnitude_vector" in k for k in peft_state_dict)
    # for now we know that the "bias" keys are only associated with `lora_B`.
    lora_bias = any("lora_B" in k and k.endswith(".bias") for k in peft_state_dict)

    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
        "use_dora": use_dora,
        "lora_bias": lora_bias,
    }
    return lora_config_kwargs

def get_adapter_name(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return f"default_{len(module.r)}"
    return "default_0"

def load_lora_into_text_encoder(
    state_dict,
    network_alphas,
    text_encoder,
    lora_scale=1.0,
    text_encorder_name="text_encoder",
    adapter_name=None
    
):
    prefix = text_encorder_name
    #text_encorderのみのこす
    state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}

    state_dict = convert_state_dict_to_diffusers(state_dict)
    # convert state dict
    state_dict = convert_state_dict_to_peft(state_dict)

    # lora_Bけいのkeyを残し、valueはそのrankを代入
    rank = {k: v.shape[1] for k, v in state_dict.items() if "lora_A" not in k}
       
    if network_alphas is not None:
        network_alphas = {k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k.startswith(f"{prefix}.")}

    lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict, is_unet=False)

    if adapter_name is None:
        adapter_name = get_adapter_name(text_encoder)
    lora_config = LoraConfig(**lora_config_kwargs)

    text_encoder.load_adapter(
            adapter_name=adapter_name,
            adapter_state_dict=state_dict,
            peft_config=lora_config,
        )
    

    text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)
    return state_dict


def load_lora_into_unet(
        state_dict,
        network_alphas,
        unet,
        unet_name="unet",
        adapter_name=None,
):

    prefix = unet_name

    state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}

    first_key = next(iter(state_dict.keys()))
    if "lora_A" not in first_key:
        state_dict = convert_unet_state_dict_to_peft(state_dict)

    rank = {}
    for key, val in state_dict.items():
        # Bias layers in LoRA only have a single dimension
        if "lora_B" in key and val.ndim > 1:
            rank[f"^{key}"] = val.shape[1]


    if network_alphas is not None and len(network_alphas) >= 1:
        alpha_keys = [k for k in network_alphas.keys() if k.startswith(f"{prefix}.")]
        network_alphas = {
            k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys
        }

    lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict)

    if adapter_name is None:
        adapter_name = get_adapter_name(unet)
    lora_config = LoraConfig(**lora_config_kwargs)

    inject_adapter_in_model(lora_config, unet, adapter_name=adapter_name)
    incompatible_keys = set_peft_model_state_dict(unet, state_dict, adapter_name)
    return state_dict

    

