"""for llama 2"""
import torch, os
import json
from collections import OrderedDict
from tqdm import tqdm
import os
# from model_center.model.config import LlamaConfig

def transform_to_hf(bmt_model, param_size):
    model_hf = OrderedDict()

    model_hf['model.embed_tokens.weight'] = bmt_model["input_embedding.weight"].contiguous().float()
    model_hf['model.norm.weight'] = bmt_model["encoder.output_layernorm.weight"].contiguous().float()
    model_hf['lm_head.weight'] = bmt_model['output_projection.weight'].contiguous().float()

    if "7b" in param_size:
        layernum = 32
    elif "13b" in param_size:
        layernum = 40
    elif "70b" in param_size:
        layernum = 80

    for lnum in range(layernum):
        hf_pfx = f"model.layers.{lnum}"
        bmt_pfx = f"encoder.layers.{lnum}"
        
        model_hf[f"{hf_pfx}.input_layernorm.weight"] = bmt_model[f"{bmt_pfx}.self_att.layernorm_before_attention.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.self_attn.q_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.project_q.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.self_attn.k_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.project_k.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.self_attn.v_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.project_v.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.self_attn.o_proj.weight"] = bmt_model[f"{bmt_pfx}.self_att.self_attention.attention_out.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.post_attention_layernorm.weight"] = bmt_model[f"{bmt_pfx}.ffn.layernorm_before_ffn.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.mlp.gate_proj.weight"] = bmt_model[f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight"].contiguous().float()
        model_hf[f"{hf_pfx}.mlp.up_proj.weight"] = bmt_model[f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight"].contiguous().float()

        model_hf[f"{hf_pfx}.mlp.down_proj.weight"] = bmt_model[f"{bmt_pfx}.ffn.ffn.w_out.weight"].contiguous().float()
    return model_hf

def add_configs(inpath, outpath, base_hf_config):
    bmt_config = json.load(open(os.path.join(inpath, "config.json")))
    hf_config = base_hf_config
    hf_config["hidden_size"] = bmt_config["dim_model"]
    hf_config["intermediate_size"] = bmt_config["dim_ff"]
    hf_config["num_hidden_layes"] = bmt_config["num_layers"]
    hf_config["num_attention_heads"] = bmt_config["num_heads"]
    hf_config["rms_norm_eps"] = bmt_config["norm_eps"]
    with open(os.path.join(outpath, "config.json"), 'w') as f:
        json.dump(hf_config, f)



if __name__ == "__main__":
    import sys
    import shutil
    in_path = sys.argv[1]
    model_type = sys.argv[-1]
    if "7b" in model_type:
        param_size = "7b"
    elif "13b" in model_type:
        param_size = "13b"
    elif "70b" in model_type:
        param_size = "70b"
    else:
        param_size = "7b"
        # raise ValueError(f"cannot detect param_size automatically from {model_type}")
    
    if "chat" in model_type:
        param_size += "-chat"
    
    hf_ref_config_path = f"/home/wanghanqing/projects/models/Llama-2-7b-hf/config.json"
    with open(hf_ref_config_path, "r") as fr:
        base_hf_config = json.load(fr)

    out_path = in_path + "_hf"
    os.makedirs(out_path, exist_ok=True)
    if not os.path.exists(os.path.join(out_path, "pytorch_model.bin")):
        print("transforming...")
        hf_state_dict = transform_to_hf(torch.load(os.path.join(in_path, "pytorch_model.pt")), param_size)
        print("done")
        torch.save(hf_state_dict, os.path.join(out_path, "pytorch_model.bin"))
    base_dir = f"/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/en/checkpoints/epoch_2"
    add_configs(base_dir, out_path, base_hf_config)
    for n in ["generation_config.json", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "tokenizer.model"]:
        if os.path.exists(os.path.join(base_dir, n)):
            shutil.copy(os.path.join(base_dir, n), os.path.join(out_path, n))
    print("saved")
    print(list(os.listdir(out_path)))




