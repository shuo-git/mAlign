"""for llama 2"""
import torch, os
import json
from collections import OrderedDict
from tqdm import tqdm
import os
# from model_center.model.config import LlamaConfig

# base_hf_config = {
#   "_name_or_path": "meta-llama/Llama-2-13b-hf",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 5120,
#   "initializer_range": 0.02,
#   "intermediate_size": 13824,
#   "max_length": 4096,
#   "max_position_embeddings": 4096,
#   "model_type": "llama",
#   "num_attention_heads": 40,
#   "num_hidden_layers": 40,
#   "num_key_value_heads": 40,
#   "pad_token_id": 0,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-05,
#   "rope_scaling": None,
#   "tie_word_embeddings": False,
#   "torch_dtype": "float32",
#   "transformers_version": "4.31.0",
#   "use_cache": True,
#   "vocab_size": 32000
# }

base_hf_config = {
  "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_length": 4096,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "tie_word_embeddings": False,
  "torch_dtype": "float32",
  "transformers_version": "4.31.0",
  "use_cache": True,
  "vocab_size": 32000
}


# inpath = f"/data/ultrallama/ultrallama-13b_reasoning_step5200_mc"
# outpath = f"/data/ultrallama/ultrallama-13b_reasoning_step5200_hf"
# os.makedirs(outpath, exist_ok=True)
# bmt_model = torch.load(os.path.join(inpath, "checkpoint.pt"))
def transform_to_hf(bmt_model, param_size):
    model_hf = OrderedDict()

    model_hf['model.embed_tokens.weight'] = bmt_model["input_embedding.weight"].contiguous().float()
    # print(bmt_model["input_embedding.weight"].size())
    # print("observe pad token embedding")
    # print(bmt_model["input_embedding.weight"][-1])
    # print("observe unk token embedding")
    # print(bmt_model["input_embedding.weight"][0])
    model_hf['model.norm.weight'] = bmt_model["encoder.output_layernorm.weight"].contiguous().float()
    model_hf['lm_head.weight'] = bmt_model['output_projection.weight'].contiguous().float()

    if param_size == "7b":
        layernum = 32
    elif param_size == "13b":
        layernum = 40
    elif param_size == "70b":
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

def add_configs(inpath, outpath):
    # config = {
    # 'dim_model': hf_config.hidden_size,
    # 'dim_ff': hf_config.intermediate_size,
    # 'num_layers': hf_config.num_hidden_layers,
    # 'num_heads': hf_config.num_attention_heads,
    # 'num_heads_kv': hf_config.num_key_value_heads,
    # 'dim_head': hf_config.hidden_size // hf_config.num_attention_heads,
    # 'norm_eps': hf_config.rms_norm_eps,
    # }
    bmt_config = json.load(open(os.path.join(inpath, "config.json")))
    hf_config = base_hf_config
    hf_config["hidden_size"] = bmt_config["dim_model"]
    hf_config["intermediate_size"] = bmt_config["dim_ff"]
    hf_config["num_hidden_layes"] = bmt_config["num_layers"]
    hf_config["num_attention_heads"] = bmt_config["num_heads"]
    # hf_config["num_key_value_heads"] = bmt_config["num_heads_kv"]
    hf_config["rms_norm_eps"] = bmt_config["norm_eps"]
    with open(os.path.join(outpath, "config.json"), 'w') as f:
        json.dump(hf_config, f)



if __name__ == "__main__":
    import sys
    import shutil
    in_path = sys.argv[-1]
    if "7b" in in_path:
        param_size = "7b"
    elif "13b" in in_path:
        param_size = "13b"
    elif "70b" in in_path:
        param_size = "70b"
    else:
        raise ValueError(f"cannot detect param_size automatically from {in_path}")
    # in_path = "/data/checkpoints/ultrallama/ultrachat_llama-65b/step_600"
    out_path = in_path + "_hf"
    os.makedirs(out_path, exist_ok=True)
    if not os.path.exists(os.path.join(out_path, "pytorch_model.bin")):
        print("transforming...")
        hf_state_dict = transform_to_hf(torch.load(os.path.join(in_path, "pytorch_model.pt")), param_size)
        print("done")
        torch.save(hf_state_dict, os.path.join(out_path, "pytorch_model.bin"))
    base_dir = f"/data/llama-2-{param_size}"
    add_configs(base_dir, out_path)
    for n in ["generation_config.json", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "tokenizer.model"]:
        if os.path.exists(os.path.join(base_dir, n)):
            shutil.copy(os.path.join(base_dir, n), os.path.join(out_path, n))
    print("saved")
    print(list(os.listdir(out_path)))




