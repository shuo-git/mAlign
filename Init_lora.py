from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import bmtrain as bmt

name_map = {"self_attn.q_proj.weight":["self_att.self_attention.project_q_lora.lora_A.weight","self_att.self_attention.project_q_lora.lora_mid.weight","self_att.self_attention.project_q_lora.lora_B.weight"],
            "self_attn.k_proj.weight":["self_att.self_attention.project_k_lora.lora_A.weight","self_att.self_attention.project_k_lora.lora_mid.weight","self_att.self_attention.project_k_lora.lora_B.weight"],
            "self_attn.v_proj.weight":["self_att.self_attention.project_v_lora.lora_A.weight","self_att.self_attention.project_v_lora.lora_mid.weight","self_att.self_attention.project_v_lora.lora_B.weight"],
            "self_attn.o_proj.weight":["self_att.self_attention.attention_out_lora.lora_A.weight","self_att.self_attention.attention_out_lora.lora_mid.weight","self_att.self_attention.attention_out_lora.lora_B.weight"],
            "mlp.up_proj.weight":["ffn.ffn.w_in.w_0_lora.lora_A.weight","ffn.ffn.w_in.w_0_lora.lora_mid.weight","ffn.ffn.w_in.w_0_lora.lora_B.weight"],
            "mlp.gate_proj.weight":["ffn.ffn.w_in.w_1_lora.lora_A.weight","ffn.ffn.w_in.w_1_lora.lora_mid.weight","ffn.ffn.w_in.w_1_lora.lora_B.weight"],
            "mlp.down_proj.weight":["ffn.ffn.w_out_lora.lora_A.weight","ffn.ffn.w_out_lora.lora_mid.weight","ffn.ffn.w_out_lora.lora_B.weight"],
            }

def get_lora_weight(finetuned_model,pretrained_model,dim):
    weight_dict = {}
    
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model, 
                                        device_map="auto")
    finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=finetuned_model, 
                                    device_map="auto")
    
    for param_name, param_value in tqdm(finetuned_model.named_parameters()):
        if "self_attn" in param_name or "mlp" in param_name:
            
            U , S , V = torch.svd(param_value)
            
            param_name = param_name.replace("model", "encoder")
            
            for key in name_map.keys():
                if key in param_name:
                    param_name1 = param_name.replace(key, name_map[key][0])
                    param_name2 = param_name.replace(key, name_map[key][1])
                    param_name3 = param_name.replace(key, name_map[key][2])
           
            weight_dict[param_name1] = U[:dim]
            weight_dict[param_name2] = S[:dim]
            weight_dict[param_name3] = V[:,:dim] 
            

    return weight_dict

def init_lora_weight(model,finetuned_model,pretrained_model,dim):
    weight_dict = get_lora_weight(finetuned_model=finetuned_model,pretrained_model=pretrained_model,dim=dim)
    
    for n,p in model.named_parameters():
        
        if "lora" in n:
            dict(model.named_parameters())[n].data = weight_dict[n]