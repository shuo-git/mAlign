import sys
import torch

# tmp_path = "/home/wanghanqing/projects/utils/transfer_tmp.bin"


lora_extract_dict = {
    "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/en/checkpoints/epoch_2/pytorch_model.pt": "/home/wanghanqing/projects/exp/mAlign_exp/mAlign_LoRAs/en/lora.pt",
    "/home/wanghanqing/projects/exp/mAlign_exp/lang_LoRAs/zh/checkpoints/epoch_2/pytorch_model.pt": "/home/wanghanqing/projects/exp/mAlign_exp/mAlign_LoRAs/zh/lora.pt",
    "/home/wanghanqing/projects/exp/mAlign_exp/LoRAs/math_new_prompt/checkpoints/epoch_2/pytorch_model.pt": "/home/wanghanqing/projects/exp/mAlign_exp/mAlign_LoRAs/math/lora.pt",
    "/home/wanghanqing/projects/exp/mAlign_exp/LoRAs/multitask_LoRA/checkpoints/epoch_2/pytorch_model.pt": "/home/wanghanqing/projects/exp/mAlign_exp/mAlign_LoRAs/multitask/lora.pt",
}




def filter_only_lora(input_model_path, output_model_path):
    model_state_dict = torch.load(input_model_path)

    # 创建一个新的状态字典，只包含包含特定关键字的参数
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if 'lora' in k}

    # 保存过滤后的状态字典到新文件
    torch.save(filtered_state_dict, output_model_path)

for path in lora_extract_dict.keys():
    filter_only_lora(path, lora_extract_dict[path])

# encoder.layers[31].self_att.self_attention.project_q_lora.zh.lora_B.weight
# finished                                                                                                   
# encoder.layers.31.lora_fusion_gate.weight                                                                  
# encoder.layers.31.self_att.layernorm_before_attention.weight               
# encoder.layers.31.self_att.self_attention.project_q.weight                 
# encoder.layers.31.self_att.self_attention.project_q_lora.zh.lora_A.weight
# encoder.layers.31.self_att.self_attention.project_q_lora.zh.lora_B.weight    
# encoder.layers.31.self_att.self_attention.project_q_lora.math.lora_A.weight  
# encoder.layers.31.self_att.self_attention.project_q_lora.math.lora_B.weight    
# encoder.layers.31.self_att.self_attention.project_k.weight                     
# encoder.layers.31.self_att.self_attention.project_k_lora.zh.lora_A.weight
# encoder.layers.31.self_att.self_attention.project_k_lora.zh.lora_B.weight  
# encoder.layers.31.self_att.self_attention.project_k_lora.math.lora_A.weight
# encoder.layers.31.self_att.self_attention.project_k_lora.math.lora_B.weight
# encoder.layers.31.self_att.self_attention.project_v.weight                   
# encoder.layers.31.self_att.self_attention.project_v_lora.zh.lora_A.weight    
# encoder.layers.31.self_att.self_attention.project_v_lora.zh.lora_B.weight      
# encoder.layers.31.self_att.self_attention.project_v_lora.math.lora_A.weight    
# encoder.layers.31.self_att.self_attention.project_v_lora.math.lora_B.weight                                
# encoder.layers.31.self_att.self_attention.attention_out.weight                                             
# encoder.layers.31.self_att.self_attention.attention_out_lora.zh.lora_A.weight
# encoder.layers.31.self_att.self_attention.attention_out_lora.zh.lora_B.weight
# encoder.layers.31.self_att.self_attention.attention_out_lora.math.lora_A.weight
# encoder.layers.31.self_att.self_attention.attention_out_lora.math.lora_B.weight
# encoder.layers.31.ffn.layernorm_before_ffn.weight                                                          
# encoder.layers.31.ffn.ffn.w_in.w_0.weight                                                                  
# encoder.layers.31.ffn.ffn.w_in.w_0_lora.zh.lora_A.weight                    
# encoder.layers.31.ffn.ffn.w_in.w_0_lora.zh.lora_B.weight  
# encoder.layers.31.ffn.ffn.w_in.w_0_lora.math.lora_A.weight
# encoder.layers.31.ffn.ffn.w_in.w_0_lora.math.lora_B.weight
# encoder.layers.31.ffn.ffn.w_in.w_1.weight                                                                  
# encoder.layers.31.ffn.ffn.w_in.w_1_lora.zh.lora_A.weight
# encoder.layers.31.ffn.ffn.w_in.w_1_lora.zh.lora_B.weight
# encoder.layers.31.ffn.ffn.w_in.w_1_lora.math.lora_A.weight
# encoder.layers.31.ffn.ffn.w_in.w_1_lora.math.lora_B.weight                  
# encoder.layers.31.ffn.ffn.w_out.weight
# encoder.layers.31.ffn.ffn.w_out_lora.zh.lora_A.weight 
# encoder.layers.31.ffn.ffn.w_out_lora.zh.lora_B.weight 
# encoder.layers.31.ffn.ffn.w_out_lora.math.lora_A.weight
# encoder.layers.31.ffn.ffn.w_out_lora.math.lora_B.weight
