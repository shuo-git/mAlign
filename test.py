import torch
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
# from lorahub.algorithm import lorahub_learning, lorahub_inference
# from lorahub.constant import LORA_MODULE_NAMES
import random
import bmtrain as bmt
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer
import argparse
import sys
from Init_lora import get_lora_weight,init_lora_weight
sys.path.append("/home/pingbowen/workspace/mAlign/ModelCenter")
sys.path.append("/home/pingbowen/workspace/mAlign")

def get_model_tokenizer(args):
    
    bmt.init_distributed(
        seed=args.seed,
        zero_level=3,
    )
    
    
    bmt.print_rank("loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    bmt.print_rank("finished")
    bmt.print_rank("loading model...")
    model = Llama.from_pretrained(args.model_name_or_path)
    bmt.init_parameters(model)
    
    model.load_state_dict(torch.load(args.model_name_or_path + "/pytorch_model.pt"),strict=False)
    
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    print(f"{dict(model.named_parameters())['encoder.layers.0.self_att.self_attention.project_q_lora.lora_A.weight']} \n {dict(model.named_parameters())['encoder.layers.0.self_att.self_attention.project_q.weight'].shape}") 
    
    if args.enable_init:
        finetuned_model = "/data/public/opensource_models/WizardLM/WizardMath-7B-V1.0"
        pretrained_model = "/data/public/opensource_models/meta-llama/Llama-2-7b-hf"
        init_lora_weight(model,finetuned_model=finetuned_model,pretrained_model=pretrained_model,dim=64)

    print(f"{dict(model.named_parameters())['encoder.layers.0.self_att.self_attention.project_q.weight']} \n {dict(model.named_parameters())['encoder.layers.0.self_att.self_attention.project_q.weight'].shape}") 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_name_or_path", default='/data/public/opensource_models/meta-llama/Llama-2-7b-mc')
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--enable_init",action="store_true", help="lora weight form Wizard_math")
    args = parser.parse_args()
    
    get_model_tokenizer(args)

# # get a list of modules to be used in the composition
# modules = get_lora_module_list()
# print("modules:", modules)

# # construct input list and output list
# example_inputs, examples_outputs = [], []
# for example in get_examples_for_learning():
#     example_inputs.append(example["input"])
#     examples_outputs.append(example["output"])

# # perform LoRAHub learning
# module_weights, model, tokenizer = lorahub_learning(lora_module_list=modules,
#                                                     example_inputs=example_inputs,
#                                                     example_outputs=examples_outputs,
#                                                     max_inference_step=40,
#                                                     batch_size=1)

# print("module_weights:", module_weights)





# dim = [-0.001,0,0.001]

# for i in range(24,32):
#     k_tuned = dict(finetuned_model.named_parameters())[f"model.layers.{i}.self_attn.k_proj.weight"] 
#     # k_origin = dict(pretrained_model.named_parameters())[f"model.layers.{i}.self_attn.k_proj.weight"]

#     # diff = (k_tuned - k_origin).detach()
#     diff = k_tuned.detach()
#     count = np.sum((diff.numpy() >= -0.001) & (diff.numpy() <= 0.001))
#     print(count / (4096 * 4096))


#  /data/public/opensource_models/codellama/codellama-7b-python-hf    /data/public/opensource_models/WizardLM/Llama-2-7B-32K-Instruct 

# /data/public/opensource_models/meta-llama/Llama-2-7b-hf      /data/public/opensource_models/WizardLM/WizardMath-7B-V1.0

# /data/public/opensource_models/WizardLM/Yarn-Llama-2-7b-128k


# mpl.rcParams['font.size'] = 16  # 设置字体大小为12
'''
w_in = ["50.27%" ,"51.02%" ,"51.48%" ,"52.01%" ,"53.37%","54.28%"]
w_gate = ["49.66%" ,"50.64%" ,"53.07%" ,"52.69%" ,"54.66%","54.06%"]
w_out = ["50.64%" ,"51.02%" ,"50.95%" ,"52.62%" ,"52.76%","54.05%"]

w_in_float = [float(val.strip('%')) / 100 for val in w_in]
w_gate_float = [float(val.strip('%')) / 100 for val in w_gate]
w_out_float = [float(val.strip('%')) / 100 for val in w_out]
'''


'''
plt.figure(figsize=(8, 6))
plt.plot(dim, w_in_float, marker='o', linestyle='-',label='Up-Proj')
plt.plot(dim, w_gate_float, marker='o', linestyle='-',label='Gate-Proj')
plt.plot(dim, w_out_float, marker='o', linestyle='-',label='Down-Proj')

plt.xlabel('Number of Retained Singular Values', fontsize=18)
plt.ylabel('Model Performance on GSM8K', fontsize=18)

plt.axhline(y=0.55268,color='gray', linestyle='--')
plt.text(5, 0.553, 'Full-Rank Delta Weight', color='black', ha='left', va='bottom')
'''

# plt.figure(figsize=(8, 6))

# plt.plot(dim, m_7b_float, marker='o', linestyle='-',label='WizardMath-7B')
# plt.plot(dim, m_13b_float, marker='o', linestyle='-',label='WizardMath-13B')
# plt.xticks(dim, ['64', '128', '256', '512','768','1024'])
# plt.legend(loc='lower right')

# plt.axhline(y=0.55268,color='gray', linestyle='--')
# plt.text(5, 0.553, 'Full-Rank Delta Weight-7b', color='black', ha='left', va='bottom')

# plt.axhline(y=0.6353,color='purple', linestyle='--')
# plt.text(5, 0.6353, 'Full-Rank Delta Weight-13b', color='black', ha='left', va='bottom')

# plt.xlabel('Number of Retained Singular Values', fontsize=18)
# plt.ylabel('Model Performance on GSM8K', fontsize=18)

# plt.grid(True)
# plt.ylim(0.48, 0.58)

# plt.hist(diff, bins=dim, edgecolor='blue')
# plt.show()
# plt.savefig('output3.png', format='png')
# plt.savefig('output.pdf', format='pdf')
