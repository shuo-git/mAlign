import json
import sys


ipt_filename = sys.argv[1]
opt_filename = sys.argv[2]

with open(ipt_filename, 'r') as fr:
    orig_content = json.load(fr)

ret_content = []
for idx, item in enumerate(orig_content):
    ret_item = {}
    model = item.get("model", "")
    if model == "Model: GPT-4":
        ret_item['id'] = f"{idx}_gpt4"
    else:
        ret_item['id'] = f"{idx}_chatgpt"
    ret_item['conversations'] = item['items']
    ret_content.append(ret_item)

with open(opt_filename, 'w') as fw:
    json.dump(ret_content, fw, ensure_ascii=False, indent=2)
