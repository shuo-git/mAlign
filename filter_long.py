import json
from typing import *
import sys
import bmtrain as bmt
sys.path.append("/home/wangshuo1/code/mAlign/ModelCenter")
from model_center.tokenizer import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("/data/public/opensource_models/meta-llama/Llama-2-70b-chat-hf")


def get_tokenized_length(ipt_list):
    system = "<s>[INST] "
    tokenized_ids = []
    for i, c in enumerate(ipt_list):
        if i == 0:
            # system and 1st user message
            c_input = system + c + " [/INST]"
            tmp_tokenized_ids = tokenizer(c_input, add_special_tokens=False)["input_ids"]
            tokenized_ids += tmp_tokenized_ids
        elif i % 2 == 1:
            # model
            c_input = c + " </s>"
            tmp_tokenized_ids = tokenizer(c_input, add_special_tokens=False)["input_ids"]
            tokenized_ids += tmp_tokenized_ids
        else:
            # user
            c_input = "<s>[INST] " + c + " [/INST]"
            tmp_tokenized_ids = tokenizer(c_input, add_special_tokens=False)["input_ids"]
            tokenized_ids += tmp_tokenized_ids

    return len(tokenized_ids)


def load_alpaca_data(data_file, lang='en'):
    new_data = []
    content = json.load(open(data_file, 'r'))
    num_examples = len(content)
    bmt.print_rank(f"[{lang}-original data] {data_file}: {num_examples} dialogues")
    for item in content:
        temp_input = (item['instruction'].strip() + ' ' + item['input'].strip()).strip()
        temp_output = item['output'].strip()
        if get_tokenized_length([temp_input, temp_output]) > 4096:
            continue
        new_data.append(item)
    bmt.print_rank(f"[{lang}-original data] {data_file}: {len(new_data)} dialogues left")
    return new_data


def load_code_data(data_file):
    new_data = []
    with open(data_file, 'r') as fr:
        lines = fr.readlines()
    for idx, line in enumerate(lines):
        content = json.loads(line.strip())
        if 'problem' in content:
            temp_input = content['problem']
            temp_output = content['solution']
        else:
            temp_input = content['instruction']
            temp_output = content['response']
        temp_id = f"code_{idx}"
        t_l = get_tokenized_length([temp_input, temp_output])
        if t_l < 4096:
            new_data.append(line)
    return new_data


def load_jsonl_data(data_file):
    new_data = []
    with open(data_file) as f:
        lines = f.readlines()
    num_examples = len(lines)
    bmt.print_rank(f"{data_file}: {num_examples} dialogues")
    for l in lines:
        content = json.loads(l.strip())
        temp_list = content['data']
        while get_tokenized_length(temp_list) > 4096 and len(temp_list) > 2:
            temp_list = temp_list[:-2]
        if get_tokenized_length(temp_list) > 4096:
            continue
        assert len(temp_list) % 2 == 0
        new_content = {'id': content['id'], 'data': temp_list}
        new_data.append(json.dumps(new_content, ensure_ascii=False) + '\n')
    return new_data


def check_alternate_human_gpt(conv):
    length = len(conv)
    if len(conv) % 2 != 0:
        bmt.print_rank(conv)
        return False
    tags = [i for _ in range(len(conv)//2) for i in ["human", "gpt"]]
    for i in range(len(conv)):
        if tags[i] != conv[i]["from"]:
            bmt.print_rank(conv)
            return False
    return True

def load_sharegpt_data(data_file, lang='en'):
    new_data = []
    data = json.load(open(data_file, "r"))
    num_examples = len(data)
    bmt.print_rank(f"[{lang}-original data] {data_file}: {num_examples} dialogues")
    for idx, item in enumerate(data):
        conv = item["conversations"]
        if conv[0]["from"] != "human":
            conv = conv[1:]
        if conv[-1]["from"] != "gpt":
            conv = conv[:-1]
        if check_alternate_human_gpt(conv):
            dialogues = [c["value"] for c in conv]
            if get_tokenized_length(dialogues) <= 4096:
                data = {"id": item["id"], "data": dialogues}
                new_data.append(data)
        if idx % 4999 == 0:
            bmt.print_rank(f"finished {idx+1} items")
    return new_data

def load_sharegpt_q_switch_data(data_file):
    new_data = []
    data = json.load(open(data_file, "r"))
    num_examples = len(data)
    bmt.print_rank(f"[question-switching data] {data_file}: {num_examples} dialogues")
    for idx, item in enumerate(data):
        conv = item["conversations"]
        if conv[0]["from"] != "human":
            conv = conv[1:]
        if conv[-1]["from"] != "gpt":
            conv = conv[:-1]
        if check_alternate_human_gpt(conv):
            for i in range(len(conv)):
                if conv[i]['from'] == 'human':
                    conv[i]['value'] = conv[i]['value'].strip() + "\n\nPlease Answer in English."
            dialogues = [c["value"] for c in conv]
            if get_tokenized_length(dialogues) <= 4096:
                data = {"id": item["id"], "data": dialogues}
                new_data.append(data)
        if idx % 4999 == 0:
            bmt.print_rank(f"finished {idx+1} items")
    return new_data


def get_by_id(ipt_list, tgt_id):
    for item in ipt_list:
        if item['id'] == tgt_id:
            return item
    return None


if __name__ == "__main__":
    # new_dataset = load_jsonl_data(sys.argv[1])
    # with open(sys.argv[2], 'w') as fw:
    #     fw.writelines(new_dataset)
    # new_dataset = load_alpaca_data(sys.argv[1], lang=sys.argv[3])
    # json.dump(new_dataset, open(sys.argv[2], "w"), indent=2, ensure_ascii=False)
    new_lines = load_code_data(sys.argv[1])
    with open(sys.argv[2], 'w') as fw:
        fw.writelines(new_lines)