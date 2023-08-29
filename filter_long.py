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
    en_dataset = load_sharegpt_data("sharegpt_clean_en_fschat_common.json")
    # en_dataset = load_sharegpt_data("test_512.json")
    en_ids = []
    for data in en_dataset:
        assert data["id"] not in en_ids
        en_ids.append(data["id"])
    
    qs_dataset = load_sharegpt_q_switch_data("sharegpt_clean_en_fschat_q_switch_zh.json")
    # qs_dataset = load_sharegpt_q_switch_data("test_512.json")
    qs_ids = []
    for data in qs_dataset:
        assert data["id"] not in qs_ids
        qs_ids.append(data["id"])
    
    shared_ids = []
    for eid in en_ids:
        if eid in qs_ids:
            shared_ids.append(eid)
    
    shared_en_dataset = []
    shared_qs_dataset = []
    for sid in shared_ids:
        en_item = get_by_id(en_dataset, sid)
        qs_item = get_by_id(qs_dataset, sid)
        assert len(en_item['data']) == len(qs_item['data'])
        for i in range(1, len(en_item['data']), 2):
            assert en_item['data'][i] == qs_item['data'][i]
        shared_en_dataset.append(en_item)
        shared_qs_dataset.append(qs_item)
    
    with open("/data/public/multilingual/exp_20230810/ShareGPT/sharegpt_clean_en_fschat_common_4k_filter_share.json", "w") as fw:
        json.dump(shared_en_dataset, fw, ensure_ascii=False, indent=2)
    
    with open("/data/public/multilingual/exp_20230810/ShareGPT/sharegpt_clean_en_fschat_q_switch_zh_4k_filter_share.json", "w") as fw:
        json.dump(shared_qs_dataset, fw, ensure_ascii=False, indent=2)