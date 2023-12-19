import os
import json
from typing import *


import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
import copy
import random
import bmtrain as bmt


def load_jsonl_data(data_file):
    with open(data_file)as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines]

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

def load_alpaca_data(data_file, lang='en'):
    new_data = []
    data = json.load(open(data_file, "r"))
    for idx, item in enumerate(data):
        temp_id = f"alpaca_{lang}_{idx}"
        temp_input = (item['instruction'] + ' ' + item['input']).strip()
        temp_output = item['output'].strip()
        if temp_input == '' and temp_output == '':
            continue
        temp_data = {'id': temp_id, 'data': [temp_input, temp_output]}
        new_data.append(temp_data)
    return new_data

def load_sharegpt_data(data_file, lang='en'):
    new_data = []
    data = json.load(open(data_file, "r"))
    num_examples = len(data)
    bmt.print_rank(f"[{lang}-original data] {data_file}: {num_examples} dialogues")
    for item in data:
        conv = item["conversations"]
        if conv[0]["from"] != "human":
            conv = conv[1:]
        if conv[-1]["from"] != "gpt":
            conv = conv[:-1]
        if check_alternate_human_gpt(conv):
            temp_data = {"id": item["id"], "data": [c["value"] for c in conv]}
            new_data.append(temp_data)
    return new_data

def load_sharegpt_q_switch_data(data_file):
    new_data = []
    data = json.load(open(data_file, "r"))
    num_examples = len(data)
    bmt.print_rank(f"[question-switching data] {data_file}: {num_examples} dialogues")
    for item in data:
        conv = item["conversations"]
        if conv[0]["from"] != "human":
            conv = conv[1:]
        if conv[-1]["from"] != "gpt":
            conv = conv[:-1]
        if check_alternate_human_gpt(conv):
            for i in range(len(conv)):
                if conv[i]['from'] == 'human':
                    conv[i]['value'] = conv[i]['value'].strip() + "\n\nPlease Answer in English."
            temp_data = {"id": item["id"], "data": [c["value"] for c in conv]}
            new_data.append(temp_data)
    return new_data
    
IGNORE_INDEX=-100


def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)
    ids = [instance["id"] for instance in instances]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        ids = ids,
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 sep: List = ["EOS", "\n"],
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, bmt.print_rank("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", bmt.print_rank("only tail truncate support")
    

    
    @property
    def end_token(self):
        return self.tokenizer.eos_token

    def tokenize_example(self, example):
        # system = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
        system = "<s>[INST] "
        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            if i == 0:
                # system and 1st user message
                c_input = system + c + " [/INST]"
                tmp_tokenized_ids = self.tokenizer(c_input, add_special_tokens=False)["input_ids"]
                tokenized_ids += tmp_tokenized_ids
                labels += [IGNORE_INDEX] * len(tmp_tokenized_ids)
            elif i % 2 == 1:
                # model
                c_input = c + " </s>"
                tmp_tokenized_ids = self.tokenizer(c_input, add_special_tokens=False)["input_ids"]
                tokenized_ids += tmp_tokenized_ids
                labels += tmp_tokenized_ids
            else:
                # user
                c_input = "<s>[INST] " + c + " [/INST]"
                tmp_tokenized_ids = self.tokenizer(c_input, add_special_tokens=False)["input_ids"]
                tokenized_ids += tmp_tokenized_ids
                labels += [IGNORE_INDEX] * len(tmp_tokenized_ids)

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels), "id": example["id"]}

    def pad_truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        tokenized_example["attention_mask"] = torch.LongTensor([1]*len(tokenized_example["input_ids"]))
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                if k == "id":
                    continue
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]
        elif old_len < self.max_seq_length:
            tokenized_example["input_ids"] = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]*(self.max_seq_length - old_len)), tokenized_example["input_ids"]])
            tokenized_example["labels"] = torch.cat([torch.LongTensor([IGNORE_INDEX]*(self.max_seq_length - old_len)), tokenized_example["labels"]])
            tokenized_example["attention_mask"] = torch.LongTensor([0]*(self.max_seq_length - old_len) + [1]*old_len)
        assert len(tokenized_example["input_ids"]) == len(tokenized_example["labels"]) == len(tokenized_example["attention_mask"]) == self.max_seq_length
        return tokenized_example


    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.pad_truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)


if __name__ == "__main__":
    import sys
    qs_dataset = load_sharegpt_q_switch_data(sys.argv[1])