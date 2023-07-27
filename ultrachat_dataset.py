import os
import json
from typing import *


import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
import copy
import random


def load_single_file(data_file):
    with open(data_file)as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines]

def load_raw_data(data_dir, max_sample=None, random_state=0, split=None):
    def match(f_name, split):
        if split is None:
            return True
        if isinstance(split, str):
            return split in f_name
        elif isinstance(split, list):
            for s in split:
                if s in f_name:
                    return True
        return False
    raw_dataset = []

    for f_ in os.listdir(data_dir):
        if f_.endswith("json") or f_endswith("jsonl"):
            if match(f_, split):
                f_ = os.path.join(data_dir, f_)
                print(f"load data from {f_}")
                raw_dataset += load_single_file(f_)
    if max_sample is not None:
        random.seed(random_state)
        raw_dataset = list(random.sample(raw_dataset, max_sample))
    return raw_dataset

def check_alternate_human_gpt(conv):
    length = len(conv)
    if len(conv) % 2 != 0:
        print(conv)
        return False
    tags = [i for _ in range(len(conv)//2) for i in ["human", "gpt"]]
    for i in range(len(conv)):
        if tags[i] != conv[i]["from"]:
            print(conv)
            return False
    return True

def load_sharegpt_data(data_file, lang='en'):
    new_data = []
    data = json.load(open(data_file, "r"))
    num_examples = len(data)
    print(f"[{lang}-original data] {data_file}: {num_examples} dialogues")
    for item in data:
        conv = item["conversations"]
        if conv[0]["from"] != "human":
            conv = conv[1:]
        if conv[-1]["from"] != "gpt":
            conv = conv[:-1]
        if check_alternate_human_gpt(conv):
            data = {"id": item["id"], "data": [c["value"] for c in conv]}
            new_data.append(data)
    return new_data

def load_sharegpt_q_switch_data(data_file):
    new_data = []
    data = json.load(open(data_file, "r"))
    num_examples = len(data)
    print(f"[question-switching data] {data_file}: {num_examples} dialogues")
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
            data = {"id": item["id"]+'_q_switch', "data": [c["value"] for c in conv]}
            new_data.append(data)
    return new_data

    
IGNORE_INDEX=-100


def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
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
        assert self.teacher_forcing, print("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", print("only tail truncate support")
    

    
    @property
    def end_token(self):
        if self._end_token is not None:
            return self._end_token
        end_token = self.sep[0]
        if end_token == "EOS":
            self._end_token = self.tokenizer.eos_token
        else:
            self._end_token = end_token
        return self._end_token

    def tokenize_example(self, example):
        end_token = self.end_token
        tags = [i for _ in range(len(example["data"])//2) for i in ["User", "Assistant"]]
        if example["id"].startswith("reasoning-"):
            assert len(example["data"]) == 2, print(example)
            tags = ["Question", "Answer"]
        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            if i % 2 == 1:
                # model
                c_input = self.start_token + tags[i] + ": "
                tokenized = self.tokenizer(c_input, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                c_generate = c + end_token
                tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]
            else:
                # user
                if i == 0:
                    # no start token
                    c_new = self.tokenizer.bos_token + tags[i] + ": " + c
                else:
                    c_new = self.start_token + tags[i] + ": " + c
                tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}

    def pad_truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        tokenized_example["attention_mask"] = torch.LongTensor([1]*len(tokenized_example["input_ids"]))
        if old_len > self.max_seq_length:
            for k in tokenized_example:
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