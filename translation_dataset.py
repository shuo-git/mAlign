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
import random


def load_mono_data(data_file, lang='en'):
    new_data = []
    with open(data_file, 'r') as fr:
        lines = fr.readlines()
    num_examples = len(lines)
    bmt.print_rank(f"[{lang} monolingual data] {data_file}: {num_examples} lines")
    for idx, l in enumerate(lines):
        temp_id = f"{data_file}_{idx}"
        data = {"id": temp_id, "data": [l.strip()]}
        new_data.append(data)
    return new_data


def load_para_data(data_file, direction='en2zh'):
    new_data = []
    with open(data_file, 'r') as fr:
        lines = fr.readlines()
        json_lines = [json.loads(l.strip()) for l in lines]
    num_examples = len(json_lines)
    bmt.print_rank(f"[{direction} parallel data] {data_file}: {num_examples} pairs")
    for idx, l in enumerate(json_lines):
        temp_id = f"{data_file}_{idx}"
        data = {"id": temp_id, "data": [l['src'], l['tgt']]}
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
                 random_instruction: Optional[bool] = False,
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        self.random_instruction = random_instruction
        assert self.teacher_forcing, bmt.print_rank("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        assert self.truncate_method == "tail", bmt.print_rank("only tail truncate support")
    

    
    @property
    def end_token(self):
        return self.tokenizer.eos_token

    def tokenize_example(self, example):
        if len(example["data"]) == 1:
            c_input = "<s>" + example["data"][0] + "</s>"
        else:
            src_line = example["data"][0]
            tgt_line = example["data"][1]
            if self.random_instruction:
                instructions = [
                    "Please translate the sentences below into Chinese.",
                    "I need you to provide the Chinese translation for the following sentences.",
                    "Your task is to render the sentences below into Chinese.",
                    "Can you convert the following sentences into Chinese for me?",
                    "Translate the sentences written below into Chinese, please.",
                    "I'd like you to translate the following sentences into Chinese.",
                    "Your assignment is to translate the sentences below into Chinese.",
                    "Please convert the sentences below into Chinese.",
                    "Could you translate the following sentences into Chinese, please?",
                    "Your job is to render the following sentences into Chinese."
                ]
                random.shuffle(instructions)
                instruction = instructions[0]
            else:
                instruction = "[METAINST] Translate into Chinese [/METAINST]"
            c_input = f"<s>[INST] {instruction} {src_line} [/INST] {tgt_line} </s>"
        
        tokenized_ids = self.tokenizer(c_input, add_special_tokens=False)["input_ids"]
        labels = tokenized_ids

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
    pass