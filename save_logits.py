import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import bmtrain as bmt
from functools import partial
import time
import os
import sys
sys.path.append("/home/wangshuo1/code/mAlign/ModelCenter")
sys.path.append("/home/wangshuo1/code/mAlign")
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer

from llama2_dataset import PromptIterableDataset, collator, load_sharegpt_data, load_sharegpt_q_switch_data, load_sharegpt_pro_data
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def get_model_tokenizer(args):
    bmt.print_rank("loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    bmt.print_rank("finished")
    bmt.print_rank("loading model...")
    model = Llama.from_pretrained(args.model_name_or_path)
    bmt.print_rank("finished")
    tokenizer.pad_token = tokenizer.eos_token
    if args.load_ckpt is not None:
        bmt.print_rank(f"loading checkpoint from {args.load_ckpt}")
        bmt.load(model, os.path.join(args.load_ckpt, "checkpoint.pt"))

    return model, tokenizer


def setup_model(args):
    model, tokenizer = get_model_tokenizer(args)
    bmt.synchronize()
    return tokenizer, model


def train(args):

    bmt.init_distributed(
        seed=args.seed,
        zero_level=3,
    )
    
    original_dataset = []
    if args.sharegpt_en_dataset is not None:
        original_dataset += load_sharegpt_pro_data(args.sharegpt_en_dataset, "en")
    bmt.print_rank("total training instance number:", len(original_dataset))

    tokenizer, model = setup_model(args)

    bmt.synchronize()
    
    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    train_start_time = time.time()
    global_step = 0

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    dataset = original_dataset

    data_per_gpu = len(dataset) // bmt.world_size()
    dataset = dataset[bmt.rank() * data_per_gpu : (bmt.rank() + 1) * data_per_gpu]


    dataset = PromptIterableDataset(dataset, tokenizer = tokenizer, max_seq_length = args.max_seq_length, teacher_forcing=True, truncate_method="tail")
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_device, collate_fn=partial(collator, tokenizer))

    for step, inputs in enumerate(dataloader):
        st = time.time()

        with bmt.inspect.inspect_tensor() as inspector:
            ids = inputs.pop("ids")
            for k in inputs:
                inputs[k] = inputs[k].cuda()
            labels = inputs.pop("labels")
            logits = model(**inputs).logits
            assert len(ids) == 1
            temp_item = {"id": ids[0], "label": labels[0].cpu(), "logits": logits[0].cpu().detach()}
            torch.save(temp_item, args.save_logits_dir + '/' + temp_item["id"] + '.pt')

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, len(tokenizer))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_func(shift_logits, shift_labels)
        
            global_loss = bmt.sum_loss(loss).item()

    
        global_step += 1
        # progress_bar.update(1)

        # record time and loss
        iteration_time = time.time() - st

        avg_time_recorder.record(iteration_time)
        avg_loss_recorder.record(global_loss)

        # print time and loss
        if global_step % args.logging_step == 0:
            bmt.print_rank(
                "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | time: {:.4f} seconds | total_time_passed: {:.4f} minutes".format(
                    global_step,
                    global_loss,
                    avg_loss_recorder.value,
                    avg_time_recorder.value,
                    (time.time() - train_start_time) / 60
                )
            )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--sharegpt_en_dataset", default=None, type=str)

    parser.add_argument("--model_name_or_path", default='/mnt/data/user/tc_agi/user/chenyulin/llama/llama-7b')
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)
    parser.add_argument("--logging_step", default=100, type=int)
    
    parser.add_argument("--save_logits_dir", type=str, required=True)

    parser.add_argument("--load_ckpt", type=str, default=None, help="resumed ckpt")

    args = parser.parse_args()

    train(args)
    args.model = args.model_name_or_path.split("/")[-1]
