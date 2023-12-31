#! /usr/bin/env python3
import argparse
import torch
from transformers.optimization import get_linear_schedule_with_warmup
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

from translation_dataset import PromptIterableDataset, collator, load_mono_data, load_para_data
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

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay, eps=1e-5, betas=(0.9, 0.95)
    )
    if args.load_ckpt is not None:
        file_name = os.path.join(args.load_ckpt, "optim.rank-{}.opt".format(bmt.rank()))
        bmt.print_rank(file_name)
        if os.path.exists(file_name):
            bmt.print_rank("start to load grad ckpt {}".format(file_name))
            states = torch.load(file_name)
            optimizer.load_state_dict(states)
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    if args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    elif args.lr_decay_style == "cosine":
        bmt.print_rank("use cosine")
        lr_scheduler = bmt.lr_scheduler.Cosine(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    elif args.lr_decay_style == "noam":
        bmt.print_rank("use noam")
        lr_scheduler = bmt.lr_scheduler.Noam(
            optimizer,
            start_lr=args.lr,
            warmup_iter=int(args.warmup_iters),
            end_iter=args.lr_decay_iters,
            num_iter=args.start_step,
        )
    else:
        raise NotImplementedError
    return lr_scheduler


def setup_model_and_optimizer(args):
    model, tokenizer = get_model_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler



def train(args):

    # print("WORLD_SIZE: " + os.environ.get("SLURM_NPROCS", "1"))
    # print("RANK: " + os.environ.get("SLURM_PROCID", "0"))
    # print("LOCAL_WORLD_SIZE: " + os.environ.get("SLURM_TASKS_PER_NODE","8")[0])
    # print("MASTER_ADDR: " + os.environ.get("MASTER_ADDR", "localhost"))
    # print("MASTER_PORT: " + os.environ.get("MASTER_PORT", "10010"))

    bmt.init_distributed(
        seed=args.seed,
        zero_level=3,
    )

    if args.wandb and bmt.rank() == 0:
        wandb.init()
    
    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)
    
    original_dataset = []
    if args.en_mono_dataset is not None:
        original_dataset += load_mono_data(args.en_mono_dataset, "en")
    if args.zh_mono_dataset is not None:
        original_dataset += load_mono_data(args.zh_mono_dataset, "zh")
    if args.en2zh_para_dataset is not None:
        original_dataset += load_para_data(args.en2zh_para_dataset, "en2zh")
    if args.zh2en_para_dataset is not None:
        original_dataset += load_para_data(args.zh2en_para_dataset, "zh2en")
    bmt.print_rank("total training instance number:", len(original_dataset))
    if args.random_instruction:
        bmt.print_rank("Random Translation Instruction")
    else:
        bmt.print_rank("Fixed Translation Instruction")
    
    args.train_iters = int((args.epochs * len(original_dataset)) / (args.batch_size_per_device * bmt.world_size())) + 1
    args.warmup_iters = int(args.train_iters * 0.04)
    bmt.print_rank("total training iterations:", args.train_iters)
    bmt.print_rank("warm-up iterations:", args.warmup_iters)

    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    bmt.synchronize()
    
    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    train_start_time = time.time()
    global_step = 0

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    
    for epoch in range(args.epochs):
        indices = torch.randperm(len(original_dataset))
        dataset = [original_dataset[i] for i in indices]

        data_per_gpu = len(dataset) // bmt.world_size()
        dataset = dataset[bmt.rank() * data_per_gpu : (bmt.rank() + 1) * data_per_gpu]


        dataset = PromptIterableDataset(dataset, tokenizer = tokenizer, max_seq_length = args.max_seq_length, teacher_forcing=True, truncate_method="tail", random_instruction=args.random_instruction)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_per_device, collate_fn=partial(collator, tokenizer))

        if global_step >= args.train_iters:
            break
        # progress_bar = tqdm(range(len(dataloader)), disable=not bmt.rank()==0, desc=f"epoch {epoch}")

        for step, inputs in enumerate(dataloader):
            if global_step < args.start_step:
                global_step += 1
                continue
            st = time.time()

            with bmt.inspect.inspect_tensor() as inspector:
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
                labels = inputs.pop("labels")
                logits = model(**inputs).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, len(tokenizer))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_func(shift_logits, shift_labels)
            
                global_loss = bmt.sum_loss(loss).item()

                optim_manager.backward(loss)

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=args.clip_grad)
                    optim_manager.step()
                    optim_manager.zero_grad()
     
            global_step += 1
            # progress_bar.update(1)

            # record time and loss
            iteration_time = time.time() - st

            avg_time_recorder.record(iteration_time)
            avg_loss_recorder.record(global_loss)

            # print time and loss
            if global_step % args.logging_step == 0:
                bmt.print_rank(
                    "| Epoch: {:3d} | Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} | time: {:.4f} seconds | total_time_passed: {:.4f} minutes".format(
                        epoch,
                        global_step,
                        global_loss,
                        avg_loss_recorder.value,
                        lr_scheduler.current_lr,
                        avg_time_recorder.value,
                        (time.time() - train_start_time) / 60
                    )
                )
                if args.wandb and bmt.rank() == 0:
                    wandb.log({
                        "loss": global_loss,
                        "average_loss": avg_loss_recorder.value,
                        "lr": lr_scheduler.current_lr,
                    }, step=global_step)
                if args.tensorboard and bmt.rank() == 0:
                    writer.add_scalar("loss/train", global_loss, global_step)
                    writer.add_scalar("average_loss/train", avg_loss_recorder.value, global_step)
                    writer.add_scalar("lr/train", lr_scheduler.current_lr, global_step)


            # save model
            # if global_step % args.save_step == 0:
            #     save_dir = os.path.join(args.save_dir, f"checkpoints/step_{global_step}")
            #     os.makedirs(save_dir, exist_ok=True)

            #     bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))
            #     print("saving optimizer state", os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank()))
            #     torch.save(optimizer.state_dict(),
            #                os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank()))

            #     if bmt.rank() == 0:
            #         # torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
            #         torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
            #         tokenizer.save_pretrained(save_dir)
            #     bmt.print_rank(f"model saved at {save_dir}")
            
            if global_step == args.train_iters:
                break
        
        # save the model at the current epoch
        save_dir = os.path.join(args.save_dir, f"checkpoints/epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))
        print("saving optimizer state", os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank()))
        torch.save(optimizer.state_dict(),
                    os.path.join(save_dir, "optim.rank-%d.opt" % bmt.rank()))

        if bmt.rank() == 0:
            # torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
            tokenizer.save_pretrained(save_dir)
        bmt.print_rank(f"model saved at {save_dir}")
    
    # save the final model
    # save_dir = os.path.join(args.save_dir, f"checkpoints/last")
    # os.makedirs(save_dir, exist_ok=True)
    # bmt.save(model, os.path.join(save_dir, "pytorch_model.pt"))
    # tokenizer.save_pretrained(save_dir)
    # bmt.print_rank(f"model saved at {save_dir}")
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--en_mono_dataset", default=None, type=str)
    parser.add_argument("--zh_mono_dataset", default=None, type=str)
    parser.add_argument("--en2zh_para_dataset", default=None, type=str)
    parser.add_argument("--zh2en_para_dataset", default=None, type=str)
    parser.add_argument("--random_instruction", default=False, action="store_true")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_name_or_path", default='/mnt/data/user/tc_agi/user/chenyulin/llama/llama-7b')
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--batch_size_per_device", default=2, type=int)
    parser.add_argument("--logging_step", default=100, type=int)
    # parser.add_argument("--save_step", default=50000, type=int)
    parser.add_argument("--data_dir", default=None, type=str)
    
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--with_eval", action="store_true")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="gradient clipping")
    # Learning rate.
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay rate")
    parser.add_argument("--loss-scale", type=float, default=6553600, help="loss scale")
    parser.add_argument("--train-iters", type=int, default=2000000)
    parser.add_argument("--save_dir", type=str, default="/data/models/chenyulin/ultrachat-llama")
    parser.add_argument("--max_sample", type=int, default=None, help="max training sample num for ultrachat")



    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument(
        "--lr-decay-style",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine", "exponential", "noam"],
        help="learning rate decay function",
    )
    parser.add_argument("--lr-decay-iters", type=int, default=None, help="lr decay steps")
    parser.add_argument(
        "--start-step", type=int, default=0, help="step to start or continue training"
    )
    parser.add_argument("--tensorboard", type=str, default=None, help="lr decay steps")
    parser.add_argument("--load_ckpt", type=str, default=None, help="resumed ckpt")


    args = parser.parse_args()

    train(args)
    args.model = args.model_name_or_path.split("/")[-1]
