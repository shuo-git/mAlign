MASTER_ADDR=`hostname `
MASTER_PORT=12345
echo $MASTER_ADDR
export OMP_NUM_THREADS=1
CODE=/data/public/multilingual/whq/mAlign
SIZE="7b"
BSZ=4
FREQ=1
EXP=/data/public/multilingual/whq/mAlign/result/
DATA=/home/wanghanqing/projects/exp/data/MetaMath
system_prompts=("以下是描述一个任务的指令。请写出一个适当的回复来完成这个请求。" "Below-is-an-instruction-that-describes-a-task.-Write-a-response-that-appropriately-completes-the-request.")
mkdir -p $EXP
OPTS=""
OPTS+=" --logging_step 1"
OPTS+=" --batch_size_per_device $BSZ"
OPTS+=" --gradient_accumulation_steps $FREQ"
OPTS+=" --epochs 1"
OPTS+=" --lr 1e-4"
OPTS+=" --max_seq_length 4096"
OPTS+=" --weight-decay 0.1"
OPTS+=" --start-step 0"
OPTS+=" --model_name_or_path /data/public/opensource_models/meta-llama/Llama-2-${SIZE}-mc"
OPTS+=" --tensorboard ${EXP}/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --save_step 1000"
OPTS+=" --save_dir ${EXP}"
# OPTS+=" --metamath_dataset ${DATA}/MetaMathQA-395K.json"
# OPTS+=" --system_prompt Below-is-an-instruction-that-describes-a-task.-Write-a-response-that-appropriately-completes-the-request."
OPTS+=" --system_prompt ${system_prompts[1]}"
OPTS+=" --lora_list zh,math"
OPTS+=" --lora_root_path /home/wanghanqing/projects/exp/mAlign_exp/mAlign_LoRAs"
OPTS+=" --few_shot_zh_math_dataset /home/wanghanqing/projects/exp/data/fusion_few_shot/zh_meta_math_few_shot.jsonl"
# OPTS+=" --alpaca_dataset ${DATA}/en.json"
# OPTS+=" --sharegpt_dataset ${DATA}/sharegpt_clean_en_fschat_4k_split.json"

CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node=1 $CODE/train_llama2.py ${OPTS} &> test.log