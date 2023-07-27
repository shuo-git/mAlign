export PATH="$PATH:/home/jeeves/.local/bin"
GPUS_PER_NODE=4
# pip install transformers==4.28.1
# pip install bmtrain==0.2.2
# pip install jieba
# pip install datasets
# pip install protobuf==3.20.0
# pip install sentencepiece
# pip install einops

echo "current path"
pwd

OPTS=""
OPTS+=" --logging_step 1" 
OPTS+=" --batch_size_per_device 16"
OPTS+=" --gradient_accumulation_steps 2"
OPTS+=" --save_step 1500"
OPTS+=" --epochs 3"
OPTS+=" --lr 2e-5"
OPTS+=" --max_seq_length 2048"
OPTS+=" --weight-decay 0.0"
OPTS+=" --train-iters 4602"
OPTS+=" --warmup_iters 184"
OPTS+=" --start-step 0"
OPTS+=" --model_name_or_path /data/llama-2-7b"
OPTS+=" --model llama_2_vicuna_7b_en"
OPTS+=" --tensorboard /data/exp/20230725_llama_2_vicuna_7b_en/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --save_dir /data/exp/20230725_llama_2_vicuna_7b_en"
OPTS+=" --sharegpt_en_dataset /data/ShareGPT/sharegpt_clean_en_fschat_split.json"
# OPTS+=" --load_ckpt /mnt/data/user/tc_agi/user/chenyulin/checkpoints/ultrachat_llama-65b-3800"

# if [ ${IDC} == klara-2-pek02 ]; then
#     CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} --master_addr=${MASTER_ENDPOINT} --master_port=${MASTER_PORT} train_llama_2.py ${OPTS}"
# else
#     CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train_llama_2.py ${OPTS}"
# fi

# echo "-------final CMD is------"
# echo "${CMD}"
# echo "-------final CMD end------"
# $CMD
torchrun --nnode=1 --nproc_per_node=${GPUS_PER_NODE} train_llama2.py ${OPTS}
