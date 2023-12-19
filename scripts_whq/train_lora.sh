

MASTER_ADDR=`hostname `
MASTER_PORT=12345
echo $MASTER_ADDR
export OMP_NUM_THREADS=1
CODE=/home/wanghanqing/projects/mAlign
SIZE="7b"
BSZ=8
FREQ=1
EXP=/home/wanghanqing/projects/exp/mAlign_exp/LoRAs/en
DATA=/home/wanghanqing/projects/exp/mAlign_exp/data/en
mkdir -p $EXP
OPTS=""
OPTS+=" --logging_step 1"
OPTS+=" --batch_size_per_device $BSZ"
OPTS+=" --gradient_accumulation_steps $FREQ"
OPTS+=" --epochs 3"
OPTS+=" --lr 1e-4"
OPTS+=" --max_seq_length 4096"
OPTS+=" --weight-decay 0.1"
OPTS+=" --start-step 0"
OPTS+=" --model_name_or_path /data/public/opensource_models/meta-llama/Llama-2-${SIZE}-mc"
OPTS+=" --tensorboard ${EXP}/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --save_dir ${EXP}"
# OPTS+=" --sharegpt_en_dataset ${DATA}/sharegpt_clean_en_fschat_common_4k_split.json"
OPTS+=" --alpaca_dataset ${DATA}/en.json"
OPTS+=" --sharegpt_dataset ${DATA}/sharegpt_clean_en_fschat_4k_split.json"

torchrun --nnodes=1 --nproc_per_node=8 $CODE/train_llama2.py ${OPTS}
