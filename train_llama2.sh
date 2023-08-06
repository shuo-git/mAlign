#!/bin/bash
#SBATCH --partition=gpu3
#SBATCH --exclude=g3003
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512GB
hostname
export OMP_NUM_THREADS=1
export MASTER_PORT=34567
echo "SLURM_JOB_NODELIST"=${SLURM_JOB_NODELIST}
echo "SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES}
echo "SLURM_NODEID="${SLURM_NODEID}
echo "JOBID="${SLURM_JOBID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

SIZE="7b"
BSZ=1
EXP=/home/wangshuo1/exp/Llama-2-${SIZE}-qs-zh-test
DATA=/home/wangshuo1/datasets/ShareGPT
mkdir -p $EXP
OPTS=""
OPTS+=" --logging_step 1" 
OPTS+=" --batch_size_per_device $BSZ"
OPTS+=" --gradient_accumulation_steps 1"
OPTS+=" --epochs 3"
OPTS+=" --lr 2e-5"
OPTS+=" --max_seq_length 4096"
OPTS+=" --weight-decay 0.1"
OPTS+=" --start-step 0"
OPTS+=" --model_name_or_path /home/wangshuo1/opensource_models/meta-llama/Llama-2-${SIZE}-mc"
OPTS+=" --tensorboard ${EXP}/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --save_dir ${EXP}"
OPTS+=" --sharegpt_zh_dataset ${DATA}/sharegpt_clean_zh_fschat_4k_split.json"
# OPTS+=" --sharegpt_q_switch_dataset ${DATA}/sharegpt_clean_en_fschat_q_switch_zh_4k_split.json"

# slurm multi-node CMD
CMD="python train_llama2.py ${OPTS}"
# hard-coded multi-node CMD
# CMD="torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=g3001:12345 train_llama2.py ${OPTS}"
# single-node CMD
# CMD="torchrun --nnodes=1 --nproc_per_node=8 train_llama2.py ${OPTS}"

echo "-------final CMD is ------"
echo "${CMD}"
echo "-------final CMD end------"
# srun date
$CMD
