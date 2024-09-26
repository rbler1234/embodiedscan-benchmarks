export WANDB_MODE=offline
export MASTER_PORT=9604
# accelerate
python launch.py --name leo_tuning \
                 --mode accelerate \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --time 48 \
                 --config configs/default_debug_zero.yaml \
                 --port 2050 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition HGX \
                 task=tuning_noact \
                 note=tuning_noact \
                 pretrained_ckpt_path=/mnt/petrelfs/linjingli/mmscan_modelzoo-main/llmzoo/LEO_new/weights/yangshuais/best.pth \
