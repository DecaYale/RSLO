PROJECT_ROOT_PATH="/mnt/lustre/xuyan2/Projects/Works/RSLO/"
export PYTHONPATH="$PROJECT_ROOT_PATH:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT_PATH/rslo:$PYTHONPATH"
python -u  $PROJECT_ROOT_PATH/evaluate.py multi_proc_eval \
        --config_path $PROJECT_ROOT_PATH/config/kitti_eval_ours.prototxt \
        --model_dir ./outputs/ \
        --use_dist True \
        --gpus_per_node 1 \
        --use_apex True \
        --world_size 1 \
        --dist_port 20000 \
        --pretrained_path $PROJECT_ROOT_PATH/weights/ours.tckpt \
        --refine False \