DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

task_type=web_caption,head_ocr,webqa,element_ocr,action_prediction,element_ground,action_ground
# task_type=web_caption,webqa,element_ocr,action_prediction

python $DEBUG_MODE run.py \
    --model_name openai \
    --dataset_name_or_path /ML-A800/home/yifan/data/WebBench \
    --task_type $task_type \
    --gpus 1

