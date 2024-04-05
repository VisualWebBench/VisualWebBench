model_name=openai
task_type=web_caption,head_ocr,webqa,element_ocr,action_prediction,element_ground,action_ground

python $DEBUG_MODE run.py \
    --model_name $model_name \
    --dataset_name_or_path webbench/WebBench \
    --task_type $task_type \
    --gpus 0

