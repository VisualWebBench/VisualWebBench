model_name=llava_7b
task_type=web_caption,webqa,heading_ocr,element_ocr,element_ground,action_prediction,action_ground

python $DEBUG_MODE run.py \
    --model_name $model_name \
    --dataset_name_or_path webbench/WebBench \
    --task_type $task_type \
    --gpus 0

