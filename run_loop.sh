# DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

task_types=(meta_generate web_caption webqa element_ocr action_prediction)
models=(llava_13b llava_34b)

for model in ${models[@]}; do
    for task in ${task_types[@]}; do
        python $DEBUG_MODE run.py \
            --model_name $model \
            --task_type $task \
            --data_path output_v7 \
            --gpus 3
    done
done

task_types=(element_ground action_ground)

for model in ${models[@]}; do
    for task in ${task_types[@]}; do
        python $DEBUG_MODE run.py \
            --model_name $model \
            --task_type $task \
            --data_path output_v8 \
            --gpus 3
    done
done