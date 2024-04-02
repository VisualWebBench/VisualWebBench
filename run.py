import os
import json
import yaml
import argparse
from tqdm import tqdm

import datasets
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import model_adapters
from utils import DEFAULT_PROMPTS
from utils import (
    eval_web_caption,
    eval_head_ocr,
    eval_element_ocr,
    eval_action_prediction,
    eval_element_ground,
    eval_action_ground,
    eval_webqa,
)
from utils.constants import *


eval_metric = {
    CAPTION_TASK: eval_web_caption,
    HEAD_OCR_TASK: eval_head_ocr,
    WEBQA_TASK:eval_webqa,
    ELEMENT_OCR_TASK: eval_element_ocr,
    ELEMENT_GROUND_TASK: eval_element_ground,
    ACTION_PREDICTION_TASK: eval_action_prediction,
    ACTION_GROUND_TASK: eval_action_ground,
}

type2path = {
    CAPTION_TASK: 'meta_generate',
    HEAD_OCR_TASK: 'title_identification',
    WEBQA_TASK: "webqa",
    ELEMENT_OCR_TASK: "long_text_OCR",
    ELEMENT_GROUND_TASK: "element_ground",
    ACTION_PREDICTION_TASK: "action_prediction",
    ACTION_GROUND_TASK: "action_grounding",
}


def evaluate(
    model_adapter: model_adapters.BaseAdapter,
    prompt: str,
    dataset: datasets.Dataset,
    task_type: str,
    **kwargs,
):
    preds, golds = [], []
    print('='*80)
    print('Prompt: ', prompt)
    data_size = len(dataset)
    for idx_ in tqdm(range(data_size), desc=task_type):
        sample = dataset[idx_]
        
        if task_type in [CAPTION_TASK, HEAD_OCR_TASK]:
            cur_prompt = prompt
        elif task_type == WEBQA_TASK:
            cur_prompt = prompt.format(question=sample['question'])
        elif task_type == ELEMENT_OCR_TASK:
            cur_prompt = prompt.format(bbox_ratio=sample['bbox'])
        elif task_type == ELEMENT_GROUND_TASK:
            cur_prompt = prompt.format(element_desc=sample['elem_desc'])
        elif task_type == ACTION_PREDICTION_TASK:
            cur_prompt = prompt.format(bbox_ratio=sample['bbox'], choices_text=sample['options'])
        elif task_type == ACTION_GROUND_TASK:
            cur_prompt = prompt.format(instruction=sample['instruction'])
        else:
            raise NotImplementedError(f"Task type {task_type} not implemented.")
            
        response = model_adapter(cur_prompt, sample['image'], task_type=task_type)
        
        preds.append(response)
        golds.append(sample['answer'])

    scores = eval_metric[task_type](preds, golds)
    return scores, preds, golds


def main(args):
    model_config = yaml.load(open(f"configs/{args.model_name}.yaml"), Loader=yaml.FullLoader)
    model_path = model_config.get('model_path')
    tokenizer_path = model_config.get('tokenizer_path', model_path)
    
    device = f"cuda:{args.gpus}"

    model_name = model_path.split("/")[-1].lower()

    if "gpt" in model_name:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(
            client, model_path,
        )
    elif "gemini" in model_name:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model_path)
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(model)
    elif "claude" in model_name:
        from anthropic import Anthropic

        client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(
            client, model_path,
        )
    elif "llava" in model_name:
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import (
            get_model_name_from_path,
        )

        raw_model_name = get_model_name_from_path(model_path)
        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, raw_model_name, device_map=None, device=device,
        )
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(
            model, tokenizer, context_len, image_processor, model_config['conv_mode']
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(model, tokenizer)

    if ',' in args.task_type:
        task_types = [item.strip() for item in args.task_type.split(',')]
    else:
        task_types = [args.task_type]

    for task_type in task_types:
            # with model
        print(model_config.keys())
        prompt = model_config.get(f"{task_type}_prompt", DEFAULT_PROMPTS[f"{task_type}_prompt"])
        
        dataset = datasets.load_dataset(args.dataset_name_or_path, task_type)['test']
        
        scores, preds, golds = evaluate(
            model_adapter=model_adapter,
            prompt=prompt,
            dataset=dataset,
            task_type=task_type,
        )
        score_str = ', '.join([f"{k}: {v:.2f}" for k, v in scores.items()])
        print(f"Model: {args.model_name}, Task: {task_type}, Scores: {score_str}")

        output_res = [
            {
                "pred": pred,
                "gold": gold,
            } for pred, gold in zip(preds, golds)
        ]
        output_res = [{"score": score_str}] + output_res
        with open(os.path.join(args.output_path, f"{task_type}{'_cot' if args.cot else ''}.json"), "w") as f:
            json.dump(output_res, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name_or_path',
        default="webbench/WebBench",
        type=str,
    )
    parser.add_argument(
        '--model_name',
        default='qwen_vl',
        type=str,
        choices=[file.split(".")[0] for file in os.listdir("configs") if file.endswith(".yaml")],
    )
    parser.add_argument(
        '--task_type',
        default='meta_generate',
        type=str,
    )
    parser.add_argument(
        '--cot',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--output_path', 
        default='output', 
        type=str
    )
    parser.add_argument(
        "--gpus",
        default="0",
        type=str,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.output_path = os.path.join(args.output_path, args.model_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)

    main(args)
