import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from PIL import Image
from datasets import Dataset, Image


# id
# task_type
# website
# image
# options
# bbox
# bbox_desc
# answer
def load_dataset(raw_path, task_type, type2path, debug=False):
    assert task_type in [
        'meta_generate', 
        'web_caption', 
        'element_ocr', 
        'action_prediction', 
        'element_ground', 
        'action_ground', 
        'webqa', 
        'element_ground_bbox', 
        'action_ground_bbox',
        'element_ground_point',
        'action_ground_point',
    ]
    
    with open(os.path.join(raw_path, type2path[task_type], 'input_data.jsonl')) as fr:
        raw = fr.readlines()

    if debug:
        raw = raw[:3]

    res = []
    for idx, item in enumerate(raw, 1):
        item = json.loads(item)
        cur = {
            "id": f"{task_type}_{idx}",
            "task_type": task_type,
            "website": item['website'],
            "image": item.get('img_path', None),
            "image_size": item.get('image_size', None),
            "options": None,
            "bbox": None,
            "bbox_caption": None,
            "answer": None
        }
        if task_type == 'meta_generate':
            cur['answer'] = item['choices'][item['gt']]
        elif task_type == "web_caption":
            cur['answer'] = item['gt']
        elif task_type == "element_ocr":
            # cur['bbox'] = item['bbox_ratio']
            cur['bbox'] = str([round(t, 3) for t in item['bbox_ratio']])
            cur['bbox_caption'] = item['elem_desc']
            cur['answer'] = cur['bbox_caption']
        elif task_type == "action_prediction":
            cur['options'] = '\n'.join([f"{chr(ord('A')+ii)}. {t}" for ii, t in enumerate(item['choices'])])
            cur['bbox'] = str([round(t, 3) for t in item['bbox_ratio']])
            cur['bbox_caption'] = item['elem_desc']
            cur['answer'] = item['gt']
        elif task_type == "element_ground":
            cur['image'] = item['choice_img_path']
            cur['bbox_caption'] = item['elem_desc']
            cur['answer'] = ord(item['gt_symbol']) - ord('A')
        elif task_type in ["element_ground_bbox", "element_ground_point"]:
            w = item['image_size'][0]
            h = item['image_size'][1]
            cur['bbox'] = [item['bbox_ratio'][0]*w, item['bbox_ratio'][1]*h, item['bbox_ratio'][2]*w, item['bbox_ratio'][3]*h]
            cur['bbox_caption'] = item['elem_desc']
            cur['answer'] = cur['bbox']
        elif task_type == "action_ground":
            cur['image'] = item['choice_img_path']
            cur['bbox_caption'] = item['instruction']
            cur['answer'] = ord(item['gt_symbol']) - ord('A')
        elif task_type in ["action_ground_bbox", "action_ground_point"]:
            w = item['image_size'][0]
            h = item['image_size'][1]
            cur['bbox'] = [item['bbox_ratio'][0]*w, item['bbox_ratio'][1]*h, item['bbox_ratio'][2]*w, item['bbox_ratio'][3]*h]
            cur['bbox_caption'] = item['instruction']
            cur['answer'] = cur['bbox']
        elif task_type == "webqa":
            cur['question'] = item['question']
            cur['answer'] = item['answers']
        else:
            raise NotImplementedError
        res.append(cur)

    # dataset = Dataset.from_list(res).cast_column("image", Image())
    dataset = Dataset.from_list(res)
    return dataset
