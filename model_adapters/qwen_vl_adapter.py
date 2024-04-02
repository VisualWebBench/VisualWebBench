import re

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_adapters import BaseAdapter
from utils.constants import *


class QwenVLAdapter(BaseAdapter):
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
    ):
        super().__init__(model, tokenizer)

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        # https://github.com/QwenLM/Qwen-VL/blob/master/TUTORIAL.md#grounding-capability
        image = image.convert('RGB')
        query = self.tokenizer.from_list_format([
            {'image': img_path}, # Either a local path or an url
            {'text': query},
        ])
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)

        if task_type == CAPTION_TASK:
            pattern = re.compile(r"<meta name=\"description\" content=\"(.*)\">")
            cur_meta = re.findall(pattern, response)
            if cur_meta:
                return cur_meta[0]
            else:
                return response
        elif task_type == ACTION_PREDICTION_TASK:
            return response[0].upper()
        elif task_type in [WEBQA_TASK, ELEMENT_OCR_TASK]:
            if ":" not in response:
                return response
            response = ":".join(response.split(":")[1:])
            response = response.strip().strip('"').strip("'")
            return response
        else:
            return response
        