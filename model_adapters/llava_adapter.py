import re
import torch
from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)

from model_adapters import BaseAdapter
from utils.constants import *

class LlavaAdapter(BaseAdapter):
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        context_len: int,
        image_processor,
        conv_mode,
    ):
        super().__init__(model, tokenizer)
        self.context_len = context_len
        self.image_processor = image_processor
        self.conv_mode = conv_mode

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = [image.convert('RGB')]
        image_sizes = [x.size for x in images]

        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        if task_type == CAPTION_TASK:
            pattern = re.compile(r"<meta name=\"description\" content=\"(.*)\">")
            cur_meta = re.findall(pattern, outputs)
            if cur_meta:
                return cur_meta[0]
            else:
                return outputs
        elif task_type == ELEMENT_OCR_TASK:
            if ":" not in outputs:
                return outputs
            outputs = ":".join(outputs.split(":")[1:])
            outputs = outputs.strip().strip('"').strip("'")
            return outputs
        else:
            return outputs
