import io
import re
import base64

from openai import OpenAI
from PIL import Image

from model_adapters import BaseAdapter
from utils.constants import *


class OpenAIAdapter(BaseAdapter):
    def __init__(
        self, 
        client: OpenAI,
        model: str,
    ):
        self.client = client
        self.model = model

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        image_data = io.BytesIO()
        image.save(image_data, format="PNG")
        image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{image}"
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.0
            )

            outputs = response.choices[0].message.content
        except:
            outputs = ""

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
