import google.generativeai as genai
from PIL import Image
from tenacity import *

from model_adapters import BaseAdapter
from utils.constants import *


class GeminiAdapter(BaseAdapter):
    def __init__(
        self,
        model: genai.GenerativeModel,
    ):
        self.model = model
        
    @retry(
        stop=(stop_after_attempt(10)), 
        wait=wait_exponential(multiplier=1, min=5, max=300)
    )
    def call_llm(
        self, query, image
    ):
        outputs = self.model.generate_content([query, image])
        return outputs.text

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        try:
            outputs = self.call_llm(query, image)
        except:
            print("error")
            outputs = ""

        if task_type == ELEMENT_OCR_TASK or task_type == CAPTION_TASK:
            if ":" not in outputs:
                return outputs
            outputs = ":".join(outputs.split(":")[1:])
            outputs = outputs.strip().strip('"').strip("'")
            return outputs
        else:
            return outputs
