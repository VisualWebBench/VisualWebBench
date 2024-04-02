import io
import base64

from anthropic import Anthropic, InternalServerError
from PIL import Image
from tenacity import *

from model_adapters import BaseAdapter


class ClaudeAdapter(BaseAdapter):
    def __init__(
        self,
        client: Anthropic,
        model: str,
    ):
        self.client = client
        self.model = model
        
    @retry(
        retry=retry_if_exception_type(InternalServerError), 
        stop=(stop_after_attempt(10)), 
        wait=wait_exponential(multiplier=1, min=5, max=300)
    )
    def call_llm(
        self, query, image
    ):
        response = self.client.messages.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": f"{image}",
                            },
                        },
                        {
                            "type": "text", 
                            "text": query,
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.0
        )

        outputs = response.content[0].text
        return outputs

    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        image_data = io.BytesIO()
        image.save(image_data, format="PNG")
        image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        outputs = self.call_llm(query, image)

        if task_type == "element_ocr" or task_type == "meta_generate":
            if ":" not in outputs:
                return outputs
            outputs = ":".join(outputs.split(":")[1:])
            outputs = outputs.strip().strip('"').strip("'")
            return outputs
        else:
            return outputs
