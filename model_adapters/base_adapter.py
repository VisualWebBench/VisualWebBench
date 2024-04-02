from abc import ABC, abstractmethod

from PIL import Image


class BaseAdapter(ABC):
    def __init__(
        self, 
        model=None, 
        tokenizer=None, 
    ):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(
        self,
        query: str,
        image: Image,
        task_type: str,
    ) -> str:
        pass

    def __call__(
        self,
        query: str,
        image: str,
        task_type: str,
    ) -> str:
        return self.generate(query, image, task_type)