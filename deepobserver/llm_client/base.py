from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: str = None) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, image: str = None) -> str:
        pass