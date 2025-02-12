from abc import ABC, abstractmethod
from openai import OpenAI

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: str = None) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, base64_image: bytes= None) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is an image of a security camera scene. Describe the scene in detail.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ]
                }
            ]
        )
        return response.choices[0].message.content