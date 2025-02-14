from abc import ABC, abstractmethod
from openai import OpenAI

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_buffer(self, prompt: str, base64_images: list[bytes]) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }]
        )
        return response.choices[0].message.content

    def generate_buffer(self, prompt: str, base64_images: list[bytes]) -> str:
        content = [{"type": "text", "text": prompt}]
        for base64_image in base64_images:
            content.append(self.prepare_image_message(base64_image))

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        return response.choices[0].message.content

    def prepare_image_message(self, base64_image: bytes) -> dict:
        """
        Prepare the image message in the format expected by the API

        Args:
            base64_image: Base64 encoded image bytes
        """
        mime_type = "image/jpeg"  # Assuming JPEG format based on processor.py context

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        }