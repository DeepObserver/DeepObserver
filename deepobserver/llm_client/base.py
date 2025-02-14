from abc import ABC, abstractmethod
from openai import OpenAI

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: str = None) -> str:
        pass

    @abstractmethod
    def generate_buffer(self, prompt: str, base64_images: list[bytes]) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, base64_image: bytes= None) -> str:
        prompt: str = """
            Analyze this scene for safety hazards following these steps:

            1. List all visible elements:
            - People (positions, actions)
            - Objects and equipment
            - Environmental features
            - PPE and safety equipment

            2. Identify potential hazards:
            - Immediate physical dangers
            - Equipment/machinery risks
            - Environmental hazards
            - Unsafe practices
            - Missing safety measures

            3. For each identified hazard:
            - Describe the specific risk
            - Rate severity (1-5)
            - Rate confidence in detection (1-5)
            - Explain reasoning

            4. Additional context needed:
            - List any unclear elements
            - Note viewing limitations
        """

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

    def generate_buffer(self, prompt: str, base64_images: list[bytes]) -> str:
        pass