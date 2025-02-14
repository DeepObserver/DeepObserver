from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: str = None) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.previous_summary = None
        self.name = "GPT-4V"

    def generate(self, prompt: str, base64_image: bytes = None) -> str:
        system_message = {
            "role": "system",
            "content": "You are a computer vision AI analyzing security camera footage. "
                      "Provide detailed analysis with numerical scoring in these categories:\n"
                      "- Activity Level (0-10): Amount of motion and activity in the scene\n"
                      "- Object Density (0-10): Number and complexity of objects present\n"
                      "- Environmental Change (0-10): Changes in lighting, setting, or background\n"
                      "- Human Presence (0-10): Number and involvement of people in the scene\n"
                      "If the scene is unchanged from previous observation, respond with exactly 'situation unchanged'."
        }
        
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Previous observation: {self.previous_summary if self.previous_summary else 'None'}\n\n"
                           "If the scene has changed, provide:\n"
                           "1. Brief situation summary\n"
                           "2. Scores:\n"
                           "   - Activity Level: [0-10]\n"
                           "   - Object Density: [0-10]\n"
                           "   - Environmental Change: [0-10]\n"
                           "   - Human Presence: [0-10]\n"
                           "3. Key Observations\n\n"
                           "If the scene is essentially unchanged, respond with exactly: 'situation unchanged'",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            ]
        }

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message],
            max_tokens=600
        )
        
        result = response.choices[0].message.content
        if result != "situation unchanged":
            self.previous_summary = result
        
        return result

class ClaudeClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Anthropic(api_key=self.api_key)
        self.previous_summary = None
        self.name = "Claude 3"

    def generate(self, prompt: str, base64_image: bytes = None) -> str:
        system_prompt = ("You are a computer vision AI analyzing security camera footage. "
                      "Provide detailed analysis with numerical scoring in these categories:\n"
                      "- Activity Level (0-10): Amount of motion and activity in the scene\n"
                      "- Object Density (0-10): Number and complexity of objects present\n"
                      "- Environmental Change (0-10): Changes in lighting, setting, or background\n"
                      "- Human Presence (0-10): Number and involvement of people in the scene\n"
                      "If the scene is unchanged from previous observation, respond with exactly 'situation unchanged'.")

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=600,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Previous observation: {self.previous_summary if self.previous_summary else 'None'}\n\n"
                                   "If the scene has changed, provide:\n"
                                   "1. Brief situation summary\n"
                                   "2. Scores:\n"
                                   "   - Activity Level: [0-10]\n"
                                   "   - Object Density: [0-10]\n"
                                   "   - Environmental Change: [0-10]\n"
                                   "   - Human Presence: [0-10]\n"
                                   "3. Key Observations\n\n"
                                   "If the scene is essentially unchanged, respond with exactly: 'situation unchanged'"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        result = response.content[0].text
        if result != "situation unchanged":
            self.previous_summary = result
        
        return result