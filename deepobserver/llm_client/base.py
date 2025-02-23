from abc import ABC, abstractmethod
from openai import OpenAI
import requests
from typing import Optional
import time
import os

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

# Add new Ollama client
class OllamaClient(LLMClient):
    def __init__(self, model_name: str = "llava"):
        self.model = model_name
        self.base_url = "http://localhost:11434/api"
        self.session_start = time.strftime("%Y%m%d_%H%M%S")
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        self.log_file = f"logs/ollama_observations_{self.session_start}.txt"
        
        # Create log file with header
        with open(self.log_file, 'w') as f:
            f.write(f"Ollama {model_name} Observations\n")
            f.write(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")

    def log_observation(self, observation: str, timestamp: str = None):
        """Log an observation to the session file"""
        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")
            
        with open(self.log_file, 'a') as f:
            f.write(f"\n[{timestamp}]\n")
            f.write(observation)
            f.write("\n" + "-" * 50 + "\n")

    def _make_api_call(self, prompt: str, base64_image: Optional[str] = None) -> str:
        """Make API call to Ollama"""
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 128,
            }
        }
        
        # Format image data correctly for Ollama
        if base64_image:
            # Remove any data:image prefix if present
            if 'base64,' in base64_image:
                base64_image = base64_image.split('base64,')[1]
            payload["images"] = [base64_image]

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['response']

    def generate(self, prompt: str, base64_image: Optional[str] = None) -> str:
        """Generate response using Ollama API"""
        try:
            response = self._make_api_call(prompt, base64_image)
            # Log the observation
            self.log_observation(response)
            return response
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API error: {e}"
            self.log_observation(f"ERROR: {error_msg}")
            return error_msg

    def generate_buffer(self, prompt: str, base64_images: list[bytes]) -> str:
        """Process multiple images through Ollama"""
        # LLaVA can only process one image at a time
        if not base64_images:
            return "No images provided"
            
        # Use the first frame for analysis
        try:
            return self.generate(prompt, base64_images[0])
        except Exception as e:
            print(f"Error processing image buffer: {e}")
            return f"Error processing image buffer: {e}"