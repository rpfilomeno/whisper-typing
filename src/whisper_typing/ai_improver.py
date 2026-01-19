from google import genai
import os
from typing import Optional, Callable

class AIImprover:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", debug: bool = False, logger: Optional[Callable[[str], None]] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.debug = debug
        self.logger = logger
        
        if not api_key:
            self.client = None
            self.log("Warning: No Gemini API key provided. AI improvement disabled.")
            return

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            self.log(f"Error initializing Gemini AI: {e}")
            self.client = None

    def log(self, message: str):
        if self.logger:
            self.logger(message)
        else:
            print(message)

    @staticmethod
    def list_models(api_key: str):
        """List available Gemini models that support content generation."""
        try:
            client = genai.Client(api_key=api_key)
            models = []
            for m in client.models.list():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name)
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def improve_text(self, text: str, prompt_template: str = None) -> str:
        """Improve text using Gemini AI."""
        if not self.client:
            self.log("Gemini AI is not configured.")
            return text

        if not text:
            return ""

        # Remove 'models/' prefix if present
        model_id = self.model_name
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]

        if self.debug:
            self.log(f"DEBUG: Using Gemini model ID: {model_id}")

        try:
            if not prompt_template:
                prompt = (
                    "Refine and correct the following transcribed text. "
                    "Maintain the original meaning but improve grammar, punctuation and clarity. "
                    "Output ONLY the refined text, nothing else.\n\n"
                    f"Text: {text}"
                )
            else:
                # Use custom prompt, replacing {text} placeholder
                prompt = prompt_template.replace("{text}", text)

            if self.debug:
                self.log(f"DEBUG: Gemini raw request prompt:\n{prompt}")

            response = self.client.models.generate_content(
                model=model_id,
                contents=prompt
            )
            improved_text = response.text.strip()
            
            if self.debug:
                self.log(f"DEBUG: Gemini raw response:\n{improved_text}")
                
            return improved_text
        except Exception as e:
            self.log(f"Error during AI improvement: {e}")
            return text
