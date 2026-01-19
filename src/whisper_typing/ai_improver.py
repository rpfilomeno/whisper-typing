import google.generativeai as genai
import os

class AIImprover:
    def __init__(self, api_key: str):
        if not api_key:
            self.model = None
            print("Warning: No Gemini API key provided. AI improvement disabled.")
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            print(f"Error initializing Gemini AI: {e}")
            self.model = None

    def improve_text(self, text: str, prompt_template: str = None) -> str:
        """Improve text using Gemini AI."""
        if not self.model:
            print("Gemini AI is not configured.")
            return text

        if not text:
            return ""

        print(f"Improving text with Gemini...")
        print(f"Prompt template length: {len(prompt_template) if prompt_template else 0}")
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

            response = self.model.generate_content(prompt)
            improved_text = response.text.strip()
            print(f"Improved text: {improved_text}")
            return improved_text
        except Exception as e:
            print(f"Error during AI improvement: {e}")
            return text
