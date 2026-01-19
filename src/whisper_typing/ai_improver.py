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

    def improve_text(self, text: str) -> str:
        """Improve text using Gemini AI."""
        if not self.model:
            print("Gemini AI is not configured.")
            return text

        if not text:
            return ""

        print(f"Improving text with Gemini...")
        try:
            prompt = (
                "Refine and correct the following transcribed text. "
                "Maintain the original meaning but improve grammar, punctuation and clarity. "
                "Output ONLY the refined text, nothing else.\n\n"
                f"Text: {text}"
            )
            response = self.model.generate_content(prompt)
            improved_text = response.text.strip()
            print(f"Improved text: {improved_text}")
            return improved_text
        except Exception as e:
            print(f"Error during AI improvement: {e}")
            return text
