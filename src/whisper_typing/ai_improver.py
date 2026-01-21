"""AI text improvement using Gemini."""

from collections.abc import Callable

from google import genai
from google.api_core import exceptions


class AIImprover:
    """Improves transcribed text using Google's Gemini AI."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str = "gemini-1.5-flash",
        *,
        debug: bool = False,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the AIImprover.

        Args:
            api_key: Google Gemini API key.
            model_name: Name of the Gemini model to use.
            debug: Whether to enable debug logging.
            logger: Optional callback for logging messages.

        """
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
        except Exception as e:  # noqa: BLE001
            self.log(f"Error initializing Gemini AI: {e}")
            self.client = None

    def log(self, message: str) -> None:
        """Log a message using the configured logger.

        Args:
            message: The message to log.

        """
        if self.logger:
            self.logger(message)

    @staticmethod
    def list_models(api_key: str | None) -> list[str]:
        """List available Gemini models that support content generation.

        Args:
            api_key: Google Gemini API key.

        Returns:
            A list of supported model names.

        """
        if not api_key:
            return []
        try:
            client = genai.Client(api_key=api_key)
            return [
                m.name
                for m in client.models.list()
                if "generateContent" in m.supported_actions
            ]
        except Exception:  # noqa: BLE001
            return []

    def improve_text(self, text: str, prompt_template: str | None = None) -> str:
        """Improve text using Gemini AI.

        Args:
            text: The text to improve.
            prompt_template: Optional template for the improvement prompt.

        Returns:
            The improved text, or original text if improvement fails.

        """
        if not self.client:
            self.log("Gemini AI is not configured.")
            return text

        if not text:
            return ""

        # Remove 'models/' prefix if present
        model_id = self.model_name
        model_id = model_id.removeprefix("models/")

        if self.debug:
            self.log(f"DEBUG: Using Gemini model ID: {model_id}")

        try:
            if not prompt_template:
                prompt = (
                    "Refine and correct the following transcribed text. "
                    "Maintain the original meaning but improve grammar, "
                    "punctuation and clarity. "
                    "Output ONLY the refined text, nothing else.\n\n"
                    f"Text: {text}"
                )
            else:
                # Use custom prompt, replacing {text} placeholder
                prompt = prompt_template.replace("{text}", text)

            if self.debug:
                self.log(f"DEBUG: Gemini raw request prompt:\n{prompt}")

            response = self.client.models.generate_content(
                model=model_id, contents=prompt
            )
        except exceptions.ResourceExhausted as e:
            self.log(f"You have exceeded your Gemini API usage quota: {e}")
            return text
        except Exception as e:  # noqa: BLE001
            self.log(f"Error during AI improvement: {e}")
            return text
        else:
            improved_text = response.text.strip()
            if self.debug:
                self.log(f"DEBUG: Gemini raw response:\n{improved_text}")
            return improved_text
