import numpy as np
import torch
from transformers import pipeline


class Transcriber:
    def __init__(self, model_id="openai/whisper-base", language=None):
        self.model_id = model_id
        self.language = language
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading model {model_id} on {device} with {torch_dtype}...")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, audio_input) -> str:
        """Transcribe audio (file path or numpy array) to text."""
        input_type = "buffer" if hasattr(audio_input, 'shape') else audio_input
        print(f"Transcribing {input_type}...")
        
        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language

        result = self.pipe(
            audio_input,
            chunk_length_s=30,
            batch_size=8, # increased batch size for better GPU utilization
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
            ignore_warning=True
        )
        
        text = result["text"].strip()
        print(f"Transcription: {text}")
        return text
