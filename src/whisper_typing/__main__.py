import argparse
import json
import os
import sys
import threading
from typing import Any, Dict
from pynput import keyboard
from .audio_capture import AudioRecorder
from .transcriber import Transcriber
from .typer import Typer
from .ai_improver import AIImprover

DEFAULT_CONFIG = {
    "hotkey": "<f8>",
    "type_hotkey": "<f9>",
    "improve_hotkey": "<f10>",
    "model": "openai/whisper-base",
    "language": None,
    "gemini_api_key": ""
}

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
                print(f"Loaded config from {config_path}")
                return user_config
        except Exception as e:
            print(f"Error loading {config_path}: {e}")
    return {}

def main() -> None:
    """Run the application."""
    # 1. Load defaults
    config = DEFAULT_CONFIG.copy()
    
    # 2. Load JSON config
    file_config = load_config()
    config.update(file_config)
    
    # 3. Parse Args (override config if specified)
    parser = argparse.ArgumentParser(description="Whisper Typing - Background Speech to Text")
    parser.add_argument("--hotkey", help=f"Global hotkey to toggle recording (default: {config['hotkey']})")
    parser.add_argument("--type-hotkey", help=f"Global hotkey to type the pending text (default: {config['type_hotkey']})")
    parser.add_argument("--improve-hotkey", help=f"Global hotkey to improve text with AI (default: {config['improve_hotkey']})")
    parser.add_argument("--model", help=f"Whisper model ID (default: {config['model']})")
    parser.add_argument("--language", help=f"Language code (default: {config['language']})")
    parser.add_argument("--api-key", help="Gemini API Key (overrides config)")

    args = parser.parse_args()
    
    # Update config with CLI args only if provided
    if args.hotkey: config["hotkey"] = args.hotkey
    if args.type_hotkey: config["type_hotkey"] = args.type_hotkey
    if args.improve_hotkey: config["improve_hotkey"] = args.improve_hotkey
    if args.model: config["model"] = args.model
    if args.language: config["language"] = args.language
    if args.api_key: config["gemini_api_key"] = args.api_key
    
    print(f"Initializing Whisper Typing...")
    print(f"Record Hotkey:  {config['hotkey']}")
    print(f"Type Hotkey:    {config['type_hotkey']}")
    print(f"Improve Hotkey: {config['improve_hotkey']}")
    print(f"Model:          {config['model']}")
    print(f"AI Enabled:     {'Yes' if config['gemini_api_key'] else 'No'}")
    
    try:
        recorder = AudioRecorder()
        transcriber = Transcriber(model_id=config["model"], language=config["language"])
        improver = AIImprover(api_key=config["gemini_api_key"])
        typer = Typer()
    except Exception as e:
        print(f"Error initializing components: {e}")
        return

    is_processing = False
    pending_text = None

    def on_record_toggle():
        nonlocal is_processing, pending_text
        if is_processing:
            print("Still processing previous audio, please wait.")
            return

        if recorder.recording:
            # Stop recording and process
            print("\nStopping recording...")
            audio_path = recorder.stop()
            
            if audio_path:
                is_processing = True
                
                def process_audio():
                    nonlocal is_processing, pending_text
                    try:
                        text = transcriber.transcribe(audio_path)
                        if text:
                            pending_text = text
                            print(f"\n[PREVIEW] Transcribed text: \"{text}\"")
                            print(f"Actions: Type ({config['type_hotkey']}) | Improve ({config['improve_hotkey']})")
                        else:
                            print("\n[PREVIEW] No text transcribed.")
                    except Exception as e:
                        print(f"Error during processing: {e}")
                    finally:
                        is_processing = False
                        
                threading.Thread(target=process_audio).start()
            else:
                print("No audio recorded.")
        else:
            # Start recording
            pending_text = None 
            recorder.start()

    def on_type_confirm():
        nonlocal pending_text
        if pending_text:
            typer.type_text(pending_text)
            pending_text = None # Clear after typing
            print("\nText typed and cleared.")
        else:
            print("\nNo pending text to type. Record something first.")

    def on_improve_text():
        nonlocal pending_text, is_processing
        if is_processing:
             print("Please wait, currently processing...")
             return
             
        if pending_text:
            is_processing = True
            def run_improve():
                nonlocal pending_text, is_processing
                try:
                    improved = improver.improve_text(pending_text)
                    if improved:
                        pending_text = improved
                        print(f"\n[AI IMPROVED] \"{pending_text}\"")
                        print(f"Press {config['type_hotkey']} to type.")
                finally:
                    is_processing = False
            
            threading.Thread(target=run_improve).start()
        else:
             print("\nNo pending text to improve.")

    print(f"Ready! Press {config['hotkey']} to toggle recording.")
    
    try:
        with keyboard.GlobalHotKeys({
            config['hotkey']: on_record_toggle,
            config['type_hotkey']: on_type_confirm,
            config['improve_hotkey']: on_improve_text
        }) as h:
            h.join()
    except ValueError as e:
        print(f"Invalid hotkey format. Please use pynput format (e.g., '<f8>', '<ctrl>+<alt>+h')")
        print(f"Error details: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
