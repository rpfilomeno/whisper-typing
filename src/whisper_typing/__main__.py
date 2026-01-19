import argparse
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional
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
    "gemini_api_key": "",
    "gemini_prompt": None 
}

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
                return user_config
        except Exception as e:
            print(f"Error loading {config_path}: {e}")
    return {}

class WhisperTypingApp:
    def __init__(self):
        self.config = {}
        self.args = None
        self.recorder = None
        self.transcriber = None
        self.typer = None
        self.improver = None
        self.listener = None
        self.is_processing = False
        self.pending_text = None
        self.paused = False
        self.current_model_id = None
        self.current_language = None
        
    def load_configuration(self, args):
        """Load and merge configuration."""
        self.args = args
        self.config = DEFAULT_CONFIG.copy()
        file_config = load_config()
        self.config.update(file_config)
        
        # Override with CLI args
        if args.hotkey: self.config["hotkey"] = args.hotkey
        if args.type_hotkey: self.config["type_hotkey"] = args.type_hotkey
        if args.improve_hotkey: self.config["improve_hotkey"] = args.improve_hotkey
        if args.model: self.config["model"] = args.model
        if args.language: self.config["language"] = args.language
        if args.api_key: self.config["gemini_api_key"] = args.api_key

    def initialize_components(self):
        """Initialize or re-initialize components."""
        print(f"Initializing Whisper Typing...")
        print(f"Record Hotkey:  {self.config['hotkey']}")
        print(f"Type Hotkey:    {self.config['type_hotkey']}")
        print(f"Improve Hotkey: {self.config['improve_hotkey']}")
        print(f"Model:          {self.config['model']}")
        print(f"AI Enabled:     {'Yes' if self.config['gemini_api_key'] else 'No'}")

        try:
            # Reload Optimization: Check if model/language changed
            if (not self.transcriber or 
                self.current_model_id != self.config["model"] or 
                self.current_language != self.config["language"]):
                
                print("Loading Transcriber pipeline...")
                self.transcriber = Transcriber(model_id=self.config["model"], language=self.config["language"])
                self.current_model_id = self.config["model"]
                self.current_language = self.config["language"]
            else:
                print("Transcriber configuration unchanged, keeping existing model.")
            
            # These are cheap to recreate, but effectively just updating config is cleaner if we supported it.
            # Recreating is safer to ensure new keys/settings are picked up.
            self.recorder = AudioRecorder()
            self.typer = Typer()
            self.improver = AIImprover(api_key=self.config["gemini_api_key"])
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            return False
        return True

    def start_listener(self):
        """Setup and start the hotkey listener."""
        if self.listener:
            self.listener.stop()
        
        try:
            self.listener = keyboard.GlobalHotKeys({
                self.config['hotkey']: self.on_record_toggle,
                self.config['type_hotkey']: self.on_type_confirm,
                self.config['improve_hotkey']: self.on_improve_text
            })
            self.listener.start()
            print(f"Ready! Press {self.config['hotkey']} to toggle recording.")
            print("Commands: '/p' pause, '/q' quit, '/r' reload.")
        except ValueError as e:
            print(f"Invalid hotkey format: {e}")

    def stop(self):
        if self.listener:
            self.listener.stop()

    def toggle_pause(self):
        self.paused = not self.paused
        state = "PAUSED" if self.paused else "RESUMED"
        print(f"\nApp is now {state}.")
        if self.paused:
            print("Hotkeys effectively disabled.")
        else:
            print("Hotkeys enabled.")

    # --- Callbacks ---
    def on_record_toggle(self):
        if self.paused: return
        
        if self.is_processing:
            print("Still processing previous audio, please wait.")
            return

        if self.recorder.recording:
            print("\nStopping recording...")
            audio_path = self.recorder.stop()
            
            if audio_path:
                self.is_processing = True
                def process_audio():
                    try:
                        text = self.transcriber.transcribe(audio_path)
                        if text:
                            self.pending_text = text
                            print(f"\n[PREVIEW] Transcribed text: \"{text}\"")
                            print(f"Actions: Type ({self.config['type_hotkey']}) | Improve ({self.config['improve_hotkey']})")
                        else:
                            print("\n[PREVIEW] No text transcribed.")
                    except Exception as e:
                        print(f"Error during processing: {e}")
                    finally:
                        self.is_processing = False
                threading.Thread(target=process_audio).start()
            else:
                print("No audio recorded.")
        else:
            self.pending_text = None 
            self.recorder.start()

    def on_type_confirm(self):
        if self.paused: return

        if self.pending_text:
            self.typer.type_text(self.pending_text)
            self.pending_text = None
            print("\nText typed and cleared.")
        else:
            print("\nNo pending text to type.")

    def on_improve_text(self):
        if self.paused: return

        if self.is_processing:
             print("Please wait, currently processing...")
             return
             
        if self.pending_text:
            self.is_processing = True
            def run_improve():
                try:
                    prompt_template = self.config.get("gemini_prompt")
                    improved = self.improver.improve_text(self.pending_text, prompt_template=prompt_template)
                    if improved:
                        self.pending_text = improved
                        print(f"\n[AI IMPROVED] \"{self.pending_text}\"")
                        print(f"Press {self.config['type_hotkey']} to type.")
                finally:
                    self.is_processing = False
            threading.Thread(target=run_improve).start()
        else:
             print("\nNo pending text to improve.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper Typing - Background Speech to Text")
    parser.add_argument("--hotkey", help="Global hotkey to toggle recording")
    parser.add_argument("--type-hotkey", help="Global hotkey to type")
    parser.add_argument("--improve-hotkey", help="Global hotkey to improve text")
    parser.add_argument("--model", help="Whisper model ID")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--api-key", help="Gemini API Key")
    args = parser.parse_args()

    app = WhisperTypingApp()
    app.load_configuration(args)
    if not app.initialize_components():
        return
    app.start_listener()

    # Interactive Loop
    try:
        while True:
            cmd = input().strip().lower()
            if cmd == '/q':
                print("Exiting...")
                app.stop()
                break
            elif cmd == '/r':
                print("Reloading configuration...")
                app.stop()
                app.load_configuration(args)
                if app.initialize_components():
                    app.start_listener()
            elif cmd == '/p':
                app.toggle_pause()
            elif cmd:
                print(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print("\nExiting...")
        app.stop()

if __name__ == "__main__":
    main()
