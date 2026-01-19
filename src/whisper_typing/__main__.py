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
from .window_manager import WindowManager

DEFAULT_CONFIG = {
    "hotkey": "<f8>",
    "type_hotkey": "<f9>",
    "improve_hotkey": "<f10>",
    "model": "openai/whisper-base",
    "language": None,
    "gemini_api_key": "",
    "gemini_prompt": None,
    "microphone_name": None,
    "gemini_model": None
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

def save_config(config: Dict[str, Any], config_path: str = "config.json"):
    """Save configuration to JSON file."""
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config: {e}")

from .ai_improver import AIImprover
from .window_manager import WindowManager # Should be imported if using it

# ... (rest of imports)

class WhisperTypingApp:
    def __init__(self):
        self.config = {}
        self.args = None
        self.recorder = None
        self.transcriber = None
        self.typer = None
        self.improver = None
        self.listener = None
        self.window_manager = WindowManager() # Initialize window manager
        self.target_window_handle = None
        self.is_processing = False
        self.pending_text = None
        self.paused = False
        self.current_model_id = None
        self.current_language = None
        self.current_mic_index = None
        
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

    def select_microphone(self):
        """Interactive microphone selection."""
        devices = AudioRecorder.list_devices()
        if not devices:
            print("No input devices found!")
            return None
            
        print("\nEnter valid device ID to select, or 'c' to cancel.")
        while True:
            try:
                user_input = input("Select Microphone ID: ").strip()
                if user_input.lower() == 'c':
                    return None
                    
                idx = int(user_input)
                # verify index exists in our filtered list
                selected_dev = next((d for d in devices if d[0] == idx), None)
                
                if selected_dev:
                    print(f"Selected: {selected_dev[1]}")
                    self.config["microphone_name"] = selected_dev[1]
                    save_config(self.config) # Save permanently
                    return idx
                else:
                    print("Invalid ID. Please choose from the list above.")
            except ValueError:
                print("Invalid input. Enter a number or 'c'.")

    def get_mic_index_from_config(self):
        """Find device index based on configured name."""
        mic_name = self.config.get("microphone_name")
        if not mic_name:
            return None
            
        # We need to query devices to find the index matching the name
        import sounddevice as sd
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 and mic_name in dev['name']:
                 return i
        
        print(f"Configured microphone '{mic_name}' not found.")
        return None

    def select_gemini_model(self):
        """Interactive Gemini model selection."""
        if not self.config.get("gemini_api_key"):
            print("Gemini API Key missing. Cannot list models.")
            return None

        print("\nFetching available Gemini models...")
        models = AIImprover.list_models(self.config["gemini_api_key"])
        
        if not models:
            print("No suitable models found or API error.")
            return None
            
        print("\nAvailable Gemini Models:")
        for i, m in enumerate(models):
            print(f"{i}: {m}")
            
        print("\nEnter ID to select, or 'c' to cancel.")
        while True:
            try:
                user_input = input("Select Model ID: ").strip()
                if user_input.lower() == 'c':
                    return None
                
                idx = int(user_input)
                if 0 <= idx < len(models):
                    selected = models[idx]
                    print(f"Selected: {selected}")
                    self.config["gemini_model"] = selected
                    save_config(self.config)
                    return selected
                else:
                    print("Invalid ID.")
            except ValueError:
                print("Invalid input.")

    def initialize_components(self):
        """Initialize or re-initialize components."""
        print(f"Initializing Whisper Typing...")
        print(f"Record Hotkey:  {self.config['hotkey']}")
        print(f"Type Hotkey:    {self.config['type_hotkey']}")
        print(f"Improve Hotkey: {self.config['improve_hotkey']}")
        print(f"Model:          {self.config['model']}")
        print(f"AI Enabled:     {'Yes' if self.config['gemini_api_key'] else 'No'}")

        # Microphone Setup
        mic_index = self.get_mic_index_from_config()
        if mic_index is None:
            print("\nMicrophone not configured or not found.")
            mic_index = self.select_microphone()
            if mic_index is None:
                print("Using default system microphone.")
                mic_index = None

        self.current_mic_index = mic_index
        mic_name = self.config.get("microphone_name", "Default")
        print(f"Microphone:     {mic_name}")

        # Gemini Model Setup
        gemini_model = self.config.get("gemini_model")
        if self.config.get("gemini_api_key") and not gemini_model:
            print("\nGemini Model not configured.")
            gemini_model = self.select_gemini_model()
            if not gemini_model:
                 print("Using default model: gemini-1.5-flash")
                 gemini_model = "gemini-1.5-flash"
        
        self.config["gemini_model"] = gemini_model
        if gemini_model:
            print(f"Gemini Model:   {gemini_model}")

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
            
            # Recreate recorder with specific device
            self.recorder = AudioRecorder(device_index=self.current_mic_index)
            self.typer = Typer()
            self.improver = AIImprover(api_key=self.config["gemini_api_key"], model_name=self.config["gemini_model"])
            
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
            print("Commands: '/p' pause, '/q' quit, '/r' reload, '/c' configure.")
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
            # Finishing recording
            print("\nStopping recording...")
            audio_data = self.recorder.stop()
            
            if audio_data is not None:
                self.is_processing = True
                def process_audio():
                    try:
                        text = self.transcriber.transcribe(audio_data)
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
                print("No audio recorded (empty buffer).")
        else:
            # Starting recording
            # Capture active window
            if self.window_manager:
                self.target_window_handle = self.window_manager.get_active_window()
                print(f"Captured target window handle: {self.target_window_handle}")
            
            self.pending_text = None 
            self.recorder.start()

    def on_type_confirm(self):
        if self.paused: return

        if self.pending_text:
            # Restore focus if we have a handle
            if self.window_manager and self.target_window_handle:
                print(f"Restoring focus to window handle: {self.target_window_handle}")
                if not self.window_manager.focus_window(self.target_window_handle):
                    print("Failed to restore window focus. Aborting paste to prevent errors.")
                    print(f"Pending text retained: \"{self.pending_text}\"")
                    return # Abort
                time.sleep(0.1) # Wait for focus switch

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
    # ... (arg checks) ...
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
            elif cmd == '/c':
                print("\nConfiguration:")
                print("m: Change Microphone")
                print("g: Change Gemini Model")
                print("c: Cancel")
                choice = input("Select option: ").strip().lower()
                
                if choice == 'm':
                    app.stop() 
                    app.select_microphone()
                    if app.initialize_components():
                        app.start_listener()
                elif choice == 'g':
                    app.stop()
                    app.select_gemini_model()
                    if app.initialize_components():
                        app.start_listener()
                elif choice == 'c':
                    print("Cancelled.")
                else:
                    print("Invalid option.")
            elif cmd:
                print(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print("\nExiting...")
        app.stop()

if __name__ == "__main__":
    main()
