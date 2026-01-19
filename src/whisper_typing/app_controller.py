import os
import json
import threading
import time
from typing import Any, Dict, Optional, Callable
from dotenv import load_dotenv
from pynput import keyboard
import torch

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
    "gemini_prompt": None,
    "microphone_name": None,
    "gemini_model": None,
    "device": "cpu",
    "debug": False,
    "typing_wpm": 40
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
    """Save configuration to JSON file, excluding sensitive data."""
    try:
        # Create a copy and remove sensitive keys
        save_data = config.copy()
        save_data.pop("gemini_api_key", None)
        
        with open(config_path, "w") as f:
            json.dump(save_data, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

class WhisperAppController:
    """
    Controller for Whisper Typing App logic.
    Decoupled from UI (CLI or TUI).
    """
    def __init__(self):
        self.config = {}
        self.recorder = None
        self.transcriber = None
        self.typer = None
        self.improver = None
        self.listener = None
        self.window_manager = WindowManager()
        self.target_window_handle = None
        
        self.is_processing = False
        self.pending_text = None
        self.paused = False
        
        # State tracking for optimization
        self.current_model_id = None
        self.current_language = None
        self.current_mic_index = None
        self.current_device = None

        # Callbacks for UI updates
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_preview_update: Optional[Callable[[str, Optional[str]], None]] = None

    def log(self, message: str):
        if self.on_log:
            self.on_log(message)
        else:
            print(message) # Fallback

    def set_status(self, status: str):
        if self.on_status_change:
            self.on_status_change(status)

    def load_configuration(self, args=None):
        """Load and merge configuration."""
        self.config = DEFAULT_CONFIG.copy()
        file_config = load_config()
        self.config.update(file_config)
        
        # Load from environment
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            self.config["gemini_api_key"] = env_key
        
        # Override with CLI args if provided
        if args:
            if args.hotkey: self.config["hotkey"] = args.hotkey
            if args.type_hotkey: self.config["type_hotkey"] = args.type_hotkey
            if args.improve_hotkey: self.config["improve_hotkey"] = args.improve_hotkey
            if args.model: self.config["model"] = args.model
            if args.language: self.config["language"] = args.language
            if args.api_key: self.config["gemini_api_key"] = args.api_key

    def get_mic_index_from_config(self):
        """Find device index based on configured name."""
        mic_name = self.config.get("microphone_name")
        if not mic_name:
            return None
            
        import sounddevice as sd
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 and mic_name in dev['name']:
                 return i
        return None

    def list_input_devices(self):
        """Wrapper to list devices."""
        return AudioRecorder.list_devices()

    def update_config(self, new_config: Dict[str, Any]):
        """Update runtime config and save to file."""
        self.config.update(new_config)
        save_config(self.config)
        self.log("Configuration saved.")

    def update_env_api_key(self, api_key: str):
        """Update Gemini API Key in .env file."""
        try:
            from dotenv import set_key
            env_file = ".env"
            # Ensure file exists
            if not os.path.exists(env_file):
                with open(env_file, "w") as f:
                    pass
            set_key(env_path=env_file, key_to_set="GEMINI_API_KEY", value_to_set=api_key)
            # Update current session environment variable
            os.environ["GEMINI_API_KEY"] = api_key
            self.log("API Key updated in .env")
        except Exception as e:
            self.log(f"Error updating .env: {e}")

    def initialize_components(self) -> bool:
        """Initialize or re-initialize components."""
        self.log("Initializing components...")
        
        # Microphone Setup
        mic_index = self.get_mic_index_from_config()
        # Note: If mic not found, we default to None (System Default) 
        # instead of interactive prompt here. The UI should handle setup.
        self.current_mic_index = mic_index
        mic_name = self.config.get("microphone_name", "Default")
        
        # Gemini Model logic handled in config/UI
        
        try:
            # Reload Optimization: Check if model/language changed
            if (not self.transcriber or 
                self.current_model_id != self.config["model"] or 
                self.current_language != self.config["language"] or
                self.current_device != self.config.get("device", "cpu")):
                
                self.log(f"Loading Transcriber ({self.config['model']})...")
                device = self.config.get("device", "cpu")
                self.transcriber = Transcriber(
                    model_id=self.config["model"], 
                    language=self.config["language"],
                    device=device
                )
                self.current_model_id = self.config["model"]
                self.current_language = self.config["language"]
                self.current_device = device
            
            # Recreate recorder with specific device
            self.recorder = AudioRecorder(device_index=self.current_mic_index)
            self.typer = Typer(wpm=self.config.get("typing_wpm", 40))
            self.improver = AIImprover(
                api_key=self.config["gemini_api_key"], 
                model_name=self.config.get("gemini_model") or "gemini-1.5-flash",
                debug=self.config.get("debug", False),
                logger=self.log
            )
            
            self.log("Components initialized.")
            return True
        except Exception as e:
            self.log(f"Error initializing components: {e}")
            return False

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
            self.log(f"Hotkeys registered. Press {self.config['hotkey']} to record.")
            self.set_status("Ready")
        except ValueError as e:
            self.log(f"Invalid hotkey format: {e}")

    def stop(self):
        if self.listener:
            self.listener.stop()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.set_status("Paused")
            self.log("App paused. Hotkeys disabled.")
        else:
            self.set_status("Ready")
            self.log("App resumed.")

    # --- Callbacks ---
    def on_record_toggle(self):
        if self.paused: return
        
        if self.is_processing:
            self.log("Busy processing, ignoring record toggle.")
            return

        if self.recorder.recording:
            # Finishing recording
            self.log("Stopping recording...")
            self.set_status("Processing")
            audio_data = self.recorder.stop()
            
            if audio_data is not None:
                self.is_processing = True
                def process_audio():
                    try:
                        text = self.transcriber.transcribe(audio_data)
                        if text:
                            old_text = self.pending_text
                            self.pending_text = text
                            self.log(f"Transcribed: {text}")
                            if self.on_preview_update:
                                self.on_preview_update(text, None)
                            self.set_status("Text Ready")
                        else:
                            self.log("No text transcribed.")
                            self.set_status("Ready")
                    except Exception as e:
                        self.log(f"Error: {e}")
                        self.set_status("Error")
                    finally:
                        self.is_processing = False
                threading.Thread(target=process_audio).start()
            else:
                self.log("No audio data.")
                self.set_status("Ready")
        else:
            # Starting recording
            if self.window_manager:
                self.target_window_handle = self.window_manager.get_active_window()
            
            self.pending_text = None 
            if self.on_preview_update:
                self.on_preview_update("", None) # Clear preview
            
            self.recorder.start()
            self.set_status("Recording")
            self.log("Recording started...")

    def on_type_confirm(self):
        if self.paused: return

        # If already typing, signal to stop
        if hasattr(self, "_is_typing") and self._is_typing:
            self.log("Stopping typing simulation...")
            if hasattr(self, "typing_stop_event"):
                self.typing_stop_event.set()
            return

        if self.pending_text:
            text_to_type = self.pending_text
            # Note: We no longer clear self.pending_text or preview here
            
            if not hasattr(self, "typing_stop_event"):
                self.typing_stop_event = threading.Event()
            self.typing_stop_event.clear()
            self._is_typing = True
            
            def _check_focus():
                if not self.window_manager or not self.target_window_handle:
                    return True # Can't check
                    
                # pygetwindow objects can be compared directly if they are the same handle
                active = self.window_manager.get_active_window()
                if active and hasattr(active, '_hWnd') and hasattr(self.target_window_handle, '_hWnd'):
                    return active._hWnd == self.target_window_handle._hWnd
                return active == self.target_window_handle

            def _async_typing():
                try:
                    if self.window_manager and self.target_window_handle:
                        if not self.window_manager.focus_window(self.target_window_handle):
                            self.log("Failed to restore focus.")
                            self._is_typing = False
                            return 
                        time.sleep(0.3) 

                    self.typer.type_text(
                        text_to_type, 
                        stop_event=self.typing_stop_event,
                        check_focus=_check_focus
                    )
                    
                    if self.typing_stop_event.is_set():
                        self.log("Typing stopped.")
                    else:
                        self.log("Typing finished.")
                finally:
                    self._is_typing = False
                    self.set_status("Ready")

            threading.Thread(target=_async_typing, daemon=True).start()
        else:
            self.log("No text to type.")

    def on_improve_text(self):
        if self.paused: return

        if self.is_processing: return
             
        if self.pending_text:
            self.is_processing = True
            self.set_status("Improving AI")
            self.log("Requesting AI improvement...")
            def run_improve():
                try:
                    original_text = self.pending_text
                    prompt_template = self.config.get("gemini_prompt")
                    improved = self.improver.improve_text(original_text, prompt_template=prompt_template)
                    if improved:
                        self.pending_text = improved
                        self.log("AI Improvement applied.")
                        if self.on_preview_update:
                            self.on_preview_update(improved, original_text)
                        self.set_status("Text Ready (Improved)")
                except Exception as e:
                   self.log(f"AI Error: {e}")
                finally:
                    self.is_processing = False
            threading.Thread(target=run_improve).start()
        else:
             self.log("No text to improve.")
