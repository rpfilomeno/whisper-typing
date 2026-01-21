"""Main application controller for whisper-typing."""

import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sounddevice as sd
from dotenv import find_dotenv
from pynput import keyboard

from whisper_typing.ai_improver import AIImprover
from whisper_typing.audio_capture import AudioRecorder
from whisper_typing.transcriber import Transcriber
from whisper_typing.typer import Typer
from whisper_typing.window_manager import WindowManager

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_CONFIG: dict[str, Any] = {
    "hotkey": "<f8>",
    "type_hotkey": "<f9>",
    "improve_hotkey": "<f10>",
    "model": "openai/whisper-base",
    "language": None,
    "gemini_prompt": None,
    "microphone_name": None,
    "gemini_model": None,
    "device": "cpu",
    "compute_type": "auto",
    "debug": False,
    "typing_wpm": 40,
    "gemini_api_key": None,
}


def load_config(config_path: str = "config.json") -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        The loaded configuration dictionary.

    """
    path = Path(config_path)
    if path.exists():
        try:
            with path.open() as f:
                return json.load(f)
        except Exception:  # noqa: BLE001, S110
            pass
    return {}


def save_config(config: dict[str, Any], config_path: str = "config.json") -> None:
    """Save configuration to JSON file, excluding sensitive data.

    Args:
        config: The configuration dictionary to save.
        config_path: Path to the configuration file.

    """
    try:
        # Create a copy and remove sensitive keys
        save_data = config.copy()
        save_data.pop("gemini_api_key", None)

        with Path(config_path).open("w") as f:
            json.dump(save_data, f, indent=4)
    except Exception:  # noqa: BLE001, S110
        pass


class WhisperAppController:
    """Controller for Whisper Typing App logic.

    Decoupled from UI (CLI or TUI).
    """

    def __init__(self) -> None:
        """Initialize the WhisperAppController."""
        self.config: dict[str, Any] = {}
        self.recorder: AudioRecorder | None = None
        self.transcriber: Transcriber | None = None
        self.typer: Typer | None = None
        self.improver: AIImprover | None = None
        self.listener: keyboard.GlobalHotKeys | None = None
        self.window_manager: WindowManager = WindowManager()
        self.target_window_handle: Any | None = None

        self.is_processing: bool = False
        self.pending_text: str | None = None
        self.paused: bool = False

        # State tracking for optimization
        self.current_model_id: str | None = None
        self.current_language: str | None = None
        self.current_mic_index: int | None = None
        self.current_device: str | None = None
        self.current_compute_type: str | None = None

        self.stop_live_transcribe: threading.Event = threading.Event()
        self.live_transcribe_thread: threading.Thread | None = None

        # Callbacks for UI updates
        self.on_status_change: Callable[[str], None] | None = None
        self.on_log: Callable[[str], None] | None = None
        self.on_preview_update: Callable[[str, str | None], None] | None = None

        self.typing_stop_event: threading.Event = threading.Event()
        self._is_typing: bool = False

    def log(self, message: str) -> None:
        """Log a message using the configured UI callback.

        Args:
            message: The message to log.

        """
        if self.on_log:
            self.on_log(message)

    def set_status(self, status: str) -> None:
        """Update the application status using the configured UI callback.

        Args:
            status: The new status string.

        """
        if self.on_status_change:
            self.on_status_change(status)

    def load_configuration(self, args: Any = None) -> None:  # noqa: ANN401
        """Load and merge configuration.

        Args:
            args: Optional command line arguments to override file config.

        """
        self.config = DEFAULT_CONFIG.copy()
        file_config = load_config()
        self.config.update(file_config)

        # Load from environment
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            self.config["gemini_api_key"] = env_key

        # Override with CLI args if provided
        if args:
            if args.hotkey:
                self.config["hotkey"] = args.hotkey
            if args.type_hotkey:
                self.config["type_hotkey"] = args.type_hotkey
            if args.improve_hotkey:
                self.config["improve_hotkey"] = args.improve_hotkey
            if args.model:
                self.config["model"] = args.model
            if args.language:
                self.config["language"] = args.language
            if args.api_key:
                self.config["gemini_api_key"] = args.api_key

    def get_mic_index_from_config(self) -> int | None:
        """Find device index based on configured name.

        Returns:
            The device index if found, else None.

        """
        mic_name = self.config.get("microphone_name")
        if not mic_name:
            return None

        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and mic_name in dev["name"]:
                return i
        return None

    def list_input_devices(self) -> list[tuple[int, str]]:
        """List available audio input devices.

        Returns:
            A list of tuples containing device index and name.

        """
        return AudioRecorder.list_devices()

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update runtime config and save to file.

        Args:
            new_config: Dictionary of configuration updates.

        """
        self.config.update(new_config)
        save_config(self.config)
        self.log("Configuration saved.")

    def update_env_api_key(self, api_key: str) -> None:
        """Update Gemini API Key in .env file.

        Args:
            api_key: The new Gemini API key.

        """
        try:
            env_file = find_dotenv() or ".env"
            path = Path(env_file).absolute()
            self.log(f"Saving API key to {path}")

            # Read existing lines
            lines = []
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()

            # Update or append
            key_found = False
            new_lines = []
            for line in lines:
                if line.strip().startswith("GEMINI_API_KEY="):
                    new_lines.append(f"GEMINI_API_KEY={api_key}\n")
                    key_found = True
                else:
                    new_lines.append(line)

            if not key_found:
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines[-1] += "\n"
                new_lines.append(f"GEMINI_API_KEY={api_key}\n")

            # Write back
            with path.open("w", encoding="utf-8") as f:
                f.writelines(new_lines)

            # Update current session environment variable and config
            os.environ["GEMINI_API_KEY"] = api_key
            self.config["gemini_api_key"] = api_key
            self.log(f"API Key successfully saved to {path.name}")
        except Exception as e:  # noqa: BLE001
            self.log(f"Error saving API key: {e}")

    def initialize_components(self) -> bool:
        """Initialize or re-initialize components.

        Returns:
            True if initialization was successful, False otherwise.

        """
        self.log("Initializing components...")

        # Microphone Setup
        mic_index = self.get_mic_index_from_config()
        # Note: If mic not found, we default to None (System Default)
        # instead of interactive prompt here. The UI should handle setup.
        self.current_mic_index = mic_index

        try:
            # Reload Optimization: Check if model/language changed
            if (
                not self.transcriber
                or self.current_model_id != self.config["model"]
                or self.current_language != self.config["language"]
                or self.current_device != self.config.get("device", "cpu")
                or self.current_compute_type != self.config.get("compute_type", "auto")
            ):
                self.log(f"Loading Transcriber ({self.config['model']})...")
                device = self.config.get("device", "cpu")
                compute_type = self.config.get("compute_type", "auto")
                self.transcriber = Transcriber(
                    model_id=self.config["model"],
                    language=self.config["language"],
                    device=device,
                    compute_type=compute_type,
                )
                self.current_model_id = self.config["model"]
                self.current_language = self.config["language"]
                self.current_device = device
                self.current_compute_type = compute_type

            self.recorder = AudioRecorder(device_index=self.current_mic_index)
            self.typer = Typer(wpm=self.config.get("typing_wpm", 40))
            self.improver = AIImprover(
                api_key=self.config.get("gemini_api_key"),
                model_name=self.config.get("gemini_model") or "gemini-1.5-flash",
                debug=self.config.get("debug", False),
                logger=self.log,
            )

            self.log("Components initialized.")
        except Exception as e:  # noqa: BLE001
            self.log(f"Error initializing components: {e}")
            return False
        else:
            return True

    def start_listener(self) -> None:
        """Start the hotkey listener."""
        if self.listener:
            self.listener.stop()

        try:
            self.listener = keyboard.GlobalHotKeys(
                {
                    self.config["hotkey"]: self.on_record_toggle,
                    self.config["type_hotkey"]: self.on_type_confirm,
                    self.config["improve_hotkey"]: self.on_improve_text,
                }
            )
            self.listener.start()
            self.log(f"Hotkeys registered. Press {self.config['hotkey']} to record.")
            self.set_status("Ready")
        except ValueError as e:
            self.log(f"Invalid hotkey format: {e}")
            self.set_status("Hotkey Error")

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self.listener:
            self.listener.stop()

    def toggle_pause(self) -> None:
        """Toggle the application pause state."""
        self.paused = not self.paused
        if self.paused:
            self.set_status("Paused")
            self.log("App paused. Hotkeys disabled.")
        else:
            self.set_status("Ready")
            self.log("App resumed.")

    # --- Callbacks ---
    def on_record_toggle(self) -> None:
        """Toggle audio recording."""
        if self.paused:
            return

        if self.is_processing:
            self.log("Busy processing, ignoring record toggle.")
            return

        if not self.recorder:
            self.log("Recorder not initialized.")
            return

        if self.recorder.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Handle the start of an audio recording session."""
        if self.window_manager:
            self.target_window_handle = self.window_manager.get_active_window()

        self.pending_text = None
        if self.on_preview_update:
            self.on_preview_update("", None)  # Clear preview

        if self.recorder:
            self.recorder.start()
        self.set_status("Recording")
        self.log("Recording started...")

        # Start live transcription loop
        self.stop_live_transcribe.clear()
        self.live_transcribe_thread = threading.Thread(
            target=self._live_transcription_loop, daemon=True
        )
        self.live_transcribe_thread.start()

    def _stop_recording(self) -> None:
        """Handle the end of an audio recording session."""
        self.log("Stopping recording...")
        self.set_status("Processing")

        # Stop live transcription loop
        self.stop_live_transcribe.set()
        if self.live_transcribe_thread:
            self.live_transcribe_thread.join()

        if not self.recorder:
            return

        audio_data = self.recorder.stop()

        if audio_data is not None:
            self.is_processing = True

            def process_audio() -> None:
                try:
                    if self.transcriber:
                        text = self.transcriber.transcribe(audio_data)
                        if text:
                            self.pending_text = text
                            self.log(f"Transcribed: {text}")
                            if self.on_preview_update:
                                self.on_preview_update(text, None)
                            self.set_status("Text Ready")
                        else:
                            self.log("No text transcribed.")
                            self.set_status("Ready")
                except Exception as e:  # noqa: BLE001
                    self.log(f"Error: {e}")
                    self.set_status("Error")
                finally:
                    self.is_processing = False

            threading.Thread(target=process_audio).start()
        else:
            self.log("No audio data.")
            self.set_status("Ready")

    def _live_transcription_loop(self) -> None:
        """Periodically transcribe the current audio buffer during recording."""
        last_transcription_time = time.time()
        while not self.stop_live_transcribe.is_set():
            time.sleep(0.5)  # Update interval

            throttle_limit = 0.8
            if (
                time.time() - last_transcription_time < throttle_limit
            ):  # Throttle to ~1s
                continue

            if not self.recorder or not self.transcriber:
                continue

            audio_data = self.recorder.get_current_data()
            audio_buffer_min_len = 8000
            if (
                audio_data is not None and len(audio_data) > audio_buffer_min_len
            ):  # At least 0.5s of audio
                try:
                    text = self.transcriber.transcribe(audio_data)
                    if text and text != self.pending_text:
                        self.pending_text = text
                        if self.on_preview_update:
                            self.on_preview_update(text, None)
                    last_transcription_time = time.time()
                except Exception:  # noqa: BLE001, S110
                    # Don't log errors too frequently in the loop
                    pass

    def on_type_confirm(self) -> None:
        """Confirm and start typing the transcribed text."""
        if self.paused:
            return

        if self._is_typing:
            self.log("Stopping typing simulation...")
            self.typing_stop_event.set()
            return

        if self.pending_text:
            text_to_type = self.pending_text
            self.typing_stop_event.clear()
            self._is_typing = True

            threading.Thread(
                target=self._async_typing_wrapper, args=(text_to_type,), daemon=True
            ).start()
        else:
            self.log("No text to type.")

    def _async_typing_wrapper(self, text: str) -> None:
        """Wrap asynchronous typing simulation."""
        try:
            if self.window_manager and self.target_window_handle:
                if not self.window_manager.focus_window(self.target_window_handle):
                    self.log("Failed to restore focus.")
                    self._is_typing = False
                    return
                time.sleep(0.3)

            if self.typer:
                self.typer.type_text(
                    text,
                    stop_event=self.typing_stop_event,
                    check_focus=self._check_typing_focus,
                )

                if self.typing_stop_event.is_set():
                    self.log("Typing stopped.")
                else:
                    self.log("Typing finished.")
        finally:
            self._is_typing = False
            self.set_status("Ready")

    def _check_typing_focus(self) -> bool:
        """Check if the target window still has focus."""
        if not self.window_manager or not self.target_window_handle:
            return True

        active = self.window_manager.get_active_window()
        if (
            active
            and hasattr(active, "_hWnd")
            and hasattr(self.target_window_handle, "_hWnd")
        ):
            return bool(active._hWnd == self.target_window_handle._hWnd)  # noqa: SLF001
        return bool(active == self.target_window_handle)

    def on_improve_text(self) -> None:
        """Improve the current pending text using AI."""
        if self.paused:
            return

        if self.is_processing:
            return

        if self.pending_text:
            if not self.config.get("gemini_api_key"):
                self.log("AI Improvement disabled: Gemini API Key missing.")
                return

            self.is_processing = True
            self.set_status("Improving AI")
            self.log("Requesting AI improvement...")

            def run_improve() -> None:
                try:
                    original_text = self.pending_text
                    prompt_template = self.config.get("gemini_prompt")
                    if self.improver:
                        improved = self.improver.improve_text(
                            original_text, prompt_template=prompt_template
                        )
                        if improved:
                            self.pending_text = improved
                            self.log("AI Improvement applied.")
                            if self.on_preview_update:
                                self.on_preview_update(improved, original_text)
                            self.set_status("Text Ready (Improved)")
                except Exception as e:  # noqa: BLE001
                    self.log(f"AI Error: {e}")
                finally:
                    self.is_processing = False

            threading.Thread(target=run_improve).start()
        else:
            self.log("No text to improve.")
