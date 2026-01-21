"""Configuration screens for the TUI."""

import os
from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Select,
)

from whisper_typing.ai_improver import AIImprover
from whisper_typing.app_controller import WhisperAppController
from whisper_typing.constants import WHISPER_MODELS


class ConfigurationScreen(Screen[bool]):
    """Screen for configuring application settings."""

    CSS = """
    ConfigurationScreen {
        align: center middle;
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: auto;
        padding: 0 1;
        width: 90%;
        max-width: 120;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    #title {
        column-span: 2;
        height: 1;
        content-align: center middle;
        text-style: bold;
        background: $primary;
        color: $text;
        margin-bottom: 1;
    }

    Label {
        column-span: 1;
        height: 3;
        content-align: left middle;
    }

    Select, Input {
        column-span: 1;
        width: 100%;
    }

    #buttons {
        column-span: 2;
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, controller: WhisperAppController) -> None:
        """Initialize the ConfigurationScreen.

        Args:
            controller: The application controller instance.

        """
        super().__init__()
        self.controller = controller
        self.inputs: dict[str, Any] = {}

    def _get_mic_options(self) -> tuple[list[tuple[str, int | None]], int | None]:
        """Get microphone options for selection.

        Returns:
            A tuple of (options_list, start_value).

        """
        config = self.controller.config
        devices = self.controller.list_input_devices()
        mic_options: list[tuple[str, int | None]] = [
            (name, index) for index, name in devices
        ]
        mic_options.insert(0, ("Default System Mic", None))

        current_mic = config.get("microphone_name")
        start_value = None
        if current_mic:
            for idx, name in devices:
                if name == current_mic:
                    start_value = idx
                    break
        return mic_options, start_value

    def _get_gemini_options(self) -> tuple[list[tuple[str, str]], str]:
        """Get Gemini model options for selection.

        Returns:
            A tuple of (options_list, current_value).

        """
        config = self.controller.config
        gemini_api_key = config.get("gemini_api_key")
        gemini_models: list[tuple[str, str]] = []
        if gemini_api_key:
            model_ids = AIImprover.list_models(api_key=gemini_api_key)
            gemini_models = [(m.split("/")[-1], m) for m in model_ids]

        if not gemini_models:
            gemini_models = [
                ("Gemini 1.5 Flash", "models/gemini-1.5-flash"),
                ("Gemini 1.5 Pro", "models/gemini-1.5-pro"),
                ("Gemini 2.0 Flash", "models/gemini-2.0-flash"),
            ]

        current_gemini_model = config.get("gemini_model") or "models/gemini-1.5-flash"
        if current_gemini_model and not any(
            m[1] == current_gemini_model for m in gemini_models
        ):
            gemini_models.append(
                (current_gemini_model.split("/")[-1], current_gemini_model)
            )
        return gemini_models, current_gemini_model

    def compose(self) -> ComposeResult:
        """Compose the configuration screen layout."""
        config = self.controller.config
        mic_options, start_value = self._get_mic_options()
        compute_type_options = [
            ("Auto (Recommended)", "auto"),
            ("float16 (Fast GPU)", "float16"),
            ("int8 (Fast CPU)", "int8"),
            ("int8_float16", "int8_float16"),
            ("float32 (Accurate)", "float32"),
        ]
        device_options = [("CPU", "cpu"), ("GPU (CUDA)", "cuda")]
        gemini_models, current_gemini_model = self._get_gemini_options()

        yield Container(
            Label("Configuration", id="title"),
            Label("Microphone:"),
            Select(mic_options, value=start_value, id="mic_select"),
            Label("Whisper Model:"),
            Select(WHISPER_MODELS, value=config.get("model"), id="model_select"),
            Label("Device:"),
            Select(
                device_options, value=config.get("device", "cpu"), id="device_select"
            ),
            Label("Compute Type:"),
            Select(
                compute_type_options,
                value=config.get("compute_type", "auto"),
                id="compute_type_select",
            ),
            Label("Gemini API Key:"),
            Input(
                value=config.get("gemini_api_key") or "",
                password=True,
                id="api_key_input",
            ),
            Label("Gemini Model:"),
            Select(gemini_models, value=current_gemini_model, id="gemini_model_select"),
            Label("Record Hotkey:"),
            Input(value=config.get("hotkey"), id="hotkey_input"),
            Label("Type Hotkey:"),
            Input(value=config.get("type_hotkey"), id="type_hotkey_input"),
            Label("Typing Speed (WPM):"),
            Input(value=str(config.get("typing_wpm", 40)), id="typing_wpm_input"),
            Label("Debug Mode:"),
            Checkbox(value=config.get("debug", False), id="debug_checkbox"),
            Label("Refocus Window:"),
            Checkbox(
                value=config.get("refocus_window", True), id="refocus_checkbox"
            ),
            Horizontal(
                Button("Save", variant="primary", id="save_btn"),
                Button("Cancel", variant="error", id="cancel_btn"),
                id="buttons",
            ),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "save_btn":
            self.save_and_exit()
        elif event.button.id == "cancel_btn":
            self.app.pop_screen()

    def action_cancel(self) -> None:
        """Cancel the configuration and exit the screen."""
        self.app.pop_screen()

    def _get_new_config(self) -> dict[str, Any]:
        """Gather current settings from UI widgets.

        Returns:
            A dictionary containing the new configuration values.

        """
        model_select = self.query_one("#model_select", Select)
        device_select = self.query_one("#device_select", Select)
        hotkey_input = self.query_one("#hotkey_input", Input)
        type_input = self.query_one("#type_hotkey_input", Input)
        gemini_model_select = self.query_one("#gemini_model_select", Select)
        debug_checkbox = self.query_one("#debug_checkbox", Checkbox)
        refocus_checkbox = self.query_one("#refocus_checkbox", Checkbox)
        typing_wpm_input = self.query_one("#typing_wpm_input", Input)
        compute_type_select = self.query_one("#compute_type_select", Select)

        try:
            typing_wpm = int(typing_wpm_input.value)
        except ValueError:
            typing_wpm = 40

        new_config = {
            "microphone_name": None,
            "model": model_select.value,
            "device": device_select.value,
            "compute_type": compute_type_select.value,
            "gemini_model": gemini_model_select.value,
            "debug": debug_checkbox.value,
            "refocus_window": refocus_checkbox.value,
            "hotkey": hotkey_input.value,
            "type_hotkey": type_input.value,
            "typing_wpm": typing_wpm,
        }

        # Handle Microphone Name
        mic_select = self.query_one("#mic_select", Select)
        mic_idx = mic_select.value
        if mic_idx is not None:
            devices = self.controller.list_input_devices()
            for idx, name in devices:
                if idx == mic_idx:
                    new_config["microphone_name"] = name
                    break

        return new_config

    def save_and_exit(self) -> None:
        """Gather current settings, save them, and exit the screen."""
        new_config = self._get_new_config()

        # Change detection
        current_config = self.controller.config
        has_changes = any(
            current_config.get(key) != value for key, value in new_config.items()
        )

        # Special handling for API Key separately
        api_input = self.query_one("#api_key_input", Input)
        env_api_key = os.getenv("GEMINI_API_KEY", "")
        if api_input.value != env_api_key:
            self.controller.update_env_api_key(api_input.value)
            has_changes = True  # Signal that we should reload/re-init if key changed

        if has_changes:
            self.controller.update_config(new_config)
            self.dismiss(result=True)  # Return True to indicate save and reload
        else:
            self.dismiss(result=False)  # Return False to indicate no changes


class ApiKeyPromptScreen(ModalScreen[str | None]):
    """Screen for prompting the user for a Gemini API key on startup."""

    CSS = """
    ApiKeyPromptScreen {
        align: center middle;
        background: $background 50%;
    }

    #api_dialog {
        padding: 1 2;
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
    }

    #api_title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }

    #api_note {
        margin-bottom: 1;
        color: $text-muted;
    }

    #api_link {
        color: $accent;
        text-style: underline;
        margin-bottom: 1;
    }

    #api_input {
        margin-bottom: 1;
    }

    #api_buttons {
        align: center middle;
    }

    Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the API key prompt screen layout."""
        yield Container(
            Label("Gemini API Key Required", id="api_title"),
            Label(
                "The AI text improver will not be active if an API key is not present.",
                id="api_note",
            ),
            Label("Key: https://aistudio.google.com/app/apikey", id="api_link"),
            Input(placeholder="Enter Gemini API Key...", password=True, id="api_input"),
            Horizontal(
                Button("Save", variant="primary", id="api_save_btn"),
                Button("Skip", variant="error", id="api_skip_btn"),
                id="api_buttons",
            ),
            id="api_dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "api_save_btn":
            api_key = self.query_one("#api_input", Input).value.strip()
            if api_key:
                self.dismiss(api_key)
            else:
                self.app.notify("Please enter a valid API key", severity="error")
        elif event.button.id == "api_skip_btn":
            self.dismiss(None)
