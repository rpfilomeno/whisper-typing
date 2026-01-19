from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.screen import ModalScreen, Screen
from textual.widgets import Header, Footer, Button, Label, Input, Select, Static, Checkbox
from textual.binding import Binding

from ..ai_improver import AIImprover

class ConfigurationScreen(Screen):
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
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.inputs = {}

    def compose(self) -> ComposeResult:
        config = self.controller.config
        
        # Microphones
        devices = self.controller.list_input_devices()
        # devices is list of (index, name)
        # Select takes list of (label, value)
        mic_options = [(name, index) for index, name in devices]
        # Add 'Default' option
        mic_options.insert(0, ("Default System Mic", None))
        
        current_mic = config.get("microphone_name")
        start_value = None
        if current_mic:
            # Find index for name
            for idx, name in devices:
                if name == current_mic:
                    start_value = idx
                    break

        # Whisper Models
        model_options = [
            # Standard Multilingual
            ("Tiny (39M)", "openai/whisper-tiny"),
            ("Base (74M)", "openai/whisper-base"),
            ("Small (244M)", "openai/whisper-small"),
            ("Medium (769M)", "openai/whisper-medium"),
            ("Large v1 (1550M)", "openai/whisper-large-v1"),
            ("Large v2 (1550M)", "openai/whisper-large-v2"),
            ("Large v3 (1550M)", "openai/whisper-large-v3"),
            ("Large (Latest)", "openai/whisper-large"),
            # English-Only
            ("Tiny English", "openai/whisper-tiny.en"),
            ("Base English", "openai/whisper-base.en"),
            ("Small English", "openai/whisper-small.en"),
            ("Medium English", "openai/whisper-medium.en"),
            # Distilled & Turbo
            ("Turbo (Fastest)", "openai/whisper-large-v3-turbo"),
            ("Distil Small En", "distil-whisper/distil-small.en"),
            ("Distil Medium En", "distil-whisper/distil-medium.en"),
            ("Distil Large v2", "distil-whisper/distil-large-v2"),
            ("Distil Large v3", "distil-whisper/distil-large-v3"),
        ]

        # Precision / Compute Type
        compute_type_options = [
            ("Auto (Recommended)", "auto"),
            ("float16 (Fast GPU)", "float16"),
            ("int8 (Fast CPU)", "int8"),
            ("int8_float16", "int8_float16"),
            ("float32 (Accurate)", "float32"),
        ]
        
        current_compute_type = config.get("compute_type", "auto")

        # Device Selection
        device_options = [
            ("CPU", "cpu"),
            ("GPU (CUDA)", "cuda"),
        ]

        # Gemini Models
        gemini_api_key = config.get("gemini_api_key")
        gemini_models = []
        if gemini_api_key:
            model_ids = AIImprover.list_models(gemini_api_key)
            # model_ids are like 'models/gemini-1.5-flash'
            gemini_models = [(m.split('/')[-1], m) for m in model_ids]
        
        if not gemini_models:
            # Fallback models if API call fails or no key
            gemini_models = [
                ("Gemini 1.5 Flash", "models/gemini-1.5-flash"),
                ("Gemini 1.5 Pro", "models/gemini-1.5-pro"),
                ("Gemini 2.0 Flash", "models/gemini-2.0-flash"),
            ]
        
        current_gemini_model = config.get("gemini_model") or "models/gemini-1.5-flash"
        # Ensure current model is in options so Select doesn't crash
        if current_gemini_model and not any(m[1] == current_gemini_model for m in gemini_models):
            gemini_models.append((current_gemini_model.split('/')[-1], current_gemini_model))
        
        yield Container(
            Label("Configuration", id="title"),
            
            Label("Microphone:"),
            Select(mic_options, value=start_value, id="mic_select"),
            
            Label("Whisper Model:"),
            Select(model_options, value=config.get("model"), id="model_select"),

            Label("Device:"),
            Select(device_options, value=config.get("device", "cpu"), id="device_select"),

            Label("Compute Type:"),
            Select(compute_type_options, value=current_compute_type, id="compute_type_select"),
            
            Label("Gemini API Key:"),
            Input(value=config.get("gemini_api_key") or "", password=True, id="api_key_input"),

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
            
            Horizontal(
                Button("Save", variant="primary", id="save_btn"),
                Button("Cancel", variant="error", id="cancel_btn"),
                id="buttons"
            ),
            id="dialog"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_btn":
            self.save_and_exit()
        elif event.button.id == "cancel_btn":
            self.app.pop_screen()

    def action_cancel(self):
        self.app.pop_screen()

    def save_and_exit(self):
        # Gather values
        mic_select = self.query_one("#mic_select", Select)
        model_select = self.query_one("#model_select", Select)
        device_select = self.query_one("#device_select", Select)
        api_input = self.query_one("#api_key_input", Input)
        hotkey_input = self.query_one("#hotkey_input", Input)
        type_input = self.query_one("#type_hotkey_input", Input)
        gemini_model_select = self.query_one("#gemini_model_select", Select)
        debug_checkbox = self.query_one("#debug_checkbox", Checkbox)
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
            "gemini_api_key": api_input.value,
            "gemini_model": gemini_model_select.value,
            "debug": debug_checkbox.value,
            "hotkey": hotkey_input.value,
            "type_hotkey": type_input.value,
            "typing_wpm": typing_wpm
        }
        
        # Handle Microphone Name
        mic_idx = mic_select.value
        if mic_idx is not None:
             devices = self.controller.list_input_devices()
             for idx, name in devices:
                 if idx == mic_idx:
                     new_config["microphone_name"] = name
                     break
        
        # Change detection
        current_config = self.controller.config
        has_changes = False
        
        # We check keys that are in new_config
        for key, value in new_config.items():
            if current_config.get(key) != value:
                has_changes = True
                break
        
        # Special handling for API Key separately or part of change
        import os
        env_api_key = os.getenv("GEMINI_API_KEY", "")
        if api_input.value != env_api_key:
            self.controller.update_env_api_key(api_input.value)
            has_changes = True # Signal that we should reload/re-init if key changed
        
        if has_changes:
            self.controller.update_config(new_config)
            self.dismiss(True) # Return True to indicate save and reload
        else:
            self.dismiss(False) # Return False to indicate no changes
