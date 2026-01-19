from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Log, Label, Button
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import Screen
from rich.text import Text
import difflib
from typing import Optional

from ..app_controller import WhisperAppController
from .screens import ConfigurationScreen

class WhisperTui(App):
    # ... (CSS same as before) ...
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #status_bar {
        width: 100%;
        height: 1;
        background: $primary-background;
        color: $text;
        content-align: center middle;
        text-style: bold;
    }

    #preview_area {
        height: 60%;
        border: solid $secondary;
        background: $surface;
        padding: 1;
        margin: 1;
    }
    
    #shortcuts_info {
        text-align: center;
        color: $text-muted;
        background: $surface;
        border-bottom: solid $accent;
        padding-bottom: 0;
        margin-bottom: 1;
        width: 100%;
    }

    #log_area {
        height: 1fr;
        border: solid $accent;
        background: $surface;
        margin: 1;
    }

    .status_ready { color: $success; }
    .status_recording { color: $error; background: $error 20%; }
    .status_processing { color: $warning; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "pause", "Pause"),
        Binding("c", "configure", "Configure"),
        Binding("r", "reload", "Reload Config"),
    ]

    status_message = reactive("Starting...")
    preview_text = reactive("Ready to record...")
    shortcuts_text = reactive("")

    def __init__(self, controller: WhisperAppController):
        super().__init__()
        self.controller = controller
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()

        yield Container(
            Static(self.preview_text, id="preview_area"),
            Label(self.status_message, id="status_bar"),
            Label(self.shortcuts_text, id="shortcuts_info"), 
            Log(id="log_area"),
            id="main_container"
        )

    def on_mount(self) -> None:
        self.title = "Whisper Typing"
        
        # Connect Controller Callbacks
        self.controller.on_log = self.write_log
        self.controller.on_status_change = self.update_status
        self.controller.on_preview_update = self.update_preview
        
        self.update_shortcuts_display() # Show immediately
        
        # Initialize Controller
        self.startup_controller()

    @work(exclusive=True, thread=True)
    def startup_controller(self):
        self.write_log("Loading models... (this may take a few seconds)")
        self.update_status("Loading...")
        success = self.controller.initialize_components()
        if success:
            self.controller.start_listener()
            self.update_status("Ready")
            self.write_log("Application started successfully.")
            self.write_log(f"Press {self.controller.config['hotkey']} to record.")
        else:
            self.update_status("Initialization Failed")
            self.write_log("Failed to initialize components. Check logs/config.")

    def update_shortcuts_display(self):
        cfg = self.controller.config
        text = f"Global Keys: {cfg.get('hotkey', '?')} = Record | {cfg.get('type_hotkey', '?')} = Type | {cfg.get('improve_hotkey', '?')} = Improve"
        self.shortcuts_text = text
        try:
            self.query_one("#shortcuts_info", Label).update(text)
        except Exception:
            pass

    def write_log(self, message: str) -> None:
        log_widget = self.query_one("#log_area", Log)
        log_widget.write_line(message)

    def update_status(self, status: str) -> None:
        self.status_message = status
        # Update style based on status content (simple heuristic)
        try:
            status_widget = self.query_one("#status_bar", Label)
            status_widget.update(status)
            status_widget.classes = "" # Reset
            if "Recording" in status:
                status_widget.add_class("status_recording")
            elif "Processing" in status or "Loading" in status:
                status_widget.add_class("status_processing")
            elif "Paused" in status:
                status_widget.add_class("status_processing") # Reuse warning color for pause
            else:
                status_widget.add_class("status_ready")
        except Exception:
            pass # Widget might not be mounted yet

    def update_preview(self, text: str, original_text: Optional[str] = None) -> None:
        try:
            preview_widget = self.query_one("#preview_area", Static)
            
            if not text:
                preview_widget.update("...")
                return

            if original_text is None or original_text == text:
                preview_widget.update(text)
                return

            # Visual Diff Logic
            diff_text = Text()
            # Split by words for a better diff granularity
            words1 = original_text.split()
            words2 = text.split()
            
            s = difflib.SequenceMatcher(None, words1, words2)
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag == 'equal':
                    diff_text.append(" ".join(words1[i1:i2]) + " ")
                elif tag == 'replace':
                    diff_text.append(" ".join(words1[i1:i2]) + " ", style="red strike")
                    diff_text.append(" ".join(words2[j1:j2]) + " ", style="bold green")
                elif tag == 'delete':
                    diff_text.append(" ".join(words1[i1:i2]) + " ", style="red strike")
                elif tag == 'insert':
                    diff_text.append(" ".join(words2[j1:j2]) + " ", style="bold green")
            
            preview_widget.update(diff_text)
        except Exception as e:
            self.write_log(f"Preview error: {e}")

    def action_reload(self):
        self.write_log("Reloading configuration...")
        self.controller.stop()
        self.controller.load_configuration()
        self.startup_controller()

    @work
    async def action_configure(self):
        self.controller.stop() # Pause listener while configuring
        # Need to re-start listener if they cancel? Yes.
        
        screen = ConfigurationScreen(self.controller)
        result = await self.push_screen_wait(screen)
        
        if result:
            self.write_log("Settings updated. Reloading components...")
            self.startup_controller() # Will re-init and start listener
        else:
             self.write_log("Configuration cancelled.")
             self.controller.start_listener() # Restart listener

    def action_pause(self):
        self.controller.toggle_pause()

    def action_quit(self):
        self.controller.stop()
        self.exit()
