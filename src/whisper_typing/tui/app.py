"""TUI application using Textual."""

import contextlib
import difflib
from datetime import UTC, datetime
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, RichLog, Static

from whisper_typing.app_controller import WhisperAppController
from whisper_typing.tui.screens import ApiKeyPromptScreen, ConfigurationScreen


class WhisperTui(App[None]):
    """Main TUI application for Whisper Typing."""

    CSS: ClassVar[str] = """
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

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("p", "pause", "Pause"),
        Binding("c", "configure", "Configure"),
        Binding("r", "reload", "Reload Config"),
    ]

    status_message: reactive[str] = reactive("Starting...")
    preview_text: reactive[str] = reactive("Ready to record...")
    shortcuts_text: reactive[str] = reactive("")

    def __init__(self, controller: WhisperAppController) -> None:
        """Initialize the TUI with a controller.

        Args:
            controller: The application controller instance.

        """
        super().__init__()
        self.controller = controller

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=True)

        yield Container(
            Static(self.preview_text, id="preview_area"),
            Label(self.status_message, id="status_bar"),
            Label(self.shortcuts_text, id="shortcuts_info"),
            RichLog(id="log_area", markup=True, highlight=True),
            id="main_container",
        )

        yield Footer()

    def on_mount(self) -> None:
        """Handle application mounting."""
        self.title = "Whisper Typing"

        # Connect Controller Callbacks
        self.controller.on_log = self.write_log
        self.controller.on_status_change = self.update_status
        self.controller.on_preview_update = self.update_preview

        self.update_shortcuts_display()  # Show immediately

        # Initialize Controller
        self.check_api_key_and_startup()

    @work
    async def check_api_key_and_startup(self) -> None:
        """Check for API key and then start the controller."""
        if not self.controller.config.get("gemini_api_key"):
            screen = ApiKeyPromptScreen()
            api_key = await self.push_screen_wait(screen)
            if api_key:
                self.controller.update_env_api_key(api_key)
                self.write_log("API Key saved.")
            else:
                self.write_log("API Key skipped. AI Improvement disabled.")

        self.startup_controller()

    @work(exclusive=True, thread=True)
    def startup_controller(self) -> None:
        """Initialize controller components in a background thread."""
        self.write_log("Loading models... (this may take a few seconds)")
        self.update_status("Loading...")
        success = self.controller.initialize_components()
        self.call_from_thread(self.update_shortcuts_display)
        if success:
            self.controller.start_listener()
            self.update_status("Ready")
            self.write_log("Application started successfully.")
            self.write_log(f"Press {self.controller.config['hotkey']} to record.")
        else:
            self.update_status("Initialization Failed")
            self.write_log("Failed to initialize components. Check logs/config.")

    def update_shortcuts_display(self) -> None:
        """Update the displayed global hotkeys."""
        cfg = self.controller.config
        has_api_key = bool(cfg.get("gemini_api_key"))

        record_key = cfg.get("hotkey", "?")
        type_key = cfg.get("type_hotkey", "?")
        improve_key = cfg.get("improve_hotkey", "?")

        text = f"Global Keys: {record_key} = Start/Stop | {type_key} = Type | "

        if has_api_key:
            text += f"{improve_key} = Improve"
        else:
            text += f"[dim]{improve_key} = Improve (Disabled)[/dim]"

        self.shortcuts_text = text
        with contextlib.suppress(Exception):
            self.query_one("#shortcuts_info", Label).update(Text.from_markup(text))

    def write_log(self, message: str) -> None:
        """Write a message to the TUI log area.

        Args:
            message: The message string to log.

        """
        log_widget = self.query_one("#log_area", RichLog)

        # Truncate long messages for the log view only
        display_message = message
        log_limit = 150
        if len(message) > log_limit:
            message_limit = 147
            display_message = message[:message_limit] + "..."

        timestamp = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
        # Using markup for colorized timestamp
        log_widget.write(f"[bold blue][{timestamp}][/bold blue] {display_message}")

    def update_status(self, status: str) -> None:
        """Update the TUI status bar.

        Args:
            status: The new status string to display.

        """
        self.status_message = status
        # Update style based on status content (simple heuristic)
        try:
            status_widget = self.query_one("#status_bar", Label)
            status_widget.update(status)
            status_widget.classes = ""  # Reset
            if "Recording" in status:
                status_widget.add_class("status_recording")
            elif "Processing" in status or "Loading" in status:
                status_widget.add_class("status_processing")
            elif "Paused" in status:
                status_widget.add_class(
                    "status_processing"
                )  # Reuse warning color for pause
            else:
                status_widget.add_class("status_ready")
        except Exception:  # noqa: BLE001, S110
            pass  # Widget might not be mounted yet

    def update_preview(self, text: str, original_text: str | None = None) -> None:
        """Update the TUI preview area with text and optional diff.

        Args:
            text: The new (or current) text to display.
            original_text: Optional original text to show a diff against.

        """
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
                if tag == "equal":
                    diff_text.append(" ".join(words1[i1:i2]) + " ")
                elif tag == "replace":
                    diff_text.append(" ".join(words1[i1:i2]) + " ", style="red strike")
                    diff_text.append(" ".join(words2[j1:j2]) + " ", style="bold green")
                elif tag == "delete":
                    diff_text.append(" ".join(words1[i1:i2]) + " ", style="red strike")
                elif tag == "insert":
                    diff_text.append(" ".join(words2[j1:j2]) + " ", style="bold green")

            preview_widget.update(diff_text)
        except Exception as e:  # noqa: BLE001
            self.write_log(f"Preview error: {e}")

    def action_reload(self) -> None:
        """Reload the application configuration."""
        self.write_log("Reloading configuration...")
        self.controller.stop()
        self.controller.load_configuration()
        self.startup_controller()

    @work
    async def action_configure(self) -> None:
        """Open the configuration screen."""
        self.controller.stop()  # Pause listener while configuring

        screen = ConfigurationScreen(self.controller)
        result = await self.push_screen_wait(screen)

        if result:
            self.write_log("Settings updated. Reloading components...")
            self.startup_controller()  # Will re-init and start listener
        else:
            self.write_log("Configuration cancelled.")
            self.controller.start_listener()  # Restart listener

    def action_pause(self) -> None:
        """Pause or resume the application."""
        self.controller.toggle_pause()

    def action_quit(self) -> None:
        """Quit the application."""
        self.controller.stop()
        self.exit()
