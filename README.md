# Whisper Typing

A powerful, background speech-to-text application for Windows that runs locally. It listens for a global hotkey to record your voice, transcribes it using OpenAI's Whisper model (accelerated with `transformers` and CUDA), and types the result into your active window after your confirmation.

## Features

- **Global Hotkeys**: Control recording and typing from any application.
    - **Record/Stop**: `F8` (default)
    - **Confirm Type**: `F9` (default)
    - **Improve Text**: `F10` (default) - Uses Gemini AI to fix grammar.
- **Interactive Modes**: Type `/r` to reload, `/p` to pause, `/c` to change mic, or `/q` to quit.
- **Preview Mode**: Transcribed text is shown in the console first. You decide when to paste it.
- **Local Processing**: All audio is processed locally on your machine. No data is sent to the cloud.
- **Smart Focus**: Automatically refocuses your target window before typing (uses `pygetwindow`).
- **Clipboard Typing**: Uses Ctrl+V for fast, reliable text insertion (requires `pyperclip`).
- **GPU Acceleration**: Supports NVIDIA GPUs for lightning-fast transcription (requires CUDA).
- **Customizable**: Configure hotkeys, model size (e.g., `tiny`, `base`, `large`), and language via command-line arguments.

## Prerequisites

- **Python 3.10+**
- **FFmpeg**: Must be available on your system `PATH` (used for audio processing).
- **NVIDIA GPU (Recommended)**: For optimal performance. The app will fallback to CPU but it is significantly slower.

## Installation

This project uses `uv` for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/rpfilomeno/whispher-typing.git
    cd whispher-typing
    ```

2.  **Configure API Key**:
    Create a `.env` file in the project root:
    ```env
    GEMINI_API_KEY=your_google_ai_key
    ```

3.  **Install dependencies**:
    ```bash
    uv sync
    ```

## Usage

Run the application using `uv`:

```bash
uv run whisper-typing
```

### Workflow

1.  **Start Recording**: Press **F8**. You will see "Recording started..." in the console.
2.  **Speak**: Say what you want to type.
3.  **Stop & Transcribe**: Press **F8** again. The audio will be processed.
4.  **Preview**: The transcribed text will appear in the console:
    ```
    [PREVIEW] Transcribed text: "Hello world"
    Press <f9> to type this text.
    ```
5.  **Type**: Switch to your target application (e.g., Notepad, Slack) and press **F9**. The text will be typed out automatically.

## Configuration

You can customize the application behavior via CLI arguments or `config.json`.

### Method 1: Command Line Arguments
```bash
# Change hotkeys
uv run whisper-typing --hotkey "<f7>" --type-hotkey "<f10>"

# Use a larger, more accurate model
uv run whisper-typing --model openai/whisper-large-v3

# Specify input language
uv run whisper-typing --language en
```

### Method 2: JSON Configuration
Create a `config.json` file in the same directory:
```json
{
    "hotkey": "<f8>",
    "type_hotkey": "<f9>",
    "improve_hotkey": "<f10>",
    "model": "openai/whisper-base",
    "language": null,
    "gemini_api_key": "YOUR_API_KEY_HERE"
}
```

### Priority
1. Command Line Arguments
2. `config.json` file
3. Default values


### Supported Models
Any Hugging Face compatible Whisper model, e.g.:
- `openai/whisper-tiny`
- `openai/whisper-base` (Default)
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v3`

## Troubleshooting

- **FFmpeg not found**: Ensure FFmpeg is installed and added to your system Environment Variables.
- **Audio not recording**: Check your default microphone input settings in Windows.
- **Slow Transcription**: Ensure `torch` is using your GPU. This app prints the device (cuda/cpu) on startup.
