from pynput.keyboard import Controller, Key
import time
import pyperclip

class Typer:
    def __init__(self):
        self.keyboard = Controller()

    def type_text(self, text: str):
        """Type text into the active window using clipboard paste (Ctrl+V)."""
        if not text:
            return

        print(f"Typing: {text}")
        try:
            # Save current clipboard
            old_clipboard = pyperclip.paste()
            
            # Copy new text
            pyperclip.copy(text)
            
            # Small delay to ensure clipboard is updated
            time.sleep(0.1) 
            
            # Press Ctrl+V
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.press('v')
                self.keyboard.release('v')
                
            # Wait for paste to complete
            time.sleep(0.1)
            
            # Restore clipboard (optional, generally polite but can be race-condition prone)
            # pyperclip.copy(old_clipboard) 
            
        except Exception as e:
            print(f"Error processing text output: {e}")
            # Fallback to key-by-key if clipboard fails
            print("Falling back to simulated typing...")
            for char in text:
                self.keyboard.type(char)
                time.sleep(0.005)
