from pynput.keyboard import Controller, Key
import time
import pyperclip

class Typer:
    def __init__(self):
        self.keyboard = Controller()

    def type_text(self, text: str):
        """Simulate human-like typing into the active window."""
        if not text:
            return

        print(f"Typing (at ~40 WPM): {text[:50]}...")
        try:
            # 40 WPM = 200 Characters Per Minute (assuming 5 chars per word)
            # 60 seconds / 200 characters = 0.3 seconds per character
            base_char_delay = 60.0 / (40.0 * 5.0) 
            
            import random
            
            for i, char in enumerate(text):
                self.keyboard.type(char)
                
                # Base delay with jitter (70% - 130% of base)
                delay = base_char_delay * random.uniform(0.7, 1.3)
                
                # Slower after punctuation
                if char in ".!?":
                    delay += random.uniform(0.3, 0.6)
                elif char in ",;:":
                    delay += random.uniform(0.1, 0.3)
                
                time.sleep(delay)
                
                # Extra random pauses for detection avoidance (every 15-30 chars)
                if i > 0 and i % random.randint(15, 30) == 0:
                    long_pause = random.uniform(0.2, 0.8)
                    time.sleep(long_pause)
                    
        except Exception as e:
            print(f"Error during simulated typing: {e}")
            # Emergency fallback: just dump it (though this might fail too if keyboard is the issue)
            self.keyboard.type(text)
