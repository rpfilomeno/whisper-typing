import threading
from typing import Optional

from datetime import datetime

import numpy as np
import sounddevice as sd


class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, device_index=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.recording = False
        self.frames = [] # Use list for easier access to current buffer
        self.thread = None
        self._lock = threading.Lock()

    @staticmethod
    def list_devices():
        """List all available input devices."""
        print("\nAvailable Input Devices:")
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"{i}: {dev['name']}")
                input_devices.append((i, dev['name']))
        return input_devices

    def _callback(self, indata, frames, time, status):
        """Callback for sounddevice."""
        if status:
            print(f"Status: {status}")
        with self._lock:
            self.frames.append(indata.copy())

    def _record(self):
        """Internal recording loop."""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_index,
                callback=self._callback
            ):
                while self.recording:
                    sd.sleep(100)
        except Exception as e:
            print(f"Error in recording loop: {e}")
            self.recording = False

    def start(self):
        """Start recording."""
        if self.recording:
            return
        
        self.recording = True
        with self._lock:
            self.frames = [] # Clear frames
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print("Recording started...")

    def get_current_data(self) -> Optional[np.ndarray]:
        """Get the current accumulated audio data as a numpy array."""
        with self._lock:
            if not self.frames:
                return None
            data = list(self.frames) # Copy list
            
        if not data:
            return None
            
        # Concatenate and flatten to 1D array for mono
        recording = np.concatenate(data, axis=0)
        if self.channels == 1:
            recording = recording.flatten()
        return recording

    def stop(self) -> np.ndarray:
        """Stop recording and return audio data as numpy array (float32)."""
        if not self.recording:
            return None

        self.recording = False
        self.thread.join()
        
        recording = self.get_current_data()
        print(f"Recording stopped. Captured {len(recording) if recording is not None else 0} samples.")
        return recording
