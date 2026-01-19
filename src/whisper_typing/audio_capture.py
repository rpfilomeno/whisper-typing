import queue
import tempfile
import threading
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd


class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, device_index=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.recording = False
        self.frames = queue.Queue()
        self.thread = None

    @staticmethod
    def list_devices():
        """List available input devices. Returns list of (index, name)."""
        devices = []
        try:
            # Query all devices
            all_devices = sd.query_devices()
            for i, dev in enumerate(all_devices):
                # Filter for input devices (> 0 input channels)
                if dev['max_input_channels'] > 0:
                    devices.append((i, dev['name']))
        except Exception as e:
            print(f"Error listing devices: {e}")
        return devices

    def _callback(self, indata, frames, time, status):
        """Callback for sounddevice."""
        if status:
            print(f"Status: {status}")
        self.frames.put(indata.copy())

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
            print(f"Error during recording: {e}")
            self.recording = False

    def start(self):
        """Start recording."""
        if self.recording:
            return
        
        self.recording = True
        self.frames = queue.Queue() # Clear queue
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print("Recording started...")

    def stop(self):
        """Stop recording and return audio data as numpy array."""
        if not self.recording:
            return None

        self.recording = False
        if self.thread:
            self.thread.join()
        
        # Collect all frames
        data = []
        while not self.frames.empty():
            data.append(self.frames.get())
        
        if not data:
            return None
            
        # Concatenate and return raw float32 array
        # This avoids saving to disk
        recording = np.concatenate(data, axis=0)
        
        # Flatten if mono (usually shape is (N, 1))
        if self.channels == 1:
            recording = recording.flatten()
            
        print(f"Recording stopped. captured {len(recording)} samples.")
        return recording
