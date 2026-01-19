import pygetwindow as gw
from typing import Optional, Any

class WindowManager:
    def __init__(self):
        pass
        
    def get_active_window(self) -> Optional[Any]:
        """Get the currently active window object."""
        try:
            # Returns the active window object
            window = gw.getActiveWindow()
            if window:
                return window
        except Exception as e:
            print(f"Error getting active window: {e}")
        return None

    def focus_window(self, window: Any) -> bool:
        """Bring the specified window object to the foreground."""
        if not window:
            return False
            
        try:
            if window.isActive:
                return True
                
            # If minimized, restore it
            if window.isMinimized:
                window.restore()
                
            # Activate brings it to foreground
            window.activate()
            return True
        except Exception as e:
            print(f"Error focusing window '{window.title if hasattr(window, 'title') else 'Unknown'}': {e}")
            return False
