"""
Live Coder by Tuomo Rainio 2025

"""

# run.py
#!/usr/bin/env python3
"""Launch Live Editor"""
import sys
import subprocess
import os
from pathlib import Path

def open_in_editor(filepath):
    """Open file in available code editor"""
    # Absolute path to the file
    abs_path = os.path.abspath(filepath)
    
    # Try editors in order of preference
    editors = [
        ['code', abs_path],  # VS Code
        ['code-insiders', abs_path],  # VS Code Insiders
        ['pycharm', abs_path],  # PyCharm
        ['notepad++', abs_path],  # Notepad++
        ['notepad', abs_path],  # Fallback to Notepad
    ]
    
    for editor_cmd in editors:
        try:
            subprocess.Popen(editor_cmd, 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL,
                           shell=True)
            print(f"Opened {filepath} in editor")
            return True
        except (FileNotFoundError, OSError):
            continue
    
    print(f"Could not open editor. Please open {abs_path} manually")
    return False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Run main.py as module
if __name__ == '__main__':
    # Open live.py in editor
    #open_in_editor('live.py')
    #open_in_editor('ui.py')
    
    # Launch the live coding environment
    import src.main