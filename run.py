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
import platform


def open_in_editor(filepath):
    """Open file in available code editor"""
    # Absolute path to the file
    abs_path = os.path.abspath(filepath)
    
    # Mac-specific editors
    if platform.system() == "Darwin":
        editors = [
            ['open', '-a', 'Visual Studio Code', abs_path],
            ['open', '-a', 'PyCharm', abs_path],
            ['open', abs_path],
        ]
    else:
        editors = [
            ['code', abs_path],
            ['code-insiders', abs_path],
            ['pycharm', abs_path],
            ['notepad++', abs_path],
            ['notepad', abs_path],
        ]
        
    for editor_cmd in editors:
        try:
            # Remove shell=True and pass command as list
            subprocess.Popen(editor_cmd, 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
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
    open_in_editor('ui.py')
    open_in_editor('live.py')
    
    # Launch the live coding environment
    import src.main