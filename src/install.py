"""
Live Coder by Tuomo Rainio 2025

"""

# install.py
"""Cross-platform dependency installer for Live Editor"""
import subprocess
import sys
import platform

def check_tkinter():
    """Check if tkinter is available, print install instructions if not"""
    try:
        import tkinter
        print("  tkinter: OK")
        return True
    except ImportError:
        print("  tkinter: NOT FOUND")
        # Print platform-specific install instructions
        os_name = platform.system()
        if os_name == "Darwin":
            print("    -> Install with: brew install python-tk@3.XX")
            print("       (replace 3.XX with your Python version)")
        elif os_name == "Linux":
            print("    -> Install with: sudo apt install python3-tk")
            print("       (or equivalent for your distro)")
        elif os_name == "Windows":
            print("    -> Reinstall Python from python.org with 'tcl/tk' option checked")
        return False

def install():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        # First uninstall any existing opencv packages
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'uninstall', '-y',
            'opencv-python', 'opencv-python-headless'
        ])

        # Install pip-installable requirements
        requirements = [
            'opencv-contrib-python>=4.5.0',
            'watchdog>=2.0.0',
            'Pillow>=9.0.0',  # Required for GUI preview
        ]

        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install'
        ] + requirements)

        # Check tkinter (system dependency, not pip-installable)
        print("\nChecking system dependencies...")
        tk_ok = check_tkinter()

        if tk_ok:
            print("\nAll dependencies installed!")
            print(f"Run with: {sys.executable} run.py")
        else:
            print("\nPip packages installed, but tkinter is missing.")
            print("Please install tkinter using the instructions above, then try again.")
            sys.exit(1)

    except subprocess.CalledProcessError:
        print("\nInstallation failed")
        sys.exit(1)

if __name__ == '__main__':
    install()