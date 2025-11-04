"""
Live Coder by Tuomo Rainio 2025

"""

# install.py
"""Cross-platform dependency installer for Live Editor"""
import subprocess
import sys

def install():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        # First uninstall any existing opencv packages
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'uninstall', '-y',
            'opencv-python', 'opencv-python-headless'
        ])
        
        # Install opencv-contrib-python which includes full GUI support
        requirements = [
            'opencv-contrib-python>=4.5.0',  # Changed from opencv-python
            'PyQt5>=5.15.0',
            'watchdog>=2.0.0',
        ]
        
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install'
        ] + requirements)
        
        print("\n✓ Installation complete!")
        print(f"Run with: {sys.executable} run.py")
    except subprocess.CalledProcessError:
        print("\n✗ Installation failed")
        sys.exit(1)

if __name__ == '__main__':
    install()