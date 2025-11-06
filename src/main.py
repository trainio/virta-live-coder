"""
Live Coder by Tuomo Rainio 2025

"""

import os
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false' # Disable on Windows
# For Linux, rebuild opencv-python with GTK support or:

import cv2
cv2.setLogLevel(0) # 0 = silent, 3 = errors only

import math
import time
import sys
from pathlib import Path
import platform

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ui      # Now from root
import live    # Now from root

from . import gui as gui_module
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import importlib
import shutil
from datetime import datetime
from pathlib import Path


def list_cameras(max_test=10):
    """Test and list all available cameras"""
    available = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                available.append({
                    'index': i,
                    'width': int(width),
                    'height': int(height),
                    'fps': fps,
                    'backend': backend
                })
                
                print(f"Camera {i}:")
                print(f"  Resolution: {int(width)}x{int(height)}")
                print(f"  FPS: {fps}")
                print(f"  Backend: {backend}")
            cap.release()
    
    if not available:
        print("No cameras found")
    else:
        print(f"\nFound {len(available)} camera(s)")
    
    return available


class FrameHistory:
    """Manages a circular buffer of recent frames"""
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.frames = []
        self.auto_mode = True  # True = automatic, False = manual
    
    def add(self, frame):
        """Add frame to history"""
        self.frames.append(frame.copy())
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
    
    def __getitem__(self, index):
        """
        Access frame by index counting backwards from newest:
        history[0] = current/newest frame
        history[1] = 1 frame ago
        history[10] = 10 frames ago
        history[99] = 99 frames ago (oldest if buffer full)
        
        Also supports negative indexing:
        history[-1] = oldest frame
        history[-2] = second oldest
        """
        if not self.frames:
            return None
        
        if index>self.max_size-1 or index == None or not isinstance(index, int):
            index = self.max_size

        if index>len(self.frames)-1:
            index=len(self.frames)-1

        # Positive index: count back from newest (reverse order)
        if index >= 0:
            if index >= len(self.frames):
                return None
            return self.frames[-(index + 1)]
        
        # Negative index: count from oldest (normal order)
        else:
            return self.frames[index]
    
    def __len__(self):
        return len(self.frames)
    
    def toggle_mode(self):
        """Switch between automatic and manual modes"""
        self.auto_mode = not self.auto_mode
        return "AUTO" if self.auto_mode else "MANUAL"




class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.needs_reload_live = False
        self.needs_reload_ui = False
    
    def on_modified(self, event):
        if event.src_path.endswith('live.py'):
            self.needs_reload_live = True
        elif event.src_path.endswith('ui.py'):
            self.needs_reload_ui = True

event_handler = ReloadHandler()
observer = Observer()
observer.schedule(event_handler, path='.', recursive=False)
observer.start()

current_camera_index = 0
cameras = list_cameras()
max_cameras = len(cameras) if cameras else 1

# GUI
# Create single instance
gui = gui_module.LiveGUI()
# Setup UI controls
ui.setup(gui)
gui.create_window()  # Now create window with all controls

# Initialize frame history and attach to gui for access from live.py
history = FrameHistory(max_size=100)
#gui.history = history  # Make accessible from live.py via gui.history

# WINDOW
cv2.namedWindow('Live', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
# Set initial window size (width, height)
cv2.resizeWindow('Live', 1920, 1080)  # Or any size you want
cv2.moveWindow('Live', 200, 0)  # x=400, y=100

CAMERA_BACKEND = None
if platform.system() == 'Windows':
    CAMERA_BACKEND = cv2.CAP_DSHOW
else:  # Linux
    CAMERA_BACKEND = cv2.CAP_ANY 

# SELECT CAMERA INPUT
cap = cv2.VideoCapture(0,CAMERA_BACKEND)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

fps_history = []
current_frame = None

# Create output folder if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Timing variables
prev_frame_time = time.perf_counter()

# After cap initialization, before while True:
start_time = time.perf_counter()
show_startup_message = True
startup_logo = None
startup_logo_loaded = False

print(f"Current working directory: {os.getcwd()}")

while True:
    # Start total loop timing
    loop_start = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    if event_handler.needs_reload_ui:
        try:
            # Close window first
            if gui.window_open:
                gui.on_close()
            
            # Clear old definitions
            gui.control_defs.clear()
            gui.control_names.clear()
            gui.params.clear()  # Also clear params to reset values
            gui.row = 0
            
            # Reload and setup
            importlib.reload(ui)
            ui.setup(gui)
            
            # Recreate window with new controls
            gui.create_window()
            
            event_handler.needs_reload_ui = False
        except Exception as e:
            print(f"UI reload error: {e}")

    # Reload effects if changed
    if event_handler.needs_reload_live:
        try:
            importlib.reload(live)
            event_handler.needs_reload_live = False
        except Exception as e:
            print(f"Live reload error: {e}")
    
    # Update GUI
    gui.update()

    # TIME ONLY THE PROCESSING (not GUI, reloads, etc)
    frame_start = time.perf_counter()
    
    # Auto mode: add every input frame to history
    if history.auto_mode:
        history.add(frame)

    # Process frame
    try:
        frame = live.processing(frame, gui, history)
        #remove frame = live.process(frame,gui)
        current_frame = frame.copy()
    except Exception as e:
        cv2.putText(frame, str(e)[:50], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)



    # In the while loop:
    if show_startup_message:
        elapsed = time.perf_counter() - start_time
        if elapsed < 5.0:
            # Load logo once
            if not startup_logo_loaded:
                logo_path = Path(__file__).parent / 'start_screen.png'
                if logo_path.exists():
                    startup_logo = cv2.imread(str(logo_path))
                    if startup_logo is not None:
                        h, w = frame.shape[:2]
                        startup_logo = cv2.resize(startup_logo, (1920, 1080))
                startup_logo_loaded = True
            
            # Show logo or text
            if startup_logo is not None:
                frame = startup_logo.copy()
            else:
                h, w = frame.shape[:2]
                cv2.putText(frame, "VIRTA", (w//2 - 50, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, "Live coder by Tuomo Rainio 2025", (w//2 - 180, h//2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            show_startup_message = False

    # Processing time only (excludes GUI, display, etc)
    process_time = (time.perf_counter() - frame_start) * 1000
    
    # Display
    cv2.imshow('Live', frame)
    
    # Calculate FPS from actual frame-to-frame time
    current_time = time.perf_counter()
    frame_delta = current_time - prev_frame_time
    prev_frame_time = current_time
    
    # Smooth FPS over last 10 frames
    if frame_delta > 0:
        fps_history.append(1.0 / frame_delta)
    if len(fps_history) > 10:
        fps_history.pop(0)
    fps = sum(fps_history) / len(fps_history) if fps_history else 0
    
    hist_mode_str = "AUTO" if history.auto_mode else "MANUAL"
    # Update title with accurate metrics
    cv2.setWindowTitle('Live', f'VIRTA LIVE CODER | Cam({current_camera_index}) | {fps:.1f} FPS | Process: {process_time:.1f}ms | History: {len(history)}/100 [{hist_mode_str}]')
    
    # Handle keyboard input
    #key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKey(30) & 0xFF  # Changed from 1 to 30ms
    # 'c' key - cycle to next camera
    if key == ord('c'):
        # Only try if we have multiple cameras
        if max_cameras <= 1:
            print("Only one camera available")
        else:
            print(f"Attempting to switch from camera {current_camera_index}...")
            
            # Try next camera WITHOUT closing current first
            next_camera = (current_camera_index + 1) % max_cameras
            
            new_cap = cv2.VideoCapture(next_camera)
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            new_cap.set(cv2.CAP_PROP_FPS, 30)
            time.sleep(0.2)
            
            # Test if it works
            ret, test_frame = new_cap.read()
            
            if ret and new_cap.isOpened():
                # Success - NOW we can close the old one
                cap.release()
                cap = new_cap
                current_camera_index = next_camera
                history.frames.clear()
                print(f"✓ Switched to camera {current_camera_index}")
            else:
                # Failed - just close the failed attempt, keep original
                new_cap.release()
                print(f"✗ Camera {next_camera} not available, staying on camera {current_camera_index}")
    # 'h' key - add frame to history (manual mode) or does nothing in auto mode
    elif key == ord('h'):
        if not history.auto_mode:
            history.add(frame)
            print(f"Frame added to history manually ({len(history)}/100)")
        else:
            print("Switch to MANUAL mode first (press 'm')")
    
    # 'm' key - toggle between auto and manual mode
    elif key == ord('m'):
        mode = history.toggle_mode()
        print(f"History mode: {mode}")

    elif key == ord('r'):
        if current_frame is not None:
            output_dir.mkdir(exist_ok=True)  # Ensure folder exists
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = output_dir / f"{timestamp}.png"
            cv2.imwrite(str(img_path), current_frame)
            code_path = output_dir / f"{timestamp}_live.py"
            shutil.copy2("live.py", code_path)
            print(f"Captured: {img_path.name} + {code_path.name}")
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
observer.stop()
observer.join()