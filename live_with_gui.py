"""
Live Effects GUI - Interactive effect toggler
Toggle effects on/off and adjust parameters without editing code

Usage:
    python live_with_gui.py [--camera 0] [--image path/to/image.jpg]
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
from pathlib import Path
import sys
from datetime import datetime
import random
import inspect

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from src.graphics import process, draw, generate, blend, analyze

# Auto-generate effects config if needed
def ensure_effects_config():
    """Generate effects_config.py if it doesn't exist or is outdated"""
    config_path = Path(__file__).parent / 'effects_config.py'
    
    # Always regenerate on launch for latest effects
    print("Generating effects configuration...")
    try:
        import src.generate_effects_array
        config_content = src.generate_effects_array.generate_effects_array()
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Generated {config_path}")
    except Exception as e:
        print(f"Warning: Could not generate effects config: {e}")
        return False
    
    return True

# Generate config and import it
if ensure_effects_config():
    try:
        # Force reload if already imported
        import importlib
        if 'effects_config' in sys.modules:
            importlib.reload(sys.modules['effects_config'])
        
        from effects_config import EFFECTS, CATEGORIES
        USE_AUTO_CONFIG = True
        print(f"✓ Loaded {len(EFFECTS)} effects")
    except ImportError as e:
        USE_AUTO_CONFIG = False
        print(f"Warning: Could not import effects_config: {e}")
else:
    USE_AUTO_CONFIG = False
    print("Warning: Using fallback effects list")


class EffectParameter:
    """Represents a parameter for an effect"""
    def __init__(self, name, value, min_val=None, max_val=None, param_type=None):
        self.name = name
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.param_type = param_type or type(value)


class Effect:
    """Represents a single effect with its parameters"""
    def __init__(self, category, name, function, params=None):
        self.category = category
        self.name = name
        self.function = function
        self.enabled = False  # In pipeline or not
        self.active = True    # Active when in pipeline (checkbox state)
        self.params = params or []

    def apply(self, frame, history=None): # <-- Note the new 'history=None' argument
        """Apply effect to frame with current parameters"""
        if not self.enabled or not self.active:
            return frame
        
        try:
            # Build kwargs from parameters (sliders)
            kwargs = {p.name: p.value for p in self.params}
            
            # --- Smartly build arguments based on function signature ---
            sig = inspect.signature(self.function)
            param_names = list(sig.parameters.keys())
            
            args_to_pass = []
            
            # 1. First argument is always the current frame
            if len(param_names) > 0:
                args_to_pass.append(frame)
                
            # 2. Check if it needs a 'frame2' and if history is available
            # We specifically look for a parameter named 'frame2'
            if 'frame2' in param_names and history and len(history) > 0:
                
                # Check if 'frame2' is the second parameter (index 1)
                frame2_idx = param_names.index('frame2')
                if frame2_idx == 1:
                    oldest_frame = history[0]
                    
                    # --- CRITICAL: Ensure frame dimensions match ---
                    h, w = frame.shape[:2]
                    old_h, old_w = oldest_frame.shape[:2]
                    
                    if (h, w) != (old_h, old_w):
                        # Resize oldest frame to match current frame
                        oldest_frame_resized = cv2.resize(oldest_frame, (w, h), interpolation=cv2.INTER_AREA)
                        args_to_pass.append(oldest_frame_resized)
                    else:
                        args_to_pass.append(oldest_frame)

            # Call the function with positional args and slider kwargs
            return self.function(*args_to_pass, **kwargs)
        
        except Exception as e:
            print(f"Error applying {self.name}: {e}")
            # On error, return the original frame to avoid crashing the pipeline
            return frame

#     def apply(self, frame):
#         """Apply effect to frame with current parameters"""
#         if not self.enabled or not self.active:
#             return frame
        
#         try:
#             # Build kwargs from parameters
#             kwargs = {p.name: p.value for p in self.params}
#             return self.function(frame, **kwargs)
#         except Exception as e:
#             print(f"Error applying {self.name}: {e}")
#             return frame


class EffectsGUI:
    """Main GUI application"""
    
    def __init__(self, camera_id=0, image_path=None):
        self.root = tk.Tk()
        self.root.title("Live Effects GUI")
        self.root.geometry("1400x900")
        
        # Video source
        self.camera_id = camera_id
        self.image_path = image_path
        self.cap = None
        self.capture_thread = None # Keep track of the thread
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Effects pipeline
        self.effects = self.build_effects_list()
        self.history = []
        self.max_history = 100
        self.last_processed_frame = None
        
        # GUI components
        self.setup_gui()
        
        # Start video thread
        if image_path:
            self.load_image()
        else:
            self.start_camera()
        
        self.update_preview()
    
    def build_effects_list(self):
        """Build list of all available effects"""
        effects = []
        
        if USE_AUTO_CONFIG:
            # Use auto-generated config
            for effect_data in EFFECTS:
                name, func, params_data = effect_data
                
                # Determine category from function
                category = 'unknown'
                if hasattr(func, '__self__'):
                    category = func.__self__.__class__.__name__
                else:
                    # Check which class it belongs to
                    for cat_name in ['process', 'draw', 'blend', 'analyze']:
                        if name in CATEGORIES.get(cat_name, []):
                            category = cat_name
                            break
                
                # Build parameters
                params = []
                for p in params_data:
                    param = EffectParameter(
                        name=p['name'],
                        value=p['default'],
                        min_val=p.get('min'),
                        max_val=p.get('max')
                    )
                    params.append(param)
                
                effects.append(Effect(category, name, func, params))
        else:
            # Fallback to manual list
            process_effects = [
                ('grayscale', process.grayscale, []),
                ('invert', process.invert, []),
                ('brighten', process.brighten, [
                    EffectParameter('value', 50, 0, 255)
                ]),
                ('contrast', process.contrast, [
                    EffectParameter('alpha', 1.5, 0.1, 3.0)
                ]),
                ('blur_gaussian', process.blur_gaussian, [
                    EffectParameter('kernel_size', 5, 1, 51)
                ]),
                ('blur_median', process.blur_median, [
                    EffectParameter('kernel_size', 15, 3, 51)
                ]),
                ('edge_canny', process.edge_canny, [
                    EffectParameter('threshold1', 50, 0, 255),
                    EffectParameter('threshold2', 150, 0, 255)
                ]),
                ('pixelate', process.pixelate, [
                    EffectParameter('pixel_size', 10, 2, 50)
                ]),
                ('glitch', process.glitch, [
                    EffectParameter('intensity', 20, 1, 100)
                ]),
                ('vignette', process.vignette, [
                    EffectParameter('intensity', 0.5, 0.0, 1.0)
                ]),
                ('scanlines', process.scanlines, [
                    EffectParameter('line_spacing', 2, 1, 10),
                    EffectParameter('line_intensity', 0.5, 0.0, 1.0)
                ]),
                ('chromatic_aberration', process.chromatic_aberration, [
                    EffectParameter('offset', 5, 0, 20)
                ]),
                ('shift_hue', process.shift_hue, [
                    EffectParameter('shift', 90, 0, 180)
                ]),
                ('posterize', process.posterize, [
                    EffectParameter('levels', 4, 2, 8)
                ]),
            ]
            
            for name, func, params in process_effects:
                effects.append(Effect('process', name, func, params))
            
            # Draw effects
            draw_effects = [
                ('grid', draw.grid, [
                    EffectParameter('spacing', 50, 10, 200)
                ]),
                ('crosshair', draw.crosshair, [
                    EffectParameter('size', 20, 5, 100)
                ]),
            ]
            
            for name, func, params in draw_effects:
                effects.append(Effect('draw', name, func, params))
        
        return effects
    
    def setup_gui(self):
        """Setup GUI layout"""
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left section - Available effects
        left_panel = tk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        
        left_title = tk.Label(left_panel, text="Available Effects", 
                             font=("Arial", 14, "bold"))
        left_title.pack(pady=(0, 10))
        
        # Search box
        search_frame = tk.Frame(left_panel)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        # old self.search_var.trace('w', lambda *args: self.filter_available_effects())
        self.search_var.trace_add('write', lambda *args: self.filter_available_effects())
        search_entry = tk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Scrollable available effects list
        left_canvas_frame = tk.Frame(left_panel)
        left_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        left_canvas = tk.Canvas(left_canvas_frame, bg='white')
        left_scrollbar = ttk.Scrollbar(left_canvas_frame, orient=tk.VERTICAL, 
                                       command=left_canvas.yview)
        self.available_frame = tk.Frame(left_canvas, bg='white')
        
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        left_canvas_window = left_canvas.create_window((0, 0), window=self.available_frame, 
                                                       anchor=tk.NW)
        
        def configure_left_scroll(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfig(left_canvas_window, width=event.width)
        
        self.available_frame.bind("<Configure>", configure_left_scroll)
        left_canvas.bind("<Configure>", configure_left_scroll)
        
        # Middle section - Pipeline controls
        middle_panel = tk.Frame(main_container, width=400)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        middle_title = tk.Label(middle_panel, text="Active Pipeline", 
                               font=("Arial", 14, "bold"))
        middle_title.pack(pady=(0, 10))
        
        # Pipeline buttons
        button_frame = tk.Frame(middle_panel)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(button_frame, text="Clear All", 
                 command=self.clear_pipeline).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Export Code", 
                 command=self.export_code).pack(side=tk.LEFT, padx=2)
        
        # <-- NEW BUTTON -->
        tk.Button(button_frame, text="Add 5 Random", 
                 command=self.add_random_effects).pack(side=tk.LEFT, padx=2)
        # <-- END NEW BUTTON -->
        
        # Scrollable pipeline list
        middle_canvas_frame = tk.Frame(middle_panel)
        middle_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        middle_canvas = tk.Canvas(middle_canvas_frame, bg='#f0f0f0')
        middle_scrollbar = ttk.Scrollbar(middle_canvas_frame, orient=tk.VERTICAL, 
                                         command=middle_canvas.yview)
        self.pipeline_frame = tk.Frame(middle_canvas, bg='#f0f0f0')
        
        middle_canvas.configure(yscrollcommand=middle_scrollbar.set)
        middle_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        middle_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        middle_canvas_window = middle_canvas.create_window((0, 0), window=self.pipeline_frame, 
                                                           anchor=tk.NW)
        
        def configure_middle_scroll(event):
            middle_canvas.configure(scrollregion=middle_canvas.bbox("all"))
            middle_canvas.itemconfig(middle_canvas_window, width=event.width)
        
        self.pipeline_frame.bind("<Configure>", configure_middle_scroll)
        middle_canvas.bind("<Configure>", configure_middle_scroll)
        
        # Right panel - Video preview
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        preview_title_frame = tk.Frame(right_panel)
        preview_title_frame.pack(fill=tk.X, pady=(0, 10))
        
        preview_title = tk.Label(preview_title_frame, text="Live Preview", 
                                font=("Arial", 14, "bold"))
        preview_title.pack(side=tk.LEFT)
        
        # Pack buttons from right-to-left
        tk.Button(preview_title_frame, text="Save Image", 
                 command=self.save_image).pack(side=tk.RIGHT)
        
        tk.Button(preview_title_frame, text="Load Image", 
                 command=self.load_image_dialog).pack(side=tk.RIGHT, padx=5)
        
        tk.Button(preview_title_frame, text="Camera", 
                 command=self.start_camera_feed).pack(side=tk.RIGHT)
        
        self.preview_label = tk.Label(right_panel, bg='black')
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Build both lists
        self.build_available_effects()
        self.build_pipeline_effects()
    
    def build_available_effects(self):
        """Build list of available effects to add"""
        # Clear existing
        for widget in self.available_frame.winfo_children():
            widget.destroy()
        
        search_term = self.search_var.get().lower() if hasattr(self, 'search_var') else ''
        
        current_category = None
        for effect in self.effects:
            # Filter by search
            if search_term and search_term not in effect.name.lower():
                continue
            
            # Category header
            if effect.category != current_category:
                current_category = effect.category
                header = tk.Label(self.available_frame, 
                                text=f"═══ {effect.category.upper()} ═══",
                                font=("Arial", 10, "bold"),
                                bg='#e0e0e0', fg='#333')
                header.pack(fill=tk.X, pady=(5, 2))
            
            # Effect item
            item_frame = tk.Frame(self.available_frame, bg='white', 
                                 relief=tk.RAISED, bd=1)
            item_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Effect name
            name_label = tk.Label(item_frame, text=effect.name, 
                                 bg='white', fg='black', font=("Arial", 10),
                                 anchor=tk.W)
            name_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
            
            # Add to pipeline button
            def make_add_callback(eff):
                def add():
                    eff.enabled = True
                    self.build_pipeline_effects()
                return add
            
            add_btn = tk.Button(item_frame, text="→", font=("Arial", 12, "bold"),
                              command=make_add_callback(effect),
                              bg='#4CAF50', fg='white', width=3,
                              relief=tk.RAISED)
            add_btn.pack(side=tk.RIGHT, padx=2, pady=2)
    
    def filter_available_effects(self):
        """Filter available effects by search term"""
        self.build_available_effects()
    
    def build_pipeline_effects(self):
        """Build list of effects in active pipeline"""
        # Clear existing
        for widget in self.pipeline_frame.winfo_children():
            widget.destroy()
        
        # Get enabled effects only
        enabled_effects = [(i, eff) for i, eff in enumerate(self.effects) if eff.enabled]
        
        if not enabled_effects:
            # Show empty message
            empty_label = tk.Label(self.pipeline_frame, 
                                  text="No effects in pipeline\n\nAdd effects from the left →",
                                  font=("Arial", 11), fg='#666',
                                  bg='#f0f0f0', pady=20)
            empty_label.pack(fill=tk.BOTH, expand=True)
            return
        
        for pipeline_idx, (global_idx, effect) in enumerate(enabled_effects):
            # Effect frame
            effect_frame = tk.Frame(self.pipeline_frame, bg='white', 
                                   relief=tk.RIDGE, bd=2)
            effect_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Top row: checkbox, name, and controls
            top_row = tk.Frame(effect_frame, bg='white')
            top_row.pack(fill=tk.X, padx=5, pady=5)
            
            # Checkbox for enable/disable (keep in pipeline)
            var = tk.BooleanVar(value=effect.active if hasattr(effect, 'active') else True)
            
            def make_toggle(eff, v):
                def toggle():
                    # Toggle active state but keep in pipeline
                    eff.active = v.get()
                return toggle
            
            cb = tk.Checkbutton(top_row, text="", variable=var,
                              command=make_toggle(effect, var),
                              bg='white')
            cb.pack(side=tk.LEFT)
            
            # Effect name and category
            name_text = f"{effect.name} ({effect.category})"
            name_label = tk.Label(top_row, text=name_text,
                                 bg='white', fg='black', font=("Arial", 10, "bold"))
            name_label.pack(side=tk.LEFT, padx=5)
            
            # Control buttons frame
            controls_frame = tk.Frame(top_row, bg='white')
            controls_frame.pack(side=tk.RIGHT)
            
            # Up button
            def make_move_up(idx):
                def move_up():
                    if idx > 0:
                        self.effects[global_idx], self.effects[self.get_global_idx(enabled_effects, idx - 1)] = \
                            self.effects[self.get_global_idx(enabled_effects, idx - 1)], self.effects[global_idx]
                        self.build_pipeline_effects()
                return move_up
            
            up_btn = tk.Button(controls_frame, text="▲", font=("Arial", 8),
                             command=make_move_up(pipeline_idx),
                             width=2, bg='#d0d0d0')
            up_btn.pack(side=tk.LEFT, padx=1)
            if pipeline_idx == 0:
                up_btn.config(state=tk.DISABLED)
            
            # Down button
            def make_move_down(idx):
                def move_down():
                    if idx < len(enabled_effects) - 1:
                        self.effects[global_idx], self.effects[self.get_global_idx(enabled_effects, idx + 1)] = \
                            self.effects[self.get_global_idx(enabled_effects, idx + 1)], self.effects[global_idx]
                        self.build_pipeline_effects()
                return move_down
            
            down_btn = tk.Button(controls_frame, text="▼", font=("Arial", 8),
                               command=make_move_down(pipeline_idx),
                               width=2, bg='#d0d0d0')
            down_btn.pack(side=tk.LEFT, padx=1)
            if pipeline_idx == len(enabled_effects) - 1:
                down_btn.config(state=tk.DISABLED)
            
            # Delete button
            def make_delete(eff):
                def delete():
                    eff.enabled = False
                    eff.active = True  # Reset active state
                    self.build_pipeline_effects()
                return delete
            
            del_btn = tk.Button(controls_frame, text="X", font=("Arial", 10, "bold"),
                              command=make_delete(effect),
                              width=2, bg='#f44336', fg='white')
            del_btn.pack(side=tk.LEFT, padx=3)
            
            # Parameters
            if effect.params:
                params_container = tk.Frame(effect_frame, bg='#f9f9f9')
                params_container.pack(fill=tk.X, padx=5, pady=(0, 5))
                
                for param in effect.params:
                    param_frame = tk.Frame(params_container, bg='#f9f9f9')
                    param_frame.pack(fill=tk.X, padx=10, pady=3)
                    
                    tk.Label(param_frame, text=f"{param.name}:", 
                            bg='#f9f9f9', fg='black', font=("Arial", 9)).pack(side=tk.LEFT)
                    
                    if param.param_type in (int, float):
                        # Value label
                        value_label = tk.Label(param_frame, 
                                             text=f"{param.value:.2f}" if param.param_type == float else f"{param.value}",
                                             bg='#f9f9f9', fg='black', font=("Arial", 9), width=8)
                        value_label.pack(side=tk.RIGHT, padx=5)
                        
                        # Slider
                        def make_slider_callback(p, lbl):
                            def callback(val):
                                if p.param_type == int:
                                    p.value = int(float(val))
                                    lbl.config(text=f"{p.value}")
                                else:
                                    p.value = float(val)
                                    lbl.config(text=f"{p.value:.2f}")
                            return callback
                        
                        slider = tk.Scale(param_frame, from_=param.min_val, 
                                        to=param.max_val, orient=tk.HORIZONTAL,
                                        resolution=0.1 if param.param_type == float else 1,
                                        command=make_slider_callback(param, value_label),
                                        showvalue=False, bg='#f9f9f9',
                                        highlightthickness=0)
                        slider.set(param.value)
                        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def get_global_idx(self, enabled_effects, pipeline_idx):
        """Get global effect index from pipeline index"""
        return enabled_effects[pipeline_idx][0]
    
    def clear_pipeline(self):
        """Clear all effects from pipeline"""
        for effect in self.effects:
            effect.enabled = False
        self.build_pipeline_effects()
    
    def add_random_effects(self):
        """Add 5 random, currently disabled effects to the pipeline."""
        
        # 1. Find effects that are not already enabled
        available_to_add = [effect for effect in self.effects if not effect.enabled]
        
        if not available_to_add:
            print("No more effects to add.")
            return

        # 2. Determine how many to add (up to 5)
        num_to_add = min(5, len(available_to_add))
        
        # 3. Select a random sample
        random_effects = random.sample(available_to_add, num_to_add)
        
        # 4. Enable them
        print(f"Adding {num_to_add} random effects:")
        for effect in random_effects:
            effect.enabled = True
            effect.active = True  # Ensure it's active when added
            print(f"  - {effect.name}")
            
        # 5. Rebuild the pipeline GUI
        self.build_pipeline_effects()
    
    def rebuild_effects_widgets(self):
        """Rebuild both effect lists"""
        self.build_available_effects()
        self.build_pipeline_effects()
    
    def start_camera(self):
        """Start camera capture thread"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        def capture_loop():
            while self.running:
                if not self.cap:
                    break
                ret, frame = self.cap.read()
                if ret:
                    # Convert BGR to BGRA
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    
                    # Add to queue (drop old frames if full)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                else:
                    # Handle camera disconnection
                    print("Camera feed lost. Waiting...")
                    threading.Event().wait(1.0)
        
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
    
    def load_image(self):
        """Load static image"""
        if self.image_path and Path(self.image_path).exists():
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
            # Put image in queue repeatedly
            def image_loop():
                while self.running:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self.frame_queue.put_nowait(img.copy())
                    except queue.Full:
                        pass
                    threading.Event().wait(0.033)  # ~30 fps
            
            self.capture_thread = threading.Thread(target=image_loop, daemon=True)
            self.capture_thread.start()
    
    def load_image_dialog(self):
        """Open file dialog to load image"""
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if filepath:
            self.image_path = filepath
            
            # Stop current thread
            self.running = False
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join()
            
            # Release camera if it was active
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.running = True
            
            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.load_image()

    def start_camera_feed(self):
        """Switch the video source back to the webcam."""
        print("Switching to camera feed...")
        
        # 1. Stop the current capture thread
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()
        
        # 2. Clean up resources (camera is released by join, just in case)
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 3. Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        # 4. Reset state
        self.image_path = None 
        
        # 5. Start the camera
        self.running = True
        self.start_camera()

    def save_image(self):
        """Save the current processed frame to the 'output' folder"""
        if self.last_processed_frame is None:
            print("No frame processed yet to save.")
            return
        
        try:
            # Create output directory
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # Milliseconds
            filename = f"saved_frame_{timestamp}.png"
            save_path = output_dir / filename
            
            # Save the frame
            cv2.imwrite(str(save_path), self.last_processed_frame)
            
            print(f"✓ Frame saved to: {save_path}")
        
        except Exception as e:
            print(f"Error saving frame: {e}")
    
    def update_preview(self):
        """Update video preview with effects applied"""
        if not self.running:
            return
        
        try:
            # Get frame from queue
            frame = self.frame_queue.get_nowait()
            
            # Apply effects pipeline
        #     for effect in self.effects:
        #         if effect.enabled:
        #             frame = effect.apply(frame)
            # Apply effects pipeline
            for effect in self.effects:
                if effect.enabled:
                        frame = effect.apply(frame, self.history)
            # Store the processed frame (BGRA) for saving
            self.last_processed_frame = frame.copy()
            
            # Update history
            #self.history.append(self.last_processed_frame.copy())
            self.history.append(frame)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Convert to RGB for display
            display_frame = cv2.cvtColor(self.last_processed_frame, cv2.COLOR_BGRA2RGB)
            
            # Resize to fit preview
            h, w = display_frame.shape[:2]
            preview_width = 800
            preview_height = int(h * preview_width / w)
            display_frame = cv2.resize(display_frame, (preview_width, preview_height))
            
            # Convert to PhotoImage
            from PIL import Image, ImageTk
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
            
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(33, self.update_preview)  # ~30 fps
    
    def export_code(self):
        """Export current pipeline as Python code"""
        lines = []
        lines.append('"""')
        lines.append('Live Coder by Tuomo Rainio 2025')
        lines.append('Generated by Effects GUI')
        lines.append('"""')
        lines.append('from src.live_code_libs import *')
        lines.append('from src.graphics import *')
        lines.append('')
        lines.append('def processing(frame, gui=None, history=None):')
        lines.append('    """Add your code here"""')
        
        # Add enabled AND active effects
        for effect in self.effects:
            if effect.enabled and effect.active:
                # Build function call
                params = []
                for param in effect.params:
                    params.append(f"{param.name}={param.value}")
                
                params_str = ", ".join(params)
                if params_str:
                    line = f"    frame = {effect.category}.{effect.name}(frame, {params_str})"
                else:
                    line = f"    frame = {effect.category}.{effect.name}(frame)"
                
                lines.append(line)
            elif effect.enabled and not effect.active:
                # Add as comment if disabled
                params = []
                for param in effect.params:
                    params.append(f"{param.name}={param.value}")
                
                params_str = ", ".join(params)
                if params_str:
                    line = f"    # frame = {effect.category}.{effect.name}(frame, {params_str})"
                else:
                    line = f"    # frame = {effect.category}.{effect.name}(frame)"
                
                lines.append(line)
        
        lines.append('    return frame')
        lines.append('')
        
        code = '\n'.join(lines)
        
        # Save to file
        filepath = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py")],
            initialfile="live_generated.py"
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(code)
            print(f"Exported to: {filepath}")
    
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        """Cleanup on close"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Effects GUI")
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to static image (instead of camera)')
    
    args = parser.parse_args()
    
    app = EffectsGUI(camera_id=args.camera, image_path=args.image)
    app.run()