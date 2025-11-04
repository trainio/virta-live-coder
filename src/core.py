# import tkinter as tk
# from tkinter import ttk

# class LiveGUI:
#     def __init__(self):
#         self.params = {}
#         self.control_defs = []
#         self.control_names = set()
#         self.window_open = False
#         self.closed = False
#         self.row = 0
#         self.create_window()
        
#     def create_window(self):
#         """Create or recreate the window"""
#         self.root = tk.Tk()
#         self.root.title("Live Controls")
#         self.root.geometry("300x400")
#         self.root.protocol("WM_DELETE_WINDOW", self.on_close)
#         self.row = 0
#         self.closed = False
#         self.window_open = True
        
#         # Rebuild all controls
#         for control in self.control_defs:
#             control['func'](*control['args'], **control['kwargs'])
    
#     def on_close(self):
#         """Handle window close"""
#         self.closed = True
#         self.window_open = False
#         self.root.destroy()
        
#     def slider(self, name, min_val, max_val, default, step=1):
#         """Add a slider control"""
#         self.control_names.add(name)
#         self.control_defs.append({
#             'func': self._create_slider,
#             'args': (name, min_val, max_val, default, step),
#             'kwargs': {}
#         })
        
#         if self.window_open and not self.closed:
#             self._create_slider(name, min_val, max_val, default, step)
#         return self
    
#     def _create_slider(self, name, min_val, max_val, default, step):
#         """Internal slider creation"""
#         frame = ttk.Frame(self.root, padding="5")
#         frame.grid(row=self.row, column=0, sticky="ew")
#         self.row += 1
        
#         if name not in self.params:
#             self.params[name] = default
        
#         label = ttk.Label(frame, text=f"{name}: {self.params[name]}")
#         label.pack()
        
#         def on_change(val):
#             value = float(val)
#             if step >= 1:
#                 value = int(value)
#             self.params[name] = value
#             label.config(text=f"{name}: {value}")
        
#         scale = ttk.Scale(frame, from_=min_val, to=max_val, 
#                          orient="horizontal", command=on_change)
#         scale.set(self.params[name])
#         scale.pack(fill="x")
    
#     def button(self, name, default=False):
#         """Add a toggle button"""
#         self.control_names.add(name)
#         self.control_defs.append({
#             'func': self._create_button,
#             'args': (name, default),
#             'kwargs': {}
#         })
        
#         if self.window_open and not self.closed:
#             self._create_button(name, default)
#         return self
    
#     def _create_button(self, name, default):
#         """Internal button creation"""
#         frame = ttk.Frame(self.root, padding="5")
#         frame.grid(row=self.row, column=0, sticky="ew")
#         self.row += 1
        
#         if name not in self.params:
#             self.params[name] = default
        
#         var = tk.BooleanVar(value=self.params[name])
        
#         def toggle():
#             self.params[name] = var.get()
        
#         check = ttk.Checkbutton(frame, text=name, variable=var, command=toggle)
#         check.pack()
    
#     def get(self, name):
#         """Get parameter value"""
#         return self.params.get(name)
    
#     def update(self):
#         """Call this in your main loop"""
#         try:
#             if self.closed:
#                 self.create_window()
#             self.root.update()
#         except tk.TclError:
#             self.closed = True
#             self.create_window()

# # Global instance
# _gui_instance = LiveGUI()

# def get_gui():
#     """Get the global GUI instance"""
#     return _gui_instance