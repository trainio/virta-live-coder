"""
Live Coder by Tuomo Rainio 2025

"""

# ui.py at root
def setup(gui):
    """Setup UI controls with given gui instance"""
    # Clear GUI on reload (do not change)
    gui.control_defs.clear()
    gui.control_names.clear()
    gui.row = 0
    
    # Define controls here
    # YOU CAN EDIT THE UI HERE!
    gui.slider("Blur", 1, 51, 15, step=2) \
       .slider("Brightness", -100, 100, 0) \
       .slider("Contrast", 0.5, 3.0, 1.0, step=0.1) \
       .button("Flip Horizontal", False) \
       .button("Invert Colors", False)