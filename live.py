"""
Live Coder by Tuomo Rainio 2025

"""

"""THIS IMPORTS ALL EXTERNAL LIBS"""
from src.live_code_libs import *
from src.graphics import *

def processing(frame, gui=None, history=None):
    """Add your code here"""
    frame = analyze.diagram_visualize(frame)
    return frame