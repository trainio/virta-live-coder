"""
Live Coder by Tuomo Rainio 2025

"""

"""THIS IMPORTS ALL EXTERNAL LIBS"""
from src.live_code_libs import *
from src.graphics import *

def processing(frame, gui=None, history=None):
    """Add your code here"""
    frame = process.grayscale(frame)
    val = gui.get("History")
    frame2 = draw.circle(frame,cx=val)

    frame = blend.add(frame, frame2)
    #frame = analyze.diagram_visualize(frame)
    return frame