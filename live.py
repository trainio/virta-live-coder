"""
Live Coder by Tuomo Rainio 2025

"""

"""THIS IMPORTS ALL EXTERNAL LIBS"""
from src.live_code_libs import *

def process(frame, gui=None, history=None):
    """Add your code here"""
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #frame = history[0] # 100 Frames back
    #frame = history[10] # 90 Frames back
    #frame = history[90] # 10 Frames back
    frame = history[-99] # most recent frame
    
    return frame