"""
Live Coder by Tuomo Rainio 2025

"""

"""THIS IMPORTS ALL EXTERNAL LIBS"""
from src.live_code_libs import *
from src.graphics import *

def processing(frame, gui=None, history=None):
    """Add your code here"""
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = process.grayscale(frame)
    # frame = process.invert(frame)
    # frame = process.scanlines(frame, line_spacing=2, line_intensity=0.5) 
    # frame = process.vignette(frame, intensity=0.5) 
    #frame = blend.history_diff_sum(history)     
    #frame = history[0] # 100 Frames back
    #frame = history[10] # 90 Frames back
    #frame = history[90] # 10 Frames back
    #frame = history[10] # most recent frame
    
    #noise = generate.noise_gaussian(frame)
    #frame = blend.add(noise, history[gui.get('Blur')])

    #frame = process.stylize(frame)

    frame = process.grayscale(frame)


    return frame