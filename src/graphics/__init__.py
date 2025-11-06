"""
Graphics Module
Organized image processing, drawing, generation, and blending effects
"""

from .process import process
from .draw import draw
from .generate import generate
from .blend import blend
from .analyze import analyze

# Backwards compatibility - map old function names to new class methods

# Process functions
grayscale = process.grayscale
invert = process.invert
brighten = process.brighten
contrast = process.contrast
blur_gaussian = process.blur_gaussian
blur_median = process.blur_median
blur_box = process.blur_box
blur_bilateral = process.blur_bilateral
edge_canny = process.edge_canny
edge_laplacian = process.edge_laplacian
edge_sobel = process.edge_sobel
threshold_binary = process.threshold_binary
threshold_adaptive = process.threshold_adaptive
erode = process.erode
dilate = process.dilate
morph_open = process.morph_open
morph_close = process.morph_close
flip_horizontal = process.flip_horizontal
flip_vertical = process.flip_vertical
rotate_90 = process.rotate_90
rotate_angle = process.rotate_angle
adjust_contrast_brightness = process.adjust_contrast_brightness
equalize_histogram = process.equalize_histogram
stylize = process.stylize
pencil_sketch = process.pencil_sketch
edge_preserving_filter = process.edge_preserving_filter
convert_hsv = process.convert_hsv
shift_hue = process.shift_hue
remove_channel = process.remove_channel
posterize = process.posterize
pixelate = process.pixelate
glitch = process.glitch
channel_threshold = process.channel_threshold
scanlines = process.scanlines
chromatic_aberration = process.chromatic_aberration
vignette = process.vignette
datamosh = process.datamosh
add_noise_uniform = process.add_noise_uniform
add_noise_gaussian = process.add_noise_gaussian
invert_channels = process.invert_channels
swap_channels = process.swap_channels
edge_masked = process.edge_masked

# Draw functions
draw_rectangle = draw.rectangle
draw_circle = draw.circle
draw_line = draw.line
draw_crosshair = draw.crosshair
draw_polygon = draw.polygon
draw_ellipse = draw.ellipse
draw_text = draw.text
draw_grid = draw.grid
colorize_edges = draw.colorize_edges

# Blend functions - two frame operations
blend_alpha = blend.alpha
blend_add = blend.add
blend_subtract = blend.subtract
blend_difference = blend.difference
blend_multiply = blend.multiply
blend_screen = blend.screen
blend_overlay = blend.overlay
blend_lighten = blend.lighten
blend_darken = blend.darken
blend_average = blend.average
blend_max = blend.max
blend_min = blend.min

# Blend functions - history operations
blend_history_average = blend.history_average
blend_history_weighted = blend.history_weighted
blend_history_max = blend.history_max
blend_history_min = blend.history_min
blend_history_median = blend.history_median
blend_history_diff_sum = blend.history_diff_sum
blend_history_trail = blend.history_trail
blend_history_onion_skin = blend.history_onion_skin
blend_history_echo = blend.history_echo

# Legacy name
frame_diff = blend.difference

# Analyze
analyze_digram_visualize = analyze.diagram_visualize

# Export everything
__all__ = [
    # Classes
    'process', 'draw', 'generate', 'blend','analyze',
    
    # Process functions (backwards compatibility)
    'grayscale', 'invert', 'brighten', 'contrast',
    'blur_gaussian', 'blur_median', 'blur_box', 'blur_bilateral',
    'edge_canny', 'edge_laplacian', 'edge_sobel',
    'threshold_binary', 'threshold_adaptive',
    'erode', 'dilate', 'morph_open', 'morph_close',
    'flip_horizontal', 'flip_vertical', 'rotate_90', 'rotate_angle',
    'adjust_contrast_brightness', 'equalize_histogram',
    'stylize', 'pencil_sketch', 'edge_preserving_filter',
    'convert_hsv', 'shift_hue', 'remove_channel',
    'posterize', 'pixelate', 'glitch', 'channel_threshold',
    'scanlines', 'chromatic_aberration', 'vignette', 'datamosh',
    'add_noise_uniform', 'add_noise_gaussian',
    'invert_channels', 'swap_channels', 'edge_masked',
    
    # Draw functions (backwards compatibility)
    'draw_rectangle', 'draw_circle', 'draw_line', 'draw_crosshair',
    'draw_polygon', 'draw_ellipse', 'draw_text', 'draw_grid',
    'colorize_edges',
    
    # Blend functions (backwards compatibility)
    'blend_alpha', 'blend_add', 'blend_subtract', 'blend_difference',
    'blend_multiply', 'blend_screen', 'blend_overlay',
    'blend_lighten', 'blend_darken', 'blend_average', 'blend_max', 'blend_min',
    'blend_history_average', 'blend_history_weighted',
    'blend_history_max', 'blend_history_min', 'blend_history_median',
    'blend_history_diff_sum', 'blend_history_trail',
    'blend_history_onion_skin', 'blend_history_echo',
    'frame_diff',

    # Analyze functions
    'analyze_digram_visualize'
]