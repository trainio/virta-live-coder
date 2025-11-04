"""
Live Coder Effects Library
Collection of reusable image processing functions

Each function should:
1. Have clear docstring with description and parameters
2. Accept frame and optional parameters
3. Return processed frame
4. Work standalone for documentation generation

"""

# def my_effect(frame, param=10):
#     """
#     Description here
    
#     Args:
#         frame: Input frame
#         param: Parameter description
    
#     Returns:
#         Processed frame
#     """
#     # your code
#     return frame

# # Add to EFFECTS dict
# EFFECTS['my_effect'] = my_effect







import cv2
import numpy as np


def grayscale(frame):
    """
    Convert frame to grayscale
    
    Args:
        frame: Input BGR frame
    
    Returns:
        Grayscale frame (single channel)
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur_gaussian(frame, kernel_size=5, sigma=0):
    """
    Apply Gaussian blur to reduce noise
    
    Args:
        frame: Input frame
        kernel_size: Size of blur kernel (must be odd)
        sigma: Standard deviation (0 = auto)
    
    Returns:
        Blurred frame
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)


def edge_canny(frame, threshold1=50, threshold2=150):
    """
    Detect edges using Canny algorithm
    
    Args:
        frame: Input frame (will be converted to grayscale)
        threshold1: Lower threshold for edge detection
        threshold2: Upper threshold for edge detection
    
    Returns:
        Binary edge map
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, threshold1, threshold2)


def threshold_binary(frame, thresh_value=127):
    """
    Apply binary threshold to grayscale image
    
    Args:
        frame: Input frame
        thresh_value: Threshold value (0-255)
    
    Returns:
        Binary thresholded frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return binary


def invert(frame):
    """
    Invert colors (negative image)
    
    Args:
        frame: Input frame
    
    Returns:
        Inverted frame
    """
    return 255 - frame


def brighten(frame, value=50):
    """
    Increase brightness
    
    Args:
        frame: Input frame
        value: Amount to add to each pixel (0-255)
    
    Returns:
        Brightened frame
    """
    return cv2.add(frame, np.ones(frame.shape, dtype=np.uint8) * value)


def contrast(frame, alpha=1.5):
    """
    Adjust contrast
    
    Args:
        frame: Input frame
        alpha: Contrast multiplier (1.0 = no change, >1 = more contrast)
    
    Returns:
        Contrast-adjusted frame
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)


def erode(frame, kernel_size=3, iterations=1):
    """
    Erode image (shrink bright regions)
    
    Args:
        frame: Input frame
        kernel_size: Size of erosion kernel
        iterations: Number of times to apply erosion
    
    Returns:
        Eroded frame
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(frame, kernel, iterations=iterations)


def dilate(frame, kernel_size=3, iterations=1):
    """
    Dilate image (expand bright regions)
    
    Args:
        frame: Input frame
        kernel_size: Size of dilation kernel
        iterations: Number of times to apply dilation
    
    Returns:
        Dilated frame
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(frame, kernel, iterations=iterations)


def frame_diff(frame1, frame2):
    """
    Calculate absolute difference between two frames
    
    Args:
        frame1: First frame
        frame2: Second frame
    
    Returns:
        Difference frame
    """
    return cv2.absdiff(frame1, frame2)


def blend(frame1, frame2=None, alpha=0.5):
    """
    Blend two frames together
    
    Args:
        frame1: First frame
        frame2: Second frame
        alpha: Weight for frame1 (0.0-1.0)
    
    Returns:
        Blended frame
    """
    if frame2 is None:
        return frame1.copy()
    return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)


def colorize_edges(frame, threshold1=50, threshold2=150, color=(0, 255, 0)):
    """
    Detect edges and colorize them
    
    Args:
        frame: Input frame
        threshold1: Lower Canny threshold
        threshold2: Upper Canny threshold
        color: Edge color in BGR format
    
    Returns:
        Frame with colored edges
    """
    edges = edge_canny(frame, threshold1, threshold2)
    output = np.zeros_like(frame)
    output[edges > 0] = color
    return output

# === BLUR EFFECTS ===

def blur_median(frame, kernel_size=15):
    """
    Apply median blur to reduce noise while preserving edges
    
    Args:
        frame: Input frame
        kernel_size: Size of the kernel (must be odd, larger = more blur)
    
    Returns:
        Median blurred frame
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(frame, kernel_size)


def blur_box(frame, kernel_size=10):
    """
    Apply simple box blur (averaging filter)
    
    Args:
        frame: Input frame
        kernel_size: Size of the blur kernel
    
    Returns:
        Box blurred frame
    """
    return cv2.blur(frame, (kernel_size, kernel_size))


def blur_bilateral(frame, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter - smooths while preserving edges
    
    Args:
        frame: Input frame
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        Bilateral filtered frame
    """
    return cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)


# === EDGE DETECTION ===

def edge_laplacian(frame):
    """
    Detect edges using Laplacian operator
    
    Args:
        frame: Input frame
    
    Returns:
        Edge map showing all edges
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_8U)


def edge_sobel(frame, dx=1, dy=0, ksize=3):
    """
    Detect edges using Sobel operator
    
    Args:
        frame: Input frame
        dx: Order of derivative in x direction (0 or 1)
        dy: Order of derivative in y direction (0 or 1)
        ksize: Size of extended Sobel kernel (1, 3, 5, or 7)
    
    Returns:
        Edge map showing directional edges
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Sobel(gray, cv2.CV_8U, dx, dy, ksize=ksize)


# === COLOR EFFECTS ===

def convert_hsv(frame):
    """
    Convert frame to HSV color space
    
    Args:
        frame: Input BGR frame
    
    Returns:
        Frame in HSV color space
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def remove_channel(frame, channel=0):
    """
    Remove a specific color channel
    
    Args:
        frame: Input frame
        channel: Channel to remove (0=Blue, 1=Green, 2=Red)
    
    Returns:
        Frame with channel removed (set to 0)
    """
    result = frame.copy()
    result[:, :, channel] = 0
    return result


# === THRESHOLDING ===

def threshold_adaptive(frame, block_size=11, c=2):
    """
    Apply adaptive threshold - adjusts threshold locally
    
    Args:
        frame: Input frame
        block_size: Size of pixel neighborhood (must be odd)
        c: Constant subtracted from mean
    
    Returns:
        Binary thresholded frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, block_size, c)


# === MORPHOLOGICAL ===

def morph_open(frame, kernel_size=5, iterations=1):
    """
    Apply morphological opening (erosion then dilation)
    Removes small bright spots
    
    Args:
        frame: Input frame
        kernel_size: Size of structuring element
        iterations: Number of times to apply
    
    Returns:
        Morphologically opened frame
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(frame, kernel_size=5, iterations=1):
    """
    Apply morphological closing (dilation then erosion)
    Fills small dark holes
    
    Args:
        frame: Input frame
        kernel_size: Size of structuring element
        iterations: Number of times to apply
    
    Returns:
        Morphologically closed frame
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# === TRANSFORMATIONS ===

def flip_horizontal(frame):
    """
    Flip frame horizontally (mirror effect)
    
    Args:
        frame: Input frame
    
    Returns:
        Horizontally flipped frame
    """
    return cv2.flip(frame, 1)


def flip_vertical(frame):
    """
    Flip frame vertically
    
    Args:
        frame: Input frame
    
    Returns:
        Vertically flipped frame
    """
    return cv2.flip(frame, 0)


def rotate_90(frame):
    """
    Rotate frame 90 degrees clockwise
    
    Args:
        frame: Input frame
    
    Returns:
        Rotated frame
    """
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def rotate_angle(frame, angle=45):
    """
    Rotate frame by arbitrary angle
    
    Args:
        frame: Input frame
        angle: Rotation angle in degrees (positive = clockwise)
    
    Returns:
        Rotated frame
    """
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


# === CONTRAST/BRIGHTNESS ===

def adjust_contrast_brightness(frame, contrast=1.5, brightness=30):
    """
    Adjust contrast and brightness
    
    Args:
        frame: Input frame
        contrast: Contrast multiplier (1.0 = no change, >1 = more contrast)
        brightness: Brightness offset (-100 to 100)
    
    Returns:
        Adjusted frame
    """
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


def equalize_histogram(frame):
    """
    Equalize histogram to enhance contrast
    
    Args:
        frame: Input frame
    
    Returns:
        Histogram equalized frame (grayscale)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


# === ARTISTIC EFFECTS ===

def stylize(frame, sigma_s=60, sigma_r=0.6):
    """
    Apply artistic stylization effect
    
    Args:
        frame: Input frame
        sigma_s: Range of spatial kernel
        sigma_r: Range of color-space kernel
    
    Returns:
        Stylized frame
    """
    return cv2.stylization(frame, sigma_s=sigma_s, sigma_r=sigma_r)


def pencil_sketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05):
    """
    Create pencil sketch effect
    
    Args:
        frame: Input frame
        sigma_s: Range of spatial kernel
        sigma_r: Range of color-space kernel
        shade_factor: Amount of shading
    
    Returns:
        Pencil sketch effect (grayscale)
    """
    _, sketch = cv2.pencilSketch(frame, sigma_s=sigma_s, sigma_r=sigma_r, 
                                  shade_factor=shade_factor)
    return sketch


def edge_preserving_filter(frame, sigma_s=60, sigma_r=0.4):
    """
    Apply edge-preserving smoothing filter
    
    Args:
        frame: Input frame
        sigma_s: Range of spatial kernel
        sigma_r: Range of color-space kernel
    
    Returns:
        Smoothed frame with preserved edges
    """
    return cv2.edgePreservingFilter(frame, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)


# === COLOR SPACE FUN ===

def shift_hue(frame, shift=90):
    """
    Shift color hue in HSV space
    
    Args:
        frame: Input frame
        shift: Amount to shift hue (0-180)
    
    Returns:
        Frame with shifted hue
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# === POSTERIZE ===

def posterize(frame, levels=4):
    """
    Reduce number of color levels (posterization effect)
    
    Args:
        frame: Input frame
        levels: Number of color levels per channel (2-8)
    
    Returns:
        Posterized frame
    """
    divisor = 256 // levels
    return (frame // divisor) * divisor


# === PIXELATE ===

def pixelate(frame, pixel_size=10):
    """
    Create pixelated/mosaic effect
    
    Args:
        frame: Input frame
        pixel_size: Size of pixel blocks (larger = more pixelated)
    
    Returns:
        Pixelated frame
    """
    h, w = frame.shape[:2]
    temp = cv2.resize(frame, (w // pixel_size, h // pixel_size), 
                     interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# === NOISE ===

def add_noise_uniform(frame, amount=50):
    """
    Add uniform random noise
    
    Args:
        frame: Input frame
        amount: Maximum noise amplitude
    
    Returns:
        Noisy frame
    """
    noise = np.random.randint(-amount, amount, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_noise_gaussian(frame, mean=0, stddev=25):
    """
    Add Gaussian random noise
    
    Args:
        frame: Input frame
        mean: Mean of noise distribution
        stddev: Standard deviation of noise
    
    Returns:
        Noisy frame
    """
    noise = np.zeros_like(frame, dtype=np.float32)
    cv2.randn(noise, mean, stddev)
    return cv2.add(frame, noise, dtype=cv2.CV_8U)


# === CHANNEL MANIPULATION ===

def invert_channels(frame, b_invert=255, g_invert=255, r_invert=255):
    """
    Invert specific color channels
    
    Args:
        frame: Input frame
        b_invert: Value to subtract from blue (0-255, 255=full invert)
        g_invert: Value to subtract from green
        r_invert: Value to subtract from red
    
    Returns:
        Frame with inverted channels
    """
    b, g, r = cv2.split(frame)
    b = b_invert - b
    g = g_invert - g
    r = r_invert - r
    return cv2.merge([b, g, r])


def swap_channels(frame, order='rbg'):
    """
    Swap color channels
    
    Args:
        frame: Input frame
        order: Channel order string (e.g., 'rbg', 'grb', 'brg')
               b=blue, g=green, r=red
    
    Returns:
        Frame with swapped channels
    """
    b, g, r = cv2.split(frame)
    channel_map = {'b': b, 'g': g, 'r': r}
    
    c1 = channel_map[order[0]]
    c2 = channel_map[order[1]]
    c3 = channel_map[order[2]]
    
    return cv2.merge([c1, c2, c3])


# === COMBINE EFFECTS ===

def edge_masked(frame, threshold1=100, threshold2=200):
    """
    Show original frame only where edges are detected
    
    Args:
        frame: Input frame
        threshold1: Lower Canny threshold
        threshold2: Upper Canny threshold
    
    Returns:
        Frame masked by edge detection
    """
    edges = cv2.Canny(frame, threshold1, threshold2)
    return cv2.bitwise_and(frame, frame, mask=edges)

# === DRAWING FUNCTIONS ===

def draw_rectangle(frame, x1=50, y1=50, x2=200, y2=150, color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle on the frame
    
    Args:
        frame: Input frame
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
        color: Rectangle color in BGR format
        thickness: Line thickness (-1 for filled)
    
    Returns:
        Frame with rectangle drawn
    """
    result = frame.copy()
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    return result


def draw_circle(frame, cx=320, cy=240, radius=50, color=(0, 0, 255), thickness=-1):
    """
    Draw a circle on the frame
    
    Args:
        frame: Input frame
        cx, cy: Center coordinates
        radius: Circle radius in pixels
        color: Circle color in BGR format
        thickness: Line thickness (-1 for filled)
    
    Returns:
        Frame with circle drawn
    """
    result = frame.copy()
    cv2.circle(result, (cx, cy), radius, color, thickness)
    return result


def draw_line(frame, x1=0, y1=0, x2=640, y2=480, color=(255, 255, 255), thickness=2):
    """
    Draw a line on the frame
    
    Args:
        frame: Input frame
        x1, y1: Starting point coordinates
        x2, y2: Ending point coordinates
        color: Line color in BGR format
        thickness: Line thickness in pixels
    
    Returns:
        Frame with line drawn
    """
    result = frame.copy()
    cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result


def draw_crosshair(frame, cx=None, cy=None, size=20, color=(0, 255, 0), thickness=1):
    """
    Draw a crosshair (horizontal + vertical lines intersecting)
    
    Args:
        frame: Input frame
        cx, cy: Center coordinates (None = frame center)
        size: Length of crosshair lines from center
        color: Crosshair color in BGR format
        thickness: Line thickness
    
    Returns:
        Frame with crosshair drawn
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    
    if cx is None:
        cx = w // 2
    if cy is None:
        cy = h // 2
    
    # Horizontal line
    cv2.line(result, (cx - size, cy), (cx + size, cy), color, thickness)
    # Vertical line
    cv2.line(result, (cx, cy - size), (cx, cy + size), color, thickness)
    
    return result


def draw_polygon(frame, points=None, color=(255, 0, 255), thickness=2, filled=False):
    """
    Draw a polygon on the frame
    
    Args:
        frame: Input frame
        points: List of (x, y) tuples for vertices (None = default triangle)
        color: Polygon color in BGR format
        thickness: Line thickness (ignored if filled=True)
        filled: Fill the polygon if True
    
    Returns:
        Frame with polygon drawn
    """
    result = frame.copy()
    
    if points is None:
        # Default triangle
        points = [[320, 100], [250, 200], [390, 200]]
    
    pts = np.array(points, np.int32)
    
    if filled:
        cv2.fillPoly(result, [pts], color)
    else:
        cv2.polylines(result, [pts], True, color, thickness)
    
    return result


def draw_ellipse(frame, cx=320, cy=240, width=100, height=50, angle=0, 
                 start_angle=0, end_angle=360, color=(255, 128, 0), thickness=2):
    """
    Draw an ellipse on the frame
    
    Args:
        frame: Input frame
        cx, cy: Center coordinates
        width: Width of ellipse
        height: Height of ellipse
        angle: Rotation angle in degrees
        start_angle: Starting angle of arc (0-360)
        end_angle: Ending angle of arc (0-360)
        color: Ellipse color in BGR format
        thickness: Line thickness (-1 for filled)
    
    Returns:
        Frame with ellipse drawn
    """
    result = frame.copy()
    cv2.ellipse(result, (cx, cy), (width, height), angle, 
                start_angle, end_angle, color, thickness)
    return result


def draw_text(frame, text="Hello World", x=100, y=100, font_scale=1.0, 
              color=(255, 255, 255), thickness=2, bg_color=None):
    """
    Draw text on the frame
    
    Args:
        frame: Input frame
        text: Text string to draw
        x, y: Bottom-left corner coordinates of text
        font_scale: Font size multiplier
        color: Text color in BGR format
        thickness: Text thickness
        bg_color: Background color (None = no background)
    
    Returns:
        Frame with text drawn
    """
    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw background rectangle if requested
    if bg_color is not None:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(result, (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), bg_color, -1)
    
    cv2.putText(result, text, (x, y), font, font_scale, color, thickness)
    return result


def draw_grid(frame, spacing=50, color=(100, 100, 100), thickness=1):
    """
    Draw a grid overlay on the frame
    
    Args:
        frame: Input frame
        spacing: Distance between grid lines in pixels
        color: Grid color in BGR format
        thickness: Line thickness
    
    Returns:
        Frame with grid overlay
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Vertical lines
    for x in range(0, w, spacing):
        cv2.line(result, (x, 0), (x, h), color, thickness)
    
    # Horizontal lines
    for y in range(0, h, spacing):
        cv2.line(result, (0, y), (w, y), color, thickness)
    
    return result

# === SPECIAL EFFECTS ===

def pixelate(frame, pixel_size=10):
    """
    Create pixelated/mosaic effect by downsampling and upsampling
    
    Args:
        frame: Input frame
        pixel_size: Size of pixel blocks (larger = more pixelated)
    
    Returns:
        Pixelated frame
    """
    h, w = frame.shape[:2]
    temp = cv2.resize(frame, (w//pixel_size, h//pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def glitch(frame, intensity=20):
    """
    Create digital glitch effect by randomly shifting horizontal strips
    
    Args:
        frame: Input frame
        intensity: Number of glitch strips to create
    
    Returns:
        Glitched frame
    """
    result = frame.copy()
    h, w = result.shape[:2]
    
    for _ in range(intensity):
        y = np.random.randint(0, h-10)
        shift = np.random.randint(-50, 50)
        result[y:y+10] = np.roll(result[y:y+10], shift, axis=1)
    
    return result


def channel_threshold(frame, threshold=128):
    """
    Apply threshold to all channels independently
    
    Args:
        frame: Input frame
        threshold: Threshold value (0-255)
    
    Returns:
        Frame with thresholded channels (binary per channel)
    """
    result = frame.copy()
    result[result < threshold] = 0
    result[result >= threshold] = 255
    return result


def scanlines(frame, line_spacing=2, line_intensity=0.5):
    """
    Add CRT/TV scanline effect
    
    Args:
        frame: Input frame
        line_spacing: Pixels between scanlines
        line_intensity: Darkness of lines (0.0-1.0, higher = darker)
    
    Returns:
        Frame with scanline overlay
    """
    result = frame.copy()
    h = result.shape[0]
    
    for y in range(0, h, line_spacing):
        result[y] = (result[y] * (1 - line_intensity)).astype(np.uint8)
    
    return result


def chromatic_aberration(frame, offset=5):
    """
    Create chromatic aberration effect (RGB channel offset)
    
    Args:
        frame: Input frame
        offset: Pixel offset for channel separation
    
    Returns:
        Frame with chromatic aberration
    """
    b, g, r = cv2.split(frame)
    
    # Shift channels in different directions
    b = np.roll(b, -offset, axis=1)  # Shift blue left
    r = np.roll(r, offset, axis=1)   # Shift red right
    
    return cv2.merge([b, g, r])


def vignette(frame, intensity=0.5):
    """
    Add dark vignette effect around edges
    
    Args:
        frame: Input frame
        intensity: Vignette darkness (0.0-1.0)
    
    Returns:
        Frame with vignette effect
    """
    h, w = frame.shape[:2]
    
    # Create radial gradient
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from center
    radius = np.sqrt(X**2 + Y**2)
    
    # Create vignette mask (values from 0 to 1)
    vignette_mask = 1 - np.clip(radius * intensity, 0, 1)
    
    # Apply to all channels
    result = frame.copy()
    for i in range(3):
        result[:, :, i] = (result[:, :, i] * vignette_mask).astype(np.uint8)
    
    return result


def datamosh(frame, block_size=8, corruption=0.3):
    """
    Create datamosh/compression artifact effect
    
    Args:
        frame: Input frame
        block_size: Size of corruption blocks
        corruption: Probability of block corruption (0.0-1.0)
    
    Returns:
        Datamoshed frame
    """
    result = frame.copy()
    h, w = result.shape[:2]
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if np.random.random() < corruption:
                # Shift block content randomly
                shift_y = np.random.randint(-block_size, block_size)
                shift_x = np.random.randint(-block_size, block_size)
                
                src_y = max(0, min(h - block_size, y + shift_y))
                src_x = max(0, min(w - block_size, x + shift_x))
                
                result[y:y+block_size, x:x+block_size] = \
                    result[src_y:src_y+block_size, src_x:src_x+block_size]
    
    return result

# Dictionary mapping function names to functions for easy lookup
EFFECTS = {
    # Basic effects
    'grayscale': grayscale,
    'invert': invert,
    'brighten': brighten,
    'contrast': contrast,
    
    # Blur effects
    'blur_gaussian': blur_gaussian,
    'blur_median': blur_median,
    'blur_box': blur_box,
    'blur_bilateral': blur_bilateral,
    
    # Edge detection
    'edge_canny': edge_canny,
    'edge_laplacian': edge_laplacian,
    'edge_sobel': edge_sobel,
    
    # Thresholding
    'threshold_binary': threshold_binary,
    'threshold_adaptive': threshold_adaptive,
    
    # Morphological operations
    'erode': erode,
    'dilate': dilate,
    'morph_open': morph_open,
    'morph_close': morph_close,
    
    # Transformations
    'flip_horizontal': flip_horizontal,
    'flip_vertical': flip_vertical,
    'rotate_90': rotate_90,
    'rotate_angle': rotate_angle,
    
    # Contrast/Brightness
    'adjust_contrast_brightness': adjust_contrast_brightness,
    'equalize_histogram': equalize_histogram,
    
    # Artistic effects
    'stylize': stylize,
    'pencil_sketch': pencil_sketch,
    'edge_preserving_filter': edge_preserving_filter,
    
    # Color space
    'convert_hsv': convert_hsv,
    'shift_hue': shift_hue,
    'remove_channel': remove_channel,
    
    # Special effects
    'posterize': posterize,
    'pixelate': pixelate,
    
    # Noise
    'add_noise_uniform': add_noise_uniform,
    'add_noise_gaussian': add_noise_gaussian,
    
    # Channel manipulation
    'invert_channels': invert_channels,
    'swap_channels': swap_channels,
    
    # Combined effects
    'frame_diff': frame_diff,
    'blend': blend,
    'colorize_edges': colorize_edges,
    'edge_masked': edge_masked,

    # Drawing functions
    'draw_rectangle': draw_rectangle,
    'draw_circle': draw_circle,
    'draw_line': draw_line,
    'draw_crosshair': draw_crosshair,
    'draw_polygon': draw_polygon,
    'draw_ellipse': draw_ellipse,
    'draw_text': draw_text,
    'draw_grid': draw_grid,

    # Special/Glitch effects
    'pixelate': pixelate,
    'glitch': glitch,
    'channel_threshold': channel_threshold,
    'scanlines': scanlines,
    'chromatic_aberration': chromatic_aberration,
    'vignette': vignette,
    'datamosh': datamosh,
}