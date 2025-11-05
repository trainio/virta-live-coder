
import cv2
import numpy as np


# ============================================================================
# PROCESS - Frame transformations
# ============================================================================

class process:
    """Image processing and transformation effects"""
    
    @staticmethod
    def grayscale(frame):
        """Convert to grayscale (maintains BGRA format)"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def invert(frame):
        """Invert colors (negative)"""
        result = frame.copy()
        result[:, :, :3] = 255 - result[:, :, :3]
        return result
    
    @staticmethod
    def brighten(frame, value=50):
        """Increase brightness"""
        result = frame.copy()
        result[:, :, :3] = cv2.add(result[:, :, :3], 
                                   np.ones(result[:, :, :3].shape, dtype=np.uint8) * value)
        return result
    
    @staticmethod
    def contrast(frame, alpha=1.5):
        """Adjust contrast (alpha: 1.0=no change, >1=more contrast)"""
        result = frame.copy()
        result[:, :, :3] = cv2.convertScaleAbs(result[:, :, :3], alpha=alpha, beta=0)
        return result
    
    @staticmethod
    def adjust_contrast_brightness(frame, contrast=1.5, brightness=30):
        """Adjust both contrast and brightness"""
        result = frame.copy()
        result[:, :, :3] = cv2.convertScaleAbs(result[:, :, :3], alpha=contrast, beta=brightness)
        return result
    
    @staticmethod
    def blur_gaussian(frame, kernel_size=5, sigma=0):
        """Gaussian blur (kernel_size must be odd)"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = frame.copy()
        result[:, :, :3] = cv2.GaussianBlur(result[:, :, :3], (kernel_size, kernel_size), sigma)
        return result
    
    @staticmethod
    def blur_median(frame, kernel_size=15):
        """Median blur - preserves edges"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        result = frame.copy()
        result[:, :, :3] = cv2.medianBlur(result[:, :, :3], kernel_size)
        return result
    
    @staticmethod
    def blur_box(frame, kernel_size=10):
        """Box blur (simple averaging)"""
        result = frame.copy()
        result[:, :, :3] = cv2.blur(result[:, :, :3], (kernel_size, kernel_size))
        return result
    
    @staticmethod
    def blur_bilateral(frame, diameter=9, sigma_color=75, sigma_space=75):
        """Bilateral filter - smooths while preserving edges"""
        result = frame.copy()
        result[:, :, :3] = cv2.bilateralFilter(result[:, :, :3], diameter, sigma_color, sigma_space)
        return result
    
    @staticmethod
    def edge_canny(frame, threshold1=50, threshold2=150):
        """Canny edge detection"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def edge_laplacian(frame):
        """Laplacian edge detection"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_8U)
        bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def edge_sobel(frame, dx=1, dy=0, ksize=3):
        """Sobel edge detection"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_8U, dx, dy, ksize=ksize)
        bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def threshold_binary(frame, thresh_value=127):
        """Binary threshold"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def threshold_adaptive(frame, block_size=11, c=2):
        """Adaptive threshold - adjusts locally"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, block_size, c)
        bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def erode(frame, kernel_size=3, iterations=1):
        """Erode - shrink bright regions"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = frame.copy()
        result[:, :, :3] = cv2.erode(result[:, :, :3], kernel, iterations=iterations)
        return result
    
    @staticmethod
    def dilate(frame, kernel_size=3, iterations=1):
        """Dilate - expand bright regions"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = frame.copy()
        result[:, :, :3] = cv2.dilate(result[:, :, :3], kernel, iterations=iterations)
        return result
    
    @staticmethod
    def morph_open(frame, kernel_size=5, iterations=1):
        """Morphological opening - removes small bright spots"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = frame.copy()
        result[:, :, :3] = cv2.morphologyEx(result[:, :, :3], cv2.MORPH_OPEN, 
                                            kernel, iterations=iterations)
        return result
    
    @staticmethod
    def morph_close(frame, kernel_size=5, iterations=1):
        """Morphological closing - fills small dark holes"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = frame.copy()
        result[:, :, :3] = cv2.morphologyEx(result[:, :, :3], cv2.MORPH_CLOSE, 
                                            kernel, iterations=iterations)
        return result
    
    @staticmethod
    def flip_horizontal(frame):
        """Mirror horizontally"""
        return cv2.flip(frame, 1)
    
    @staticmethod
    def flip_vertical(frame):
        """Mirror vertically"""
        return cv2.flip(frame, 0)
    
    @staticmethod
    def rotate_90(frame):
        """Rotate 90 degrees clockwise"""
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    @staticmethod
    def rotate_angle(frame, angle=45):
        """Rotate by arbitrary angle"""
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))
    
    @staticmethod
    def equalize_histogram(frame):
        """Histogram equalization - enhance contrast"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def stylize(frame, sigma_s=60, sigma_r=0.6):
        """Artistic stylization"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        stylized = cv2.stylization(bgr, sigma_s=sigma_s, sigma_r=sigma_r)
        return cv2.cvtColor(stylized, cv2.COLOR_BGR2BGRA)
        
    @staticmethod
    def pencil_sketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05):
        """Pencil sketch effect"""
        # Check if frame has 4 channels (BGRA), if so convert to BGR
        # if frame.shape[2] == 4:
        #     bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # else:
        #     bgr = frame  # Already BGR

        # # pencilSketch returns (color_sketch, grayscale_sketch)
        # _, sketch = cv2.pencilSketch(bgr, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
        
        # # sketch is already grayscale (1 channel), convert to BGR then BGRA
        # bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        print("ERROR: Function missing!")
        return frame #cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def edge_preserving_filter(frame, sigma_s=60, sigma_r=0.4):
        """Edge-preserving smoothing"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        filtered = cv2.edgePreservingFilter(bgr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def convert_hsv(frame):
        """Convert to HSV color space (visualized as BGR)"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.cvtColor(hsv, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def shift_hue(frame, shift=90):
        """Shift color hue (0-180)"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    
    @staticmethod
    def remove_channel(frame, channel=0):
        """Remove color channel (0=Blue, 1=Green, 2=Red)"""
        result = frame.copy()
        result[:, :, channel] = 0
        return result
    
    @staticmethod
    def posterize(frame, levels=4):
        """Reduce color levels (2-8)"""
        result = frame.copy()
        divisor = 256 // levels
        result[:, :, :3] = (result[:, :, :3] // divisor) * divisor
        return result
    
    @staticmethod
    def pixelate(frame, pixel_size=10):
        """Pixelate/mosaic effect"""
        h, w = frame.shape[:2]
        temp = cv2.resize(frame, (w // pixel_size, h // pixel_size), 
                         interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    @staticmethod
    def glitch(frame, intensity=20):
        """Digital glitch - random horizontal shifts"""
        result = frame.copy()
        h = result.shape[0]
        for _ in range(intensity):
            y = np.random.randint(0, h-10)
            shift = np.random.randint(-50, 50)
            result[y:y+10] = np.roll(result[y:y+10], shift, axis=1)
        return result
    
    @staticmethod
    def channel_threshold(frame, threshold=128):
        """Threshold all channels independently"""
        result = frame.copy()
        result[:, :, :3][result[:, :, :3] < threshold] = 0
        result[:, :, :3][result[:, :, :3] >= threshold] = 255
        return result
    
    @staticmethod
    def scanlines(frame, line_spacing=2, line_intensity=0.5):
        """CRT/TV scanline effect"""
        result = frame.copy()
        h = result.shape[0]
        for y in range(0, h, line_spacing):
            result[y, :, :3] = (result[y, :, :3] * (1 - line_intensity)).astype(np.uint8)
        return result
    
    @staticmethod
    def chromatic_aberration(frame, offset=5):
        """RGB channel offset"""
        result = frame.copy()
        result[:, :, 0] = np.roll(result[:, :, 0], -offset, axis=1)  # Blue left
        result[:, :, 2] = np.roll(result[:, :, 2], offset, axis=1)   # Red right
        return result
    
    @staticmethod
    def vignette(frame, intensity=0.5):
        """Dark vignette around edges"""
        result = frame.copy()
        h, w = result.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X**2 + Y**2)
        vignette_mask = 1 - np.clip(radius * intensity, 0, 1)
        for i in range(3):
            result[:, :, i] = (result[:, :, i] * vignette_mask).astype(np.uint8)
        return result
    
    @staticmethod
    def datamosh(frame, block_size=8, corruption=0.3):
        """Compression artifact effect"""
        original = frame.copy()
        result = frame.copy()
        h, w = result.shape[:2]
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if np.random.random() < corruption:
                    shift_y = np.random.randint(-block_size, block_size)
                    shift_x = np.random.randint(-block_size, block_size)
                    src_y = max(0, min(h - block_size, y + shift_y))
                    src_x = max(0, min(w - block_size, x + shift_x))
                    result[y:y+block_size, x:x+block_size] = \
                        original[src_y:src_y+block_size, src_x:src_x+block_size]
        return result
    
    @staticmethod
    def add_noise_uniform(frame, amount=50):
        """Add uniform random noise"""
        result = frame.copy()
        noise = np.random.randint(-amount, amount, result[:, :, :3].shape, dtype=np.int16)
        result[:, :, :3] = np.clip(result[:, :, :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return result
    
    @staticmethod
    def add_noise_gaussian(frame, mean=0, stddev=25):
        """Add Gaussian noise"""
        result = frame.copy()
        noise = np.zeros(result[:, :, :3].shape, dtype=np.float32)
        cv2.randn(noise, mean, stddev)
        result[:, :, :3] = cv2.add(result[:, :, :3], noise, dtype=cv2.CV_8U)
        return result
    
    @staticmethod
    def invert_channels(frame, b_invert=255, g_invert=255, r_invert=255):
        """Invert specific channels"""
        result = frame.copy()
        result[:, :, 0] = b_invert - result[:, :, 0]
        result[:, :, 1] = g_invert - result[:, :, 1]
        result[:, :, 2] = r_invert - result[:, :, 2]
        return result
    
    @staticmethod
    def swap_channels(frame, order='rbg'):
        """Swap color channels (e.g., 'rbg', 'grb', 'brg')"""
        result = frame.copy()
        b, g, r = result[:, :, 0].copy(), result[:, :, 1].copy(), result[:, :, 2].copy()
        channel_map = {'b': b, 'g': g, 'r': r}
        result[:, :, 0] = channel_map[order[0]]
        result[:, :, 1] = channel_map[order[1]]
        result[:, :, 2] = channel_map[order[2]]
        return result
    
    @staticmethod
    def edge_masked(frame, threshold1=100, threshold2=200):
        """Show frame only where edges detected"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        edges = cv2.Canny(bgr, threshold1, threshold2)
        result = np.zeros_like(frame)
        result[edges > 0] = frame[edges > 0]
        return result

