

import cv2
import numpy as np



# ============================================================================
# GENERATE - Create new frames
# ============================================================================

class generate:
    """Generate new frames from scratch"""
    
    @staticmethod
    def solid(width=640, height=480, color=(0, 0, 0), frame=None):
        """Create solid color frame"""
        if frame is not None:
            height, width = frame.shape[:2]
        frame = np.zeros((height, width, 4), dtype=np.uint8)
        frame[:, :, :3] = color
        frame[:, :, 3] = 255  # Full alpha
        return frame
    
    @staticmethod
    def noise_uniform(width=640, height=480, frame=None):
        """Create uniform random noise"""
        if frame is not None:
            height, width = frame.shape[:2]
        frame = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        frame[:, :, 3] = 255  # Full alpha
        return frame
    
    @staticmethod
    def noise_gaussian(width=640, height=480, mean=128, stddev=50, frame=None):
        """Create Gaussian noise"""
        if frame is not None:
            height, width = frame.shape[:2]
        result = np.random.normal(mean, stddev, (height, width, 4))
        result = np.clip(result, 0, 255).astype(np.uint8)
        result[:, :, 3] = 255  # Full alpha
        return result
    
    @staticmethod
    def gradient_horizontal(width=640, height=480, color1=(0, 0, 0), color2=(255, 255, 255), frame=None):
        """Create horizontal gradient"""
        if frame is not None:
            height, width = frame.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(width):
            ratio = i / width
            result[:, i, 0] = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            result[:, i, 1] = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            result[:, i, 2] = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        result[:, :, 3] = 255
        return result
    
    @staticmethod
    def gradient_vertical(width=640, height=480, color1=(0, 0, 0), color2=(255, 255, 255), frame=None):
        """Create vertical gradient"""
        if frame is not None:
            height, width = frame.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.uint8)
        for i in range(height):
            ratio = i / height
            result[i, :, 0] = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            result[i, :, 1] = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            result[i, :, 2] = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        result[:, :, 3] = 255
        return result
    
    @staticmethod
    def gradient_radial(width=640, height=480, color1=(0, 0, 0), color2=(255, 255, 255), frame=None):
        """Create radial gradient from center"""
        if frame is not None:
            height, width = frame.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        max_dist = np.sqrt(cx**2 + cy**2)
        
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                ratio = min(dist / max_dist, 1.0)
                result[y, x, 0] = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                result[y, x, 1] = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                result[y, x, 2] = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        
        result[:, :, 3] = 255
        return result
    
    @staticmethod
    def checkerboard(width=640, height=480, square_size=40, color1=(0, 0, 0), color2=(255, 255, 255), frame=None):
        """Create checkerboard pattern"""
        if frame is not None:
            height, width = frame.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    result[y, x, :3] = color1
                else:
                    result[y, x, :3] = color2
        result[:, :, 3] = 255
        return result
    
    @staticmethod
    def stripes_horizontal(width=640, height=480, stripe_height=40, color1=(0, 0, 0), color2=(255, 255, 255), frame=None):
        """Create horizontal stripes"""
        if frame is not None:
            height, width = frame.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            if (y // stripe_height) % 2 == 0:
                result[y, :, :3] = color1
            else:
                result[y, :, :3] = color2
        result[:, :, 3] = 255
        return result
    
    @staticmethod
    def stripes_vertical(width=640, height=480, stripe_width=40, color1=(0, 0, 0), color2=(255, 255, 255), frame=None):
        """Create vertical stripes"""
        if frame is not None:
            height, width = frame.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.uint8)
        for x in range(width):
            if (x // stripe_width) % 2 == 0:
                result[:, x, :3] = color1
            else:
                result[:, x, :3] = color2
        result[:, :, 3] = 255
        return result
