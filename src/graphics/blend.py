
import cv2
import numpy as np


# ============================================================================
# BLEND - Compositing operations
# ============================================================================

class blend:
    """Blend and composite multiple frames"""
    
    @staticmethod
    def alpha(frame1, frame2, alpha=0.5):
        """Alpha blend two frames (alpha=0.5 is 50/50)"""
        return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)
    
    @staticmethod
    def add(frame1, frame2):
        """Add two frames"""
        return cv2.add(frame1, frame2)
    
    @staticmethod
    def subtract(frame1, frame2):
        """Subtract frame2 from frame1"""
        return cv2.subtract(frame1, frame2)
    
    @staticmethod
    def difference(frame1, frame2):
        """Absolute difference between frames"""
        return cv2.absdiff(frame1, frame2)
    
    @staticmethod
    def multiply(frame1, frame2):
        """Multiply frames (darkens)"""
        result = frame1.astype(np.float32) * frame2.astype(np.float32) / 255.0
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def screen(frame1, frame2):
        """Screen blend (lightens)"""
        result = 255 - ((255 - frame1.astype(np.float32)) * (255 - frame2.astype(np.float32)) / 255.0)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def overlay(frame1, frame2):
        """Overlay blend (combines multiply and screen)"""
        result = np.zeros_like(frame1, dtype=np.float32)
        mask = frame1 < 128
        result[mask] = 2 * frame1[mask].astype(np.float32) * frame2[mask].astype(np.float32) / 255.0
        result[~mask] = 255 - 2 * (255 - frame1[~mask].astype(np.float32)) * (255 - frame2[~mask].astype(np.float32)) / 255.0
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def lighten(frame1, frame2):
        """Lighten - keeps lighter pixels"""
        return np.maximum(frame1, frame2)
    
    @staticmethod
    def darken(frame1, frame2):
        """Darken - keeps darker pixels"""
        return np.minimum(frame1, frame2)
    
    @staticmethod
    def average(frame1, frame2):
        """Average of two frames"""
        return ((frame1.astype(np.float32) + frame2.astype(np.float32)) / 2).astype(np.uint8)
    
    @staticmethod
    def max(frame1, frame2):
        """Maximum value per pixel"""
        return np.maximum(frame1, frame2)
    
    @staticmethod
    def min(frame1, frame2):
        """Minimum value per pixel"""
        return np.minimum(frame1, frame2)
    
    # === HISTORY BLENDING ===
    
    @staticmethod
    def history_average(history):
        """
        Average all frames in history
        
        Args:
            history: List/array of frames
        
        Returns:
            Averaged frame
        """
        if not history or len(history) == 0:
            return None
        
        result = np.zeros_like(history[0], dtype=np.float32)
        for frame in history:
            result += frame.astype(np.float32)
        result /= len(history)
        return result.astype(np.uint8)
    
    @staticmethod
    def history_weighted(history, weights=None):
        """
        Weighted average of history (more recent = higher weight by default)
        
        Args:
            history: List/array of frames
            weights: List of weights (None = linear decay, most recent = 1.0)
        
        Returns:
            Weighted average frame
        """
        if not history or len(history) == 0:
            return None
        
        n = len(history)
        if weights is None:
            # Linear decay: most recent=1.0, oldest=1/n
            weights = np.linspace(1/n, 1.0, n)
        
        result = np.zeros_like(history[0], dtype=np.float32)
        total_weight = 0
        
        for frame, weight in zip(history, weights):
            result += frame.astype(np.float32) * weight
            total_weight += weight
        
        result /= total_weight
        return result.astype(np.uint8)
    
    @staticmethod
    def history_max(history):
        """
        Maximum value across all history frames
        
        Args:
            history: List/array of frames
        
        Returns:
            Frame with max values
        """
        if not history or len(history) == 0:
            return None
        
        result = history[0].copy()
        for frame in history[1:]:
            result = np.maximum(result, frame)
        return result
    
    @staticmethod
    def history_min(history):
        """
        Minimum value across all history frames
        
        Args:
            history: List/array of frames
        
        Returns:
            Frame with min values
        """
        if not history or len(history) == 0:
            return None
        
        result = history[0].copy()
        for frame in history[1:]:
            result = np.minimum(result, frame)
        return result
    
    @staticmethod
    def history_median(history):
        """
        Median value across all history frames
        
        Args:
            history: List/array of frames
        
        Returns:
            Frame with median values
        """
        if not history or len(history) == 0:
            return None
        
        # Stack all frames and compute median
        stack = np.stack([f.astype(np.float32) for f in history], axis=0)
        result = np.median(stack, axis=0)
        return result.astype(np.uint8)
    
    @staticmethod
    def history_diff_sum(history):
        """
        Sum of absolute differences between consecutive frames (motion accumulation)
        
        Args:
            history: List/array of frames
        
        Returns:
            Accumulated motion frame
        """
        if not history or len(history) < 2:
            return history[0].copy() if history else None
        
        result = np.zeros_like(history[0], dtype=np.float32)
        
        for i in range(len(history) - 1):
            diff = cv2.absdiff(history[i], history[i + 1])
            result += diff.astype(np.float32)
        
        # Normalize to 0-255 range
        if result.max() > 0:
            result = (result / result.max() * 255)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def history_trail(history, decay=0.9):
        """
        Create motion trail effect by blending history with exponential decay
        
        Args:
            history: List/array of frames (oldest to newest)
            decay: Decay factor for older frames (0.0-1.0)
        
        Returns:
            Frame with motion trails
        """
        if not history or len(history) == 0:
            return None
        
        result = np.zeros_like(history[0], dtype=np.float32)
        
        # Blend from oldest to newest with exponential decay
        for i, frame in enumerate(history):
            weight = decay ** (len(history) - 1 - i)
            result += frame.astype(np.float32) * weight
        
        # Normalize
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    @staticmethod
    def history_onion_skin(history, count=5, alpha=0.3):
        """
        Onion skin effect - blend last N frames with decreasing opacity
        
        Args:
            history: List/array of frames
            count: Number of frames to blend
            alpha: Base alpha for blending
        
        Returns:
            Onion skin blended frame
        """
        if not history or len(history) == 0:
            return None
        
        # Take last 'count' frames
        frames = history[-count:] if len(history) >= count else history
        
        if len(frames) == 1:
            return frames[0].copy()
        
        # Start with the oldest frame in the selection
        result = frames[0].astype(np.float32) * alpha
        
        # Blend each subsequent frame with increasing weight
        for i, frame in enumerate(frames[1:], 1):
            weight = alpha + (1 - alpha) * (i / (len(frames) - 1))
            result = result * (1 - weight) + frame.astype(np.float32) * weight
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def history_echo(history, indices=[0, 5, 10], weights=None):
        """
        Echo effect - blend specific frames from history
        
        Args:
            history: List/array of frames
            indices: List of frame indices to blend (0=most recent)
            weights: Weights for each index (None = equal weights)
        
        Returns:
            Echo blended frame
        """
        if not history or len(history) == 0:
            return None
        
        if weights is None:
            weights = [1.0 / len(indices)] * len(indices)
        
        result = np.zeros_like(history[0], dtype=np.float32)
        
        for idx, weight in zip(indices, weights):
            if idx < len(history):
                # Negative indices work from end (most recent)
                frame = history[-(idx + 1)]
                result += frame.astype(np.float32) * weight
        
        return np.clip(result, 0, 255).astype(np.uint8)


