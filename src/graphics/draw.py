
import cv2
import numpy as np


# ============================================================================
# DRAW - Graphics and overlays
# ============================================================================

class draw:
    """Drawing graphics on frames"""
    
    @staticmethod
    def rectangle(frame, x1=50, y1=50, x2=200, y2=150, color=(0, 255, 0), thickness=2):
        """Draw rectangle"""
        result = frame.copy()
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        return result
    
    @staticmethod
    def circle(frame, cx=320, cy=240, radius=50, color=(0, 0, 255), thickness=-1):
        """Draw circle (thickness=-1 for filled)"""
        result = frame.copy()
        cv2.circle(result, (cx, cy), radius, color, thickness)
        return result
    
    @staticmethod
    def line(frame, x1=0, y1=0, x2=640, y2=480, color=(255, 255, 255), thickness=2):
        """Draw line"""
        result = frame.copy()
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        return result
    
    @staticmethod
    def crosshair(frame, cx=None, cy=None, size=20, color=(0, 255, 0), thickness=1):
        """Draw crosshair (None=center)"""
        result = frame.copy()
        h, w = frame.shape[:2]
        if cx is None:
            cx = w // 2
        if cy is None:
            cy = h // 2
        cv2.line(result, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(result, (cx, cy - size), (cx, cy + size), color, thickness)
        return result
    
    @staticmethod
    def polygon(frame, points=None, color=(255, 0, 255), thickness=2, filled=False):
        """Draw polygon (points=list of [x,y])"""
        result = frame.copy()
        if points is None:
            points = [[320, 100], [250, 200], [390, 200]]
        pts = np.array(points, np.int32)
        if filled:
            cv2.fillPoly(result, [pts], color)
        else:
            cv2.polylines(result, [pts], True, color, thickness)
        return result
    
    @staticmethod
    def ellipse(frame, cx=320, cy=240, width=100, height=50, angle=0, 
                start_angle=0, end_angle=360, color=(255, 128, 0), thickness=2):
        """Draw ellipse"""
        result = frame.copy()
        cv2.ellipse(result, (cx, cy), (width, height), angle, 
                    start_angle, end_angle, color, thickness)
        return result
    
    @staticmethod
    def text(frame, text="Hello World", x=100, y=100, font_scale=1.0, 
             color=(255, 255, 255), thickness=2, bg_color=None):
        """Draw text (bg_color=None for no background)"""
        result = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        if bg_color is not None:
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(result, (x - 5, y - text_height - 5), 
                         (x + text_width + 5, y + baseline + 5), bg_color, -1)
        cv2.putText(result, text, (x, y), font, font_scale, color, thickness)
        return result
    
    @staticmethod
    def grid(frame, spacing=50, color=(100, 100, 100), thickness=1):
        """Draw grid overlay"""
        result = frame.copy()
        h, w = frame.shape[:2]
        for x in range(0, w, spacing):
            cv2.line(result, (x, 0), (x, h), color, thickness)
        for y in range(0, h, spacing):
            cv2.line(result, (0, y), (w, y), color, thickness)
        return result
    
    @staticmethod
    def colorize_edges(frame, threshold1=50, threshold2=150, color=(0, 255, 0)):
        """Draw colored edges overlay"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        result = frame.copy()
        result[edges > 0, :3] = color
        return result

