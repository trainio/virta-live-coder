import numpy as np
import cv2

class analyze:
    """Analyze frame"""
    @staticmethod
    def diagram_visualize(frame):
        """
        Performs a digram visualization on the frame's byte stream.
        This replaces the frame with a 2D plot where (x, y) coordinates
        correspond to the frequency of 2-byte sequences <x, y> in the
        frame's raw data.
        """
        try:
            # 1. Flatten the (H, W, C) frame into a 1D byte array
            # e.g., [R, G, B, A, R, G, B, A, ...]
            flat_bytes = frame.ravel()
            
            if len(flat_bytes) < 2:
                return frame # Not enough data to process

            # 2. Create a 256x256 canvas to plot the digrams.
            # We use float32 to avoid overflow when adding frequency
            digram_canvas = np.zeros((256, 256), dtype=np.float32)

            # 3. Iterate through the byte stream and plot frequencies
            # We use a fast, vectorized numpy approach instead of a slow loop
            
            # Get all "first bytes" (x-coordinates)
            x_coords = flat_bytes[:-1]
            
            # Get all "second bytes" (y-coordinates)
            y_coords = flat_bytes[1:]
            
            # Add 1 to the (y, x) location for each pair.
            # np.add.at is a high-speed way to do this.
            np.add.at(digram_canvas, (y_coords, x_coords), 1)

            # 4. Normalize the canvas to be visible (0-255)
            # We use a logarithmic scale to make less frequent pairs visible
            # Add 1 to avoid log(0)
            digram_canvas = np.log1p(digram_canvas)
            
            # Now normalize the log-scaled data to 0-255
            cv2.normalize(digram_canvas, digram_canvas, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to a displayable 8-bit image
            digram_vis = digram_canvas.astype(np.uint8)

            # 5. Resize to fit the original frame and return
            h, w = frame.shape[:2]
            
            # Convert grayscale digram to BGRA to match the pipeline format
            output_frame = cv2.cvtColor(digram_vis, cv2.COLOR_GRAY2BGRA)
            
            # Resize and return
            return cv2.resize(output_frame, (w, h), interpolation=cv2.INTER_NEAREST)

        except Exception as e:
            print(f"Error in digram_visualize: {e}")
            return frame