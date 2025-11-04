"""
Generate documentation with example images for all effects

Usage:
    python generate_docs.py [--input test.jpg] [--output docs/]
"""

import cv2
import numpy as np
import inspect
import argparse
from pathlib import Path
import effects


def generate_test_image(width=640, height=480):
    """Generate a test image with various patterns"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background
    for i in range(height):
        img[i, :] = [i * 255 // height, 128, 255 - (i * 255 // height)]
    
    # Geometric shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (450, 150), 80, (0, 255, 255), -1)
    cv2.line(img, (0, height//2), (width, height//2), (255, 0, 0), 3)
    
    # Text
    cv2.putText(img, "LIVE CODER", (width//2 - 100, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def apply_effect_safely(func, frame, func_name):
    """Apply effect and handle errors"""
    try:
        # Get function signature to determine required parameters
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Call with just frame if it's the only parameter
        if len(params) == 1:
            result = func(frame)
        else:
            # Use default values from signature
            result = func(frame)
        
        # Convert single-channel to BGR for display
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
    except Exception as e:
        print(f"Error in {func_name}: {e}")
        # Return error image
        error_img = np.zeros_like(frame)
        cv2.putText(error_img, f"ERROR: {str(e)[:30]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return error_img


def generate_documentation(input_image=None, output_dir="docs"):
    """Generate HTML documentation with example images"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    images_path = output_path / "images"
    images_path.mkdir(exist_ok=True)
    
    # Load or generate test image
    if input_image and Path(input_image).exists():
        frame = cv2.imread(input_image)
    else:
        frame = generate_test_image()
        cv2.imwrite(str(images_path / "original.png"), frame)
    
    # Save original
    cv2.imwrite(str(images_path / "original.png"), frame)
    
    # Generate HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Live Coder Effects Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .effect { 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .effect h2 { color: #0066cc; margin-top: 0; }
        .effect-info { color: #666; margin: 10px 0; }
        .params { 
            background: #f9f9f9; 
            padding: 10px; 
            border-left: 3px solid #0066cc;
            margin: 10px 0;
        }
        .images { display: flex; gap: 20px; flex-wrap: wrap; }
        .image-container { text-align: center; }
        .image-container img { 
            max-width: 400px; 
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-label { 
            font-weight: bold; 
            margin-top: 5px; 
            color: #333;
        }
        code { 
            background: #f4f4f4; 
            padding: 2px 6px; 
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <h1>Live Coder Effects Documentation</h1>
    <p>Generated documentation with example images for all effects.</p>
    
"""
    
    # Process each effect
    print("Generating documentation...")
    for func_name, func in effects.EFFECTS.items():
        print(f"Processing: {func_name}")
        
        # Apply effect
        result = apply_effect_safely(func, frame, func_name)
        
        # Save result image
        output_filename = f"{func_name}.png"
        cv2.imwrite(str(images_path / output_filename), result)
        
        # Get function documentation
        doc = inspect.getdoc(func) or "No documentation available"
        lines = doc.split('\n')
        description = lines[0] if lines else ""
        
        # Get function signature
        sig = inspect.signature(func)
        params_str = str(sig)
        
        # Add to HTML
        html_content += f"""
    <div class="effect">
        <h2>{func_name}</h2>
        <div class="effect-info">{description}</div>
        <div class="params">
            <strong>Signature:</strong> <code>{func_name}{params_str}</code>
        </div>
        <div class="images">
            <div class="image-container">
                <img src="images/original.png" alt="Original">
                <div class="image-label">Original</div>
            </div>
            <div class="image-container">
                <img src="images/{output_filename}" alt="{func_name}">
                <div class="image-label">Result</div>
            </div>
        </div>
        <details>
            <summary>Full Documentation</summary>
            <pre>{doc}</pre>
        </details>
    </div>
"""
    
    # Close HTML
    html_content += """
</body>
</html>
"""
    
    # Write HTML file
    html_path = output_path / "index.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nDocumentation generated!")
    print(f"Open: {html_path.absolute()}")
    print(f"Total effects: {len(effects.EFFECTS)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate effects documentation")
    parser.add_argument('--input', help='Input test image (optional)', default=None)
    parser.add_argument('--output', help='Output directory', default='../docs')
    
    args = parser.parse_args()
    
    generate_documentation(args.input, args.output)