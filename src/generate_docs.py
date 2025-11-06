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
import sys

# Add parent directory to path to import graphics module
sys.path.insert(0, str(Path(__file__).parent))

from graphics import process, draw, generate, blend, analyze


def generate_test_image(width=640, height=480):
    """Generate a test image with various patterns"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Gradient background
    for i in range(height):
        img[i, :, :3] = [i * 255 // height, 128, 255 - (i * 255 // height)]
    img[:, :, 3] = 255  # Full alpha
    
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
        
        return result
    except Exception as e:
        print(f"Error in {func_name}: {e}")
        # Return error image
        error_img = np.zeros_like(frame)
        cv2.putText(error_img, f"ERROR: {str(e)[:30]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return error_img


def get_all_effects():
    """Collect all effects from all modules"""
    effects = {}
    
    # Get all static methods from each class
    for category_name, category_class in [
        ('process', process),
        ('draw', draw),
        ('generate', generate),
        ('blend', blend),
        ('analyze', analyze)
    ]:
        for name in dir(category_class):
            # Skip private methods and special methods
            if name.startswith('_'):
                continue
            
            attr = getattr(category_class, name)
            
            # Check if it's a callable method
            if callable(attr):
                # Store with category prefix
                full_name = f"{category_name}.{name}"
                effects[full_name] = attr
    
    return effects


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
        # Convert to BGRA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    else:
        frame = generate_test_image()
    
    # Save original
    original_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(str(images_path / "original.png"), original_bgr)
    
    # Generate HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Live Coder Effects Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .category { 
            background: white; 
            padding: 20px; 
            margin: 30px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .category h2 {
            color: #0066cc;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
        }
        .effect { 
            background: #fafafa; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 6px;
            border-left: 4px solid #0066cc;
        }
        .effect h3 { color: #333; margin-top: 0; }
        .effect-info { color: #666; margin: 10px 0; }
        .params { 
            background: #f0f0f0; 
            padding: 10px; 
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 14px;
        }
        .images { display: flex; gap: 20px; flex-wrap: wrap; margin: 15px 0; }
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
        details { margin-top: 10px; }
        summary { 
            cursor: pointer; 
            color: #0066cc;
            font-weight: bold;
        }
        pre { 
            background: #f9f9f9; 
            padding: 10px; 
            border-radius: 4px;
            overflow-x: auto;
        }
        .toc {
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toc h2 { margin-top: 0; }
        .toc ul { list-style: none; padding-left: 0; }
        .toc li { margin: 5px 0; }
        .toc a { color: #0066cc; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Live Coder Effects Documentation</h1>
    <p>Generated documentation with example images for all effects.</p>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#process">Process - Image transformations</a></li>
            <li><a href="#draw">Draw - Graphics and overlays</a></li>
            <li><a href="#generate">Generate - Create new frames</a></li>
            <li><a href="#blend">Blend - Compositing operations</a></li>
            <li><a href="#analyze">Analyze - Image analysis</a></li>
        </ul>
    </div>
    
"""
    
    # Get all effects organized by category
    all_effects = get_all_effects()
    
    # Group by category
    categories = {}
    for full_name, func in all_effects.items():
        category, name = full_name.split('.')
        if category not in categories:
            categories[category] = {}
        categories[category][name] = func
    
    # Category descriptions
    category_descriptions = {
        'process': 'Image processing and transformation effects',
        'draw': 'Drawing graphics and overlays on frames',
        'generate': 'Generate new frames from scratch',
        'blend': 'Blend and composite multiple frames',
        'analyze': 'Analyse frames'
    }
    
    # Process each category
    for category in ['process', 'draw', 'generate', 'blend', 'analyze']:
        if category not in categories:
            continue
        
        html_content += f"""
    <div class="category" id="{category}">
        <h2>{category.capitalize()}</h2>
        <p>{category_descriptions.get(category, '')}</p>
"""
        
        print(f"\n=== {category.upper()} ===")
        
        # Process each effect in category
        for func_name, func in sorted(categories[category].items()):
            print(f"Processing: {category}.{func_name}")
            
            # Special handling for generate functions (no frame input)
            if category == 'generate':
                try:
                    result = func(640, 480)
                except Exception as e:
                    print(f"Error in {func_name}: {e}")
                    result = frame.copy()
            # Special handling for blend functions (need two frames)
            elif category == 'blend' and func_name.startswith('history_'):
                # Create sample history
                history = [frame.copy() for _ in range(5)]
                try:
                    result = func(history)
                    if result is None:
                        result = frame.copy()
                except Exception as e:
                    print(f"Error in {func_name}: {e}")
                    result = frame.copy()
            elif category == 'blend' and not func_name.startswith('history_'):
                # Two-frame blend - use input image + generated graphics
                h, w = frame.shape[:2]
                # Create interesting frame2 with shapes and gradients
                frame2 = generate.gradient_radial(w, h, (0, 50, 100), (255, 200, 50))
                frame2 = draw.circle(frame2, w//2, h//2, min(w, h)//4, (255, 0, 255), -1)
                frame2 = draw.rectangle(frame2, w//4, h//4, 3*w//4, 3*h//4, (0, 255, 255), 3)
                
                # Save frame2 for visualization
                frame2_filename = f"blend_frame2.png"
                frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(str(images_path / frame2_filename), frame2_bgr)
                
                try:
                    result = func(frame, frame2)
                except Exception as e:
                    print(f"Error in {func_name}: {e}")
                    result = frame.copy()
                
                # Save with special handling for blend visualization
                output_filename = f"{category}_{func_name}.png"
                result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(str(images_path / output_filename), result_bgr)
                
                # Get function documentation
                doc = inspect.getdoc(func) or "No documentation available"
                lines = doc.split('\n')
                description = lines[0] if lines else ""
                
                # Get function signature
                sig = inspect.signature(func)
                params_str = str(sig)
                
                # Add to HTML with three images (frame1 + frame2 = result)
                html_content += f"""
        <div class="effect">
            <h3>{func_name}</h3>
            <div class="effect-info">{description}</div>
            <div class="params">
                <strong>Usage:</strong><br>
                <code>result = {category}.{func_name}(frame1, frame2{', ...' if len(sig.parameters) > 2 else ''})</code>
            </div>
            <div class="images">
                <div class="image-container">
                    <img src="images/original.png" alt="Frame 1">
                    <div class="image-label">Frame 1 (Input)</div>
                </div>
                <div class="image-container">
                    <img src="images/{frame2_filename}" alt="Frame 2">
                    <div class="image-label">Frame 2 (Generated)</div>
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
                continue  # Skip normal processing below
            else:
                # Normal processing
                result = apply_effect_safely(func, frame, func_name)
            
            # Save result image
            output_filename = f"{category}_{func_name}.png"
            result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(str(images_path / output_filename), result_bgr)
            
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
            <h3>{func_name}</h3>
            <div class="effect-info">{description}</div>
            <div class="params">
                <strong>Usage:</strong><br>
                <code>frame = {category}.{func_name}{params_str}</code>
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
        
        html_content += """
    </div>
"""
    
    # Close HTML
    html_content += """
    <footer style="text-align: center; margin-top: 50px; color: #666;">
        <p>Generated by Live Coder Documentation Generator (./src/generate_docs.py)</p>
    </footer>
</body>
</html>
"""
    
    # Write HTML file
    html_path = output_path / "index.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n{'='*50}")
    print(f"Documentation generated!")
    print(f"Open: {html_path.absolute()}")
    print(f"Total effects: {len(all_effects)}")
    print(f"Categories: {', '.join(categories.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate effects documentation")
    parser.add_argument('--input', help='Input test image (optional)', default=None)
    parser.add_argument('--output', help='Output directory', default='./docs')
    
    args = parser.parse_args()
    
    generate_documentation(args.input, args.output)