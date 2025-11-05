"""
Generate .pyi stub files automatically from Python modules
Extracts function signatures and creates type hints for IDE autocomplete

Usage:
    python generate_stubs.py src/graphics
"""

import ast
import argparse
from pathlib import Path
from typing import List, Dict, Any


class StubGenerator:
    """Generate .pyi stub files from Python source"""
    
    def __init__(self):
        self.imports = set()
        self.type_aliases = []
        self.classes = []
        self.functions = []
        
    def parse_file(self, filepath: Path) -> ast.Module:
        """Parse Python file into AST"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return ast.parse(f.read(), filename=str(filepath))
    
    def extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature with type hints"""
        args = []
        
        # Process arguments
        for i, arg in enumerate(node.args.args):
            arg_name = arg.arg
            
            # Skip 'self' and 'cls'
            if arg_name in ('self', 'cls'):
                continue
            
            # Get default value if exists
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                default = node.args.defaults[default_idx]
                default_str = self.ast_to_string(default)
                
                # Infer type from default value
                type_hint = self.infer_type(default)
                args.append(f"{arg_name}: {type_hint} = {default_str}")
            else:
                # No default - infer from name or use Frame
                type_hint = self.infer_type_from_name(arg_name)
                args.append(f"{arg_name}: {type_hint}")
        
        return f"({', '.join(args)})"
    
    def ast_to_string(self, node) -> str:
        """Convert AST node to string representation"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return str(node.value)
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Str):
            return f'"{node.s}"'
        elif isinstance(node, ast.Tuple):
            elements = [self.ast_to_string(e) for e in node.elts]
            return f"({', '.join(elements)})"
        elif isinstance(node, ast.List):
            elements = [self.ast_to_string(e) for e in node.elts]
            return f"[{', '.join(elements)}]"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"-{self.ast_to_string(node.operand)}"
        else:
            return "..."
    
    def infer_type(self, node) -> str:
        """Infer type from default value"""
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                return "bool"
            elif isinstance(val, int):
                return "int"
            elif isinstance(val, float):
                return "float"
            elif isinstance(val, str):
                return "str"
            elif val is None:
                return "Optional[Any]"
        elif isinstance(node, ast.Num):
            if isinstance(node.n, int):
                return "int"
            elif isinstance(node.n, float):
                return "float"
        elif isinstance(node, ast.Str):
            return "str"
        elif isinstance(node, ast.Tuple):
            # Check if it's a color tuple
            if len(node.elts) == 3:
                return "Color"
            return "Tuple"
        elif isinstance(node, ast.List):
            return "List"
        elif isinstance(node, ast.Name):
            if node.id == "None":
                return "Optional[Any]"
        
        return "Any"
    
    def infer_type_from_name(self, name: str) -> str:
        """Infer type from parameter name"""
        if name in ('frame', 'frame1', 'frame2', 'result'):
            return "Frame"
        elif name == 'history':
            return "List[Frame]"
        elif name.endswith('color') or name == 'color1' or name == 'color2':
            return "Color"
        elif name in ('width', 'height', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 
                      'cx', 'cy', 'radius', 'kernel_size', 'iterations',
                      'threshold', 'threshold1', 'threshold2', 'channel',
                      'levels', 'pixel_size', 'intensity', 'spacing',
                      'thickness', 'diameter', 'block_size', 'count',
                      'start_angle', 'end_angle', 'angle', 'shift', 'offset',
                      'stripe_height', 'stripe_width', 'square_size', 'size',
                      'line_spacing', 'value', 'amount', 'mean', 'stddev',
                      'b_invert', 'g_invert', 'r_invert', 'ksize', 'dx', 'dy'):
            return "int"
        elif name in ('alpha', 'ratio', 'sigma', 'sigma_s', 'sigma_r',
                      'sigma_color', 'sigma_space', 'contrast', 'brightness',
                      'shade_factor', 'decay', 'line_intensity', 'corruption',
                      'font_scale'):
            return "float"
        elif name in ('text', 'order'):
            return "str"
        elif name in ('filled', ):
            return "bool"
        elif name == 'points':
            return "Optional[List[List[int]]]"
        elif name == 'weights':
            return "Optional[List[float]]"
        elif name == 'indices':
            return "List[int]"
        elif name in ('bg_color', ):
            return "Optional[Color]"
        else:
            return "Any"
    
    def process_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Process a class definition"""
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node) or '',
            'methods': []
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Skip private methods
                if item.name.startswith('_'):
                    continue
                
                signature = self.extract_signature(item)
                docstring = ast.get_docstring(item) or ''
                
                # Determine return type
                if 'history' in item.name or node.name == 'blend':
                    return_type = "Optional[Frame]"
                else:
                    return_type = "Frame"
                
                class_info['methods'].append({
                    'name': item.name,
                    'signature': signature,
                    'docstring': docstring,
                    'return_type': return_type,
                    'is_static': any(isinstance(d, ast.Name) and d.id == 'staticmethod' 
                                    for d in item.decorator_list)
                })
        
        return class_info
    
    def generate_stub_content(self, module_path: Path) -> str:
        """Generate stub file content for a module"""
        tree = self.parse_file(module_path)
        
        # Extract classes
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self.process_class(node))
        
        # Build stub content
        lines = []
        lines.append('"""')
        lines.append(f'Type stubs for {module_path.stem}')
        lines.append('Auto-generated by generate_stubs.py')
        lines.append('"""')
        lines.append('')
        lines.append('from typing import Optional, List, Tuple, Any')
        lines.append('import numpy as np')
        lines.append('import numpy.typing as npt')
        lines.append('')
        lines.append('# Type aliases')
        lines.append('Frame = npt.NDArray[np.uint8]  # BGRA image array')
        lines.append('Color = Tuple[int, int, int]   # BGR color tuple')
        lines.append('')
        
        # Generate class stubs
        for cls in classes:
            lines.append(f"class {cls['name']}:")
            if cls['docstring']:
                lines.append(f'    """{cls["docstring"]}"""')
            lines.append('')
            
            if not cls['methods']:
                lines.append('    pass')
            else:
                for method in cls['methods']:
                    if method['is_static']:
                        lines.append('    @staticmethod')
                    lines.append(f"    def {method['name']}{method['signature']} -> {method['return_type']}: ...")
                    lines.append('')
        
        return '\n'.join(lines)
    
    def generate_init_stub(self, module_dir: Path) -> str:
        """Generate __init__.pyi with all classes and backwards compat"""
        lines = []
        lines.append('"""')
        lines.append('Type stubs for graphics module')
        lines.append('Auto-generated by generate_stubs.py')
        lines.append('"""')
        lines.append('')
        lines.append('from typing import Optional, List, Tuple, Any')
        lines.append('import numpy as np')
        lines.append('import numpy.typing as npt')
        lines.append('')
        lines.append('# Type aliases')
        lines.append('Frame = npt.NDArray[np.uint8]  # BGRA image array')
        lines.append('Color = Tuple[int, int, int]   # BGR color tuple')
        lines.append('')
        
        # Process each module file
        module_files = ['process.py', 'draw.py', 'generate.py', 'blend.py']
        all_classes = []
        all_functions = {}  # For backwards compatibility
        
        for module_file in module_files:
            module_path = module_dir / module_file
            if not module_path.exists():
                continue
            
            tree = self.parse_file(module_path)
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    cls_info = self.process_class(node)
                    all_classes.append(cls_info)
                    
                    # Store for backwards compat mappings
                    for method in cls_info['methods']:
                        func_name = method['name']
                        all_functions[func_name] = {
                            'class': cls_info['name'],
                            'signature': method['signature'],
                            'return_type': method['return_type']
                        }
        
        # Write class definitions
        for cls in all_classes:
            lines.append(f"class {cls['name']}:")
            if cls['docstring']:
                lines.append(f'    """{cls["docstring"]}"""')
            lines.append('')
            
            if not cls['methods']:
                lines.append('    pass')
            else:
                for method in cls['methods']:
                    if method['is_static']:
                        lines.append('    @staticmethod')
                    lines.append(f"    def {method['name']}{method['signature']} -> {method['return_type']}: ...")
                    lines.append('')
            
            lines.append('')
        
        # Write backwards compatibility functions
        lines.append('# Backwards compatibility function signatures')
        for func_name, func_info in sorted(all_functions.items()):
            lines.append(f"def {func_name}{func_info['signature']} -> {func_info['return_type']}: ...")
        
        return '\n'.join(lines)


def generate_stubs_for_module(module_path: str):
    """Generate stub files for a module directory"""
    module_dir = Path(module_path)
    
    if not module_dir.exists():
        print(f"Error: {module_dir} does not exist")
        return
    
    if not module_dir.is_dir():
        print(f"Error: {module_dir} is not a directory")
        return
    
    generator = StubGenerator()
    
    # Generate __init__.pyi
    print(f"Generating {module_dir}/__init__.pyi...")
    init_stub = generator.generate_init_stub(module_dir)
    
    init_stub_path = module_dir / "__init__.pyi"
    with open(init_stub_path, 'w', encoding='utf-8') as f:
        f.write(init_stub)
    
    print(f"✓ Created {init_stub_path}")
    print(f"  Lines: {len(init_stub.splitlines())}")
    
    # Also generate individual module stubs
    for py_file in module_dir.glob("*.py"):
        if py_file.name.startswith('_'):
            continue
        
        stub_path = py_file.with_suffix('.pyi')
        print(f"\nGenerating {stub_path.name}...")
        
        stub_content = generator.generate_stub_content(py_file)
        
        with open(stub_path, 'w', encoding='utf-8') as f:
            f.write(stub_content)
        
        print(f"✓ Created {stub_path}")
        print(f"  Lines: {len(stub_content.splitlines())}")
    
    print(f"\n{'='*50}")
    print("Stub generation complete!")
    print(f"Total stub files: {len(list(module_dir.glob('*.pyi')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .pyi stub files from Python modules")
    parser.add_argument('module_path', help='Path to module directory (e.g., src/graphics)')
    
    args = parser.parse_args()
    
    generate_stubs_for_module(args.module_path)