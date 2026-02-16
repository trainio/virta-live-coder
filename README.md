
# Live Coder



## Installation

### Pre-Requirements

- Python 3.8 or higher — Install from [python.org](https://www.python.org/downloads/)
- **Mac/Linux only:** tkinter (must be installed **before** creating a virtual environment)

  Mac:
  ```bash
  brew install python-tk@3.XX
  ```
  *(replace 3.XX with your Python version, e.g. 3.12, 3.13)*

  Linux:
  ```bash
  sudo apt-get install python3-tk
  ```

### Setup

1. Create and activate a virtual environment (optional, but recommended):

   Linux/Mac:
   ```bash
   python3 -m venv virta_venv
   source virta_venv/bin/activate
   ```

   Windows:
   ```bash
   python -m venv virta_venv
   virta_venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   python src/install.py
   ```

3. Run:
   ```bash
   python live_with_gui.py 
   python run.py
   ```


# Remove virtual environment
deactivate
rm -rf virta_venv




######################################################################
OTHER STUFF

# Open and edit these files when running without gui (python run.py)
live.py
ui.py

# Batch process files without live view
python batch_process.py

# App key bindings
q = quit
r = save frame and script

######################################################################
DOCS

# Generate documentation (html)
python src/generate_docs.py

# Read docs (html)
open docs/index.html in browser

# Generate stubs (for autocomplete)
python src/generate_stubs.py

######################################################################
ADD NEW FUNCTIONS

1. Choose the module by purpose from scr/graphics/

2. Add your function as a static method on the module's class.
Accept frame as the first parameter (a BGRA numpy array, dtype=uint8)
Return a BGRA numpy array of the same dtype

3. Register the backwards-compatible alias in __init__.py
In src/graphics/__init__.py, add two things:
a) The alias (under the appropriate comment section):
b) The export (in the __all__ list):

4. Regenerate the type stubs
The .pyi files are auto-generated. Run:
python src/generate_stubs.py src/graphics
