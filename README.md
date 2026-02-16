
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
   
   python run.py (live coding with editor)
   ```


# Remove virtual environment
deactivate
rm -rf virta_venv




######################################################################
OTHER STUFF

# Open and edit these files
live.py
ui.py

# Batch process
python batch_process.py

# App key bindings
q = quit
r = save frame and script

######################################################################
DOCS

# Generate documentation
python src/generate_docs.py

# Read docs
open docs/index.html in browser


# Generate stubs (for autocomplete)
python src/generate_stubs.py