
# VIRTA Live Coder

---

## Installation
1. Create virtual environment and activate it (optional)
2. Install dependencies
3. Run

---

# CHECK BEFORE INSTALLATION

## Pre-Requirements
Python 3.8 or higher

## Install pre-requirements
Install Python from [python.org](https://www.python.org/downloads/)

## Linux users also need to install
Install tkinter: `sudo apt-get install python3-tk`

---

SETTING UP VIRTUAL ENVIRONMENT (OPTIONAL)

## Recommended: use virtual environment Mac/Linux
python -m venv live_venv
source live_venv/bin/activate  # Linux/Mac
## or Windows:
python -m venv live_venv
live_venv\Scripts\activate

---

# INSTALLATION

## Install
python src/install.py

---

# RUNNING

## Run
python run.py

---

# OTHER STUFF

## Open and edit these files
live.py
ui.py

## Batch process
Process all frames in a folder with live.py script. This can be helpful when processing video frames or large image datasets.
python batch_process.py

## App key bindings
q = quit
r = save frame and script to 'output' folder
c = change camera
m = manual/automatic history
h = add frames to history (in manual mode only)

---

# DOCS

## Generate documentation
python src/generate_docs.py

## Read docs
open docs/index.html in browser
