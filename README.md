# Jupyter Notebooks

## Getting started

1. Install tools:
   ```
   asdf install
   ```
1. Create virtual environment:
   ```
   python -m virtualenv .venv
   source .venv/bin/activate
   ```
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
1. Start Jupyter:
   ```
   jupyter notebook --config jupyter/config.py
   # OR
   make up
   ```

Jupyter will be available at http://localhost:8888/
