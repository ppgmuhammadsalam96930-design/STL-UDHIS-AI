@echo off
python -m venv venv
call venv\Scripts\activate
pip install websockets jinja2 requests markdown pdfkit python-docx
start "" "client\Stl-udhis_psychology_duo_fixed.html"
python core\Quantum_engine.py
