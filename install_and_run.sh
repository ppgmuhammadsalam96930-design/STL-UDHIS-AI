#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install websockets jinja2 requests markdown pdfkit python-docx
xdg-open client/Stl-udhis_psychology_duo_fixed.html
python3 core/Quantum_engine.py
