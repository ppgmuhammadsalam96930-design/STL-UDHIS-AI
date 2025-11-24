#!/usr/bin/env bash
set -euo pipefail
echo "Installing system dependencies (wkhtmltopdf, fonts, cairo for weasyprint if possible)"
if command -v apt >/dev/null 2>&1; then
  sudo apt update
  sudo apt install -y wkhtmltopdf libxml2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libssl-dev
  sudo apt install -y libcairo2 libcairo2-dev libpango1.0-0 libgdk-pixbuf2.0-0
else
  echo "APT not available. Please install wkhtmltopdf and WeasyPrint dependencies manually."
fi
echo "Setting up Python venv and installing Python packages..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install openai websockets aiohttp python-docx qrcode[pil] html2image pillow pdfkit weasyprint pymongo psycopg2-binary
echo "Installer finished. To run the handler:"
echo "  export OPENAI_API_KEY=your_key_here"
echo "  export QUANTUM_OUTPUT_DIR=$(pwd)/rpp_full_features_outputs"
echo "  source venv/bin/activate && python3 quantum_rpp_handler_features.py"
