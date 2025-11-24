# STL UDHIS AI — Quantum Engine + RPP Engine

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/ppgmuhammadsalam96930-design/STL-UDHIS-AI)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![WebSocket](https://img.shields.io/badge/websocket-enabled-brightgreen.svg)]()
[![AI Engine](https://img.shields.io/badge/engine-quantum--ai-purple.svg)]()

> Quantum-powered platform to generate RPP, Modul, LKPD and interactive AI chat — ready for GitHub Pages & local Python backend.

![Preview](assets/preview.png)

## Quick Start

1. Extract bundle and open repository folder.
2. Create virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

(If you don't have `requirements.txt`, install the main deps:)
```bash
pip install websockets jinja2 requests markdown pdfkit python-docx
```

3. (Optional) Install `wkhtmltopdf` for PDF export:
```bash
# Debian/Ubuntu
sudo apt-get install -y wkhtmltopdf
```

4. Run the Quantum Engine:
```bash
python core/Quantum_engine.py
```

5. Open the UI:
- Locally: `client/Stl-udhis_psychology_duo_fixed.html`
- GitHub Pages: `https://<your-github-username>.github.io/<repo>/` (or your custom domain)

---

## What’s inside

- `core/` — Python engine, RPP engine, handlers
- `client/` — Frontend UI and websocket client
- `templates/` — All RPP templates (K13, Kurikulum Merdeka, PAUD, SMK, etc)
- `assets/preview.png` — Preview image for README

---

## WebSocket API (summary)
See `API_ROUTES.md` for detailed routes and examples.

---

## Contributing

Please fork the repository and submit pull requests. For issues, use GitHub Issues.


## Documentation

Detailed docs available in `/docs/`.

## API Routes

See `API_ROUTES.md` for WebSocket route reference.
