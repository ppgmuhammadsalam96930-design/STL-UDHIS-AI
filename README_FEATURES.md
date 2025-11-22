Quantum RPP Full Features Bundle
=================================
Includes:
- quantum_rpp_handler_features.py  (handler with DB, progress, watermark, signature, templates, zip export)
- templates/                       (example RPP templates per class)
- Stl-udhis_psychology_duo_fixed.html (copied + injected UI helper)
- installer.sh                     (installs wkhtmltopdf, weasyprint deps, python packages)
- Notes: Set OPENAI_API_KEY to enable real AI. Set MONGO_URI or POSTGRES_URL to enable DB saving.

Progress updates:
- The handler attempts to call engine.broadcast_progress(payload). If your QuantumEngine supports a broadcast_progress method, it will be used to push real-time progress via websockets to clients.
- Fallback: progress JSON file is written to outputs (progress_<base>.json) — you can serve this file from your webserver and the injected UI builder will poll it.

Digital signature:
- A SHA256 hash of generated PDF is created and saved as <pdf>.sig.txt. This is a simple integrity signature; for cryptographic signatures with certificates, integrate an HSM or OpenSSL signing step.

Template usage:
- Buttons can include data-template, data-kelas or data-jenjang attributes to pick templates (e.g. data-template=sd_kelas_2)

Example run:
sudo ./installer.sh
export OPENAI_API_KEY=sk-...
export MONGO_URI=mongodb://user:pass@host:27017/dbname  # optional
source venv/bin/activate
python3 quantum_rpp_handler_features.py
