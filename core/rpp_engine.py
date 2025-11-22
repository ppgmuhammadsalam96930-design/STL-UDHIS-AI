# rpp_engine.py
import os
import io
import json
import requests
import zipfile
from typing import Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, Template
import markdown as md
import logging
import shutil
import tempfile
import pathlib

# Optional: pdf export via pdfkit (requires wkhtmltopdf)
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except Exception:
    PDFKIT_AVAILABLE = False

# Fallback PDF generation using ReportLab (pure-Python)
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger("RPPEngine")
logger.setLevel(logging.INFO)

class RPPEngine:
    """
    Hybrid RPPEngine:
    - Prefer local templates from LOCAL_TEMPLATE_PATH
    - Fallback to GitHub raw files in GITHUB_REPO (user/repo path)
    - Render templates via Jinja2
    - Export to markdown or PDF (if pdfkit+wkhtmltopdf present)
    """

    def __init__(self, local_template_path: str = "template", github_repo_url: str = None, github_branch: str = "main"):
        self.LOCAL_TEMPLATE_PATH = local_template_path.rstrip("/")
        self.github_repo_url = github_repo_url.rstrip("/") if github_repo_url else None
        self.github_branch = github_branch
        # prepare jinja2 env for local templates (if folder exists)
        if os.path.isdir(self.LOCAL_TEMPLATE_PATH):
            self.jinja_env = Environment(loader=FileSystemLoader(self.LOCAL_TEMPLATE_PATH))
        else:
            # fallback empty env; will fetch templates from GitHub
            self.jinja_env = Environment()
        logger.info("RPPEngine initialized. local=%s github=%s", self.LOCAL_TEMPLATE_PATH, self.github_repo_url)

    # -------------------------
    # Template list / discovery
    # -------------------------
    def list_local_templates(self) -> List[str]:
        if not os.path.isdir(self.LOCAL_TEMPLATE_PATH):
            return []
        exts = (".md", ".html", ".jinja", ".j2", ".txt")
        files = []
        for root, _, filenames in os.walk(self.LOCAL_TEMPLATE_PATH):
            for f in filenames:
                if f.lower().endswith(exts):
                    rel = os.path.relpath(os.path.join(root, f), self.LOCAL_TEMPLATE_PATH)
                    files.append(rel.replace("\\\\","/"))
        return sorted(files)

    def _github_raw_base(self) -> Optional[str]:
        # convert repo url to raw base url: https://raw.githubusercontent.com/<owner>/<repo>/<branch>/
        if not self.github_repo_url:
            return None
        parts = self.github_repo_url.rstrip("/").split("/")
        try:
            owner = parts[-2]
            repo = parts[-1]
            base = f"https://raw.githubusercontent.com/{owner}/{repo}/{self.github_branch}/"
            return base
        except Exception:
            return None

    def list_github_templates(self, templates_path: str = "templates") -> List[str]:
        base = self._github_raw_base()
        if not base:
            return []
        index_url = base + templates_path.strip("/") + "/index.json"
        try:
            r = requests.get(index_url, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        candidates = ["rpp_template.md", "rpp_template_k13.md", "rpp_template_merdeka.md", "modul_template.md"]
        found = []
        for c in candidates:
            url = base + templates_path.strip("/") + "/" + c
            try:
                rr = requests.head(url, timeout=6)
                if rr.status_code == 200:
                    found.append(f"{templates_path}/{c}")
            except Exception:
                continue
        return found

    # -------------------------
    # Fetch template content
    # -------------------------
    def load_local_template(self, template_relpath: str) -> Optional[str]:
        path = os.path.join(self.LOCAL_TEMPLATE_PATH, template_relpath)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        alt = os.path.join(self.LOCAL_TEMPLATE_PATH, os.path.basename(template_relpath))
        if os.path.isfile(alt):
            with open(alt, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def fetch_github_template(self, template_relpath: str, templates_path: str = "templates") -> Optional[str]:
        base = self._github_raw_base()
        if not base:
            return None
        if template_relpath.startswith("http"):
            url = template_relpath
        else:
            if template_relpath.startswith(templates_path):
                url = base + template_relpath
            else:
                url = base + templates_path.strip("/") + "/" + template_relpath
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            logger.warning("github fetch failed: %s -> %s", url, e)
        return None

    # -------------------------
    # Render and export
    # -------------------------
    def render_template(self, template_name: str, context: Dict = None, prefer_local: bool = True, templates_path: str = "templates") -> Dict:
        ctx = context or {}
        if prefer_local:
            local = self.load_local_template(template_name)
            if local is not None:
                try:
                    tpl = self.jinja_env.from_string(local)
                    rendered = tpl.render(**ctx)
                    html = md.markdown(rendered)
                    return {"ok":True,"source":"local","content":rendered,"html":html}
                except Exception as e:
                    logger.exception("render local failed: %s", e)
                    return {"ok":False,"source":"local","error":str(e)}
        if self.github_repo_url:
            rem = self.fetch_github_template(template_name, templates_path)
            if rem is not None:
                try:
                    tpl = Template(rem)
                    rendered = tpl.render(**ctx)
                    html = md.markdown(rendered)
                    return {"ok":True,"source":"github","content":rendered,"html":html}
                except Exception as e:
                    logger.exception("render github failed: %s", e)
                    return {"ok":False,"source":"github","error":str(e)}
        return {"ok":False,"error":"template not found"}

    
def export_pdf_from_html(self, html: str, out_path: str) -> Dict:
    """Export HTML to PDF. Try pdfkit (wkhtmltopdf). Fallback to ReportLab-based markdown/pdf conversion if available."""

    # First try pdfkit (wkhtmltopdf) if available
    if PDFKIT_AVAILABLE:
        try:
            pdfkit.from_string(html, out_path)
            return {"ok": True, "path": out_path}
        except Exception as e:
            logger.exception("pdfkit generation failed: %s", e)
            # continue to fallback if possible

    # Fallback: try to convert via markdown -> plain paragraphs -> ReportLab
    if REPORTLAB_AVAILABLE:
        try:
            # Very simple conversion: strip tags and split into paragraphs
            # Prefer to use the markdown source if available; here we attempt to strip HTML tags
            import re as _re
            # remove common tags
            text = _re.sub(r'<\s*br\s*/?>', '\n', html)
            text = _re.sub(r'<[^>]+>', '', text)
            parts = [p.strip() for p in text.split('\n\n') if p.strip()]

            doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
            styles = getSampleStyleSheet()
            story = []
            for p in parts:
                # limit paragraph length per ReportLab
                # replace multiple newlines
                para_text = p.replace('\n', '<br/>')
                story.append(Paragraph(para_text, styles['Normal']))
                story.append(Spacer(1, 6))
            doc.build(story)
            return {"ok": True, "path": out_path, "method": "reportlab_fallback"}
        except Exception as e:
            logger.exception("ReportLab generation failed: %s", e)
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "pdfkit and reportlab not available"}

    def generate(self, template_name: str, ctx: Dict, prefer_local: bool = True, save_markdown: Optional[str] = None, save_pdf: Optional[str] = None) -> Dict:
        res = self.render_template(template_name, ctx, prefer_local=prefer_local)
        if not res.get("ok"):
            return res
        out = {"ok":True, "source":res.get("source"), "markdown":res.get("content")}
        if save_markdown:
            os.makedirs(os.path.dirname(save_markdown), exist_ok=True)
            with open(save_markdown, "w", encoding="utf-8") as f:
                f.write(res.get("content"))
            out["markdown_path"] = save_markdown
        if save_pdf:
            html = res.get("html") or md.markdown(res.get("content"))
            pdf_res = self.export_pdf_from_html(html, save_pdf)
            out["pdf"] = pdf_res
        return out

    def extract_uploaded_zip(self, zip_path: str, target_dir: Optional[str] = None) -> Dict:
        try:
            if not target_dir:
                target_dir = self.LOCAL_TEMPLATE_PATH
            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(target_dir)
            return {"ok":True, "extracted_to": target_dir}
        except Exception as e:
            logger.exception("zip extract failed: %s", e)
            return {"ok":False, "error":str(e)}