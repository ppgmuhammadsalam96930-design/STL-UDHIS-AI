# quantum_rpp_handler_features.py
# Full RPP handler with features: OpenAI, PDF, DOCX, PNG/JPG, QR, watermark, signature, DB, progress, templates, zip

import os, asyncio, json, hashlib, zipfile, tempfile
from datetime import datetime
from pathlib import Path

# Optional libs
try:
    from weasyprint import HTML as WeasyHTML
except Exception:
    WeasyHTML = None
try:
    import pdfkit
except Exception:
    pdfkit = None
try:
    from docx import Document
except Exception:
    Document = None
try:
    from html2image import Html2Image
except Exception:
    Html2Image = None
try:
    import qrcode
except Exception:
    qrcode = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    import pymongo
except Exception:
    pymongo = None
try:
    import psycopg2
except Exception:
    psycopg2 = None
try:
    from PIL import Image
except Exception:
    Image = None

from Quantum_engine import QuantumEngine

OUT_DIR = Path(os.environ.get('QUANTUM_OUTPUT_DIR', '/mnt/data/rpp_full_features_outputs'))
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = Path(__file__).parent / 'templates'

def get_mongo_client():
    uri = os.getenv('MONGO_URI')
    if not uri or not pymongo:
        return None
    try:
        client = pymongo.MongoClient(uri)
        return client
    except Exception:
        return None

def save_metadata_db(doc):
    try:
        client = get_mongo_client()
        if client:
            db = client.get_default_database() if getattr(client, 'get_default_database', None) else client['quantum']
            coll = db.get_collection('rpp_exports')
            coll.insert_one(doc)
            return {'db':'mongo','ok':True}
    except Exception:
        pass
    purl = os.getenv('POSTGRES_URL')
    if purl and psycopg2:
        try:
            conn = psycopg2.connect(purl)
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS rpp_exports (id SERIAL PRIMARY KEY, base_name TEXT, assets JSONB, metadata JSONB, created_at TIMESTAMP);''')
            cur.execute('INSERT INTO rpp_exports (base_name, assets, metadata, created_at) VALUES (%s,%s,%s,%s)', (doc.get('base_name'), json.dumps(doc.get('assets',{})), json.dumps(doc.get('metadata',{})), datetime.utcnow()))
            conn.commit()
            cur.close()
            conn.close()
            return {'db':'postgres','ok':True}
        except Exception:
            pass
    return {'db': None, 'ok': False}

async def ai_generate(prompt: str):
    key = os.getenv('OPENAI_API_KEY')
    if not key or not OpenAI:
        return {'text': f'[AI STUB] No OPENAI_API_KEY or openai not installed. Prompt:\n{prompt}', 'metadata':{'stub':True}}
    try:
        client = OpenAI(api_key=key)
        res = client.chat.completions.create(model='gpt-4.1', messages=[{'role':'system','content':'You are an RPP generator assistant.'},{'role':'user','content':prompt}], max_tokens=1500)
        text = res.choices[0].message.content
        return {'text': text, 'metadata':{'engine':'openai'}}
    except Exception as e:
        return {'text': f'[AI ERROR] {e}', 'metadata':{'error':str(e)}}

async def emit_progress(engine, base_name, step, detail):
    payload = {'base_name': base_name, 'step': step, 'detail': detail, 'ts': datetime.utcnow().isoformat()}
    try:
        if hasattr(engine, 'broadcast_progress'):
            try:
                maybe = engine.broadcast_progress(payload)
                if asyncio.iscoroutine(maybe):
                    await maybe
                return
            except Exception:
                pass
    except Exception:
        pass
    try:
        p = OUT_DIR / f'progress_{base_name}.json'
        p.write_text(json.dumps(payload), encoding='utf-8')
    except Exception:
        pass

def inject_watermark(html, watermark_text):
    if not watermark_text:
        return html
    css = f"""<style>.quantum-watermark { { } } </style>"""
    # For simplicity, place a basic watermark div
    css = """<style>.quantum-watermark{position:fixed;left:0;top:40%;width:100%;text-align:center;opacity:0.08;font-size:72px;transform:rotate(-30deg);pointer-events:none;z-index:9999;color:#000;}</style><div class='quantum-watermark'>%s</div>"""
    return html.replace('<body>', '<body>' + css % watermark_text, 1) if '<body>' in html.lower() else css % watermark_text + html

def sign_file_sha256(path: Path):
    try:
        b = path.read_bytes()
        h = hashlib.sha256(b).hexdigest()
        sig_path = path.with_suffix(path.suffix + '.sig.txt')
        sig_path.write_text(json.dumps({'sha256':h, 'signed_at': datetime.utcnow().isoformat()}), encoding='utf-8')
        return str(sig_path), h
    except Exception:
        return None, None

def make_assets_zip(base_name, assets: dict):
    zip_path = OUT_DIR / f'{base_name}_all_assets.zip'
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for k,v in assets.items():
                if isinstance(v, str):
                    p = Path(v)
                    if p.exists():
                        z.write(p, arcname=p.name)
        return str(zip_path)
    except Exception:
        return None

class FullFeaturesRPPEngine(QuantumEngine):
    async def _handle_button_click(self, component_id, payload):
        attrs = payload.get('elementInfo', {}).get('attributes', {}) or {}
        action = attrs.get('data-action') or attrs.get('data_action') or ''
        filename = attrs.get('data-filename') or attrs.get('data_filename') or 'rpp_export'
        template_key = attrs.get('data-template') or attrs.get('data-kelas') or attrs.get('data-jenjang') or ''
        watermark = attrs.get('data-watermark') or os.getenv('DEFAULT_WATERMARK', 'STL - RPP')
        signer = attrs.get('data-signer') or os.getenv('DEFAULT_SIGNER', 'QuantumEngine')
        prompt = payload.get('payload', {}).get('value') or payload.get('data', {}).get('value', '') or 'Buat RPP sederhana'

        if not action.startswith('rpp-'):
            return await super()._handle_button_click(component_id, payload)

        safe = ''.join(c for c in filename if c.isalnum() or c in ('_','-')).strip() or 'rpp'
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        base_name = f"{safe}_{ts}"

        await emit_progress(self, base_name, 'start', 'Starting generation')

        # Template selection
        html_template = None
        if template_key:
            try_keys = [template_key, f'sd_kelas_{template_key}', f'{template_key}.html']
            for k in try_keys:
                p = TEMPLATES_DIR / str(k)
                if p.exists():
                    html_template = p.read_text(encoding='utf-8')
                    break
        if not html_template:
            ai_res = await ai_generate(prompt)
            content_html = "<div class='q-content'>" + ai_res['text'].replace('\n','<br>') + "</div>"
            html_full = f"<html><head><meta charset='utf-8'></head><body>{content_html}</body></html>"
        else:
            ai_res = await ai_generate(prompt)
            html_full = html_template.replace('{{content}}', ai_res['text'].replace('\n','<br>'))

        await emit_progress(self, base_name, 'ai_done', 'AI generation complete')

        html_full = inject_watermark(html_full, watermark)
        await emit_progress(self, base_name, 'watermark', 'Watermark applied')

        html_path = OUT_DIR / f'{base_name}.html'
        html_path.write_text(html_full, encoding='utf-8')
        assets = {'html': str(html_path)}
        await emit_progress(self, base_name, 'saved_html', str(html_path))

        pdf_path = OUT_DIR / f'{base_name}.pdf'
        if pdfkit:
            try:
                pdfkit.from_string(html_full, str(pdf_path))
                assets['pdf'] = str(pdf_path)
            except Exception as e:
                assets['pdf_error'] = str(e)
        elif WeasyHTML:
            try:
                WeasyHTML(string=html_full).write_pdf(str(pdf_path))
                assets['pdf'] = str(pdf_path)
            except Exception as e:
                assets['pdf_error'] = str(e)
        else:
            assets['pdf'] = 'no_pdf_engine'
        await emit_progress(self, base_name, 'pdf_done', assets.get('pdf') or assets.get('pdf_error'))

        if 'pdf' in assets and assets['pdf'] and assets['pdf'] != 'no_pdf_engine':
            sig_path, sha = sign_file_sha256(Path(assets['pdf']))
            if sig_path:
                assets['pdf_sig'] = sig_path
                assets['pdf_sha256'] = sha
        await emit_progress(self, base_name, 'signed', assets.get('pdf_sig','none'))

        docx_path = OUT_DIR / f'{base_name}.docx'
        if Document:
            try:
                doc = Document()
                for line in ai_res['text'].split('\n'):
                    doc.add_paragraph(line)
                doc.save(docx_path)
                assets['docx'] = str(docx_path)
            except Exception as e:
                assets['docx_error'] = str(e)
        await emit_progress(self, base_name, 'docx_done', assets.get('docx') or assets.get('docx_error'))

        png_path = OUT_DIR / f'{base_name}.png'
        jpg_path = OUT_DIR / f'{base_name}.jpg'
        if Html2Image:
            try:
                hti = Html2Image(output_path=str(OUT_DIR))
                hti.screenshot(html_str=html_full, save_as=png_path.name)
                assets['png'] = str(png_path)
                try:
                    if Image:
                        im = Image.open(png_path)
                        im.convert('RGB').save(jpg_path, 'JPEG', quality=90)
                        assets['jpg'] = str(jpg_path)
                except Exception as e:
                    assets['jpg_error'] = str(e)
            except Exception as e:
                assets['png_error'] = str(e)
        else:
            assets['images'] = 'no_image_engine'
        await emit_progress(self, base_name, 'images_done', assets.get('png') or assets.get('png_error'))

        if qrcode:
            try:
                qr_path = OUT_DIR / f'{base_name}-qr.png'
                img = qrcode.make(f'RPP:{base_name}')
                img.save(qr_path)
                assets['qr'] = str(qr_path)
            except Exception as e:
                assets['qr_error'] = str(e)
        await emit_progress(self, base_name, 'qr_done', assets.get('qr') or assets.get('qr_error'))

        zip_path = make_assets_zip(base_name, assets)
        if zip_path:
            assets['zip'] = zip_path
        await emit_progress(self, base_name, 'zip_done', zip_path or 'zip_failed')

        metadata = {'prompt': prompt, 'watermark': watermark, 'signer': signer, 'template': template_key, 'generated_at': datetime.utcnow().isoformat()}
        doc = {'base_name': base_name, 'assets': assets, 'metadata': metadata, 'created_at': datetime.utcnow().isoformat()}
        db_res = save_metadata_db(doc)

        await emit_progress(self, base_name, 'db_saved', db_res)

        return {'type':'rpp_assets', 'status':'generated', 'assets': assets, 'metadata': metadata, 'db': db_res}

if __name__ == '__main__':
    engine = FullFeaturesRPPEngine()
    asyncio.run(engine.start_server())
