"""
AUTO-SYNC HTML module for backend_final_ultra_v8_GLOBAL_PRO

Features:
- /sync_html  : POST JSON {"file_path": "/mnt/data/....html"} -> parses file server-side with BeautifulSoup and extracts script ids, script srcs, data-* attrs, known patterns (najibdev, DEV_mga13, __MGA_LEGACY_KEY, registerLegacyApiKey) and returns structured report.
- /sync_report: POST JSON from client (injected script) with live runtime info (scriptIds, sessionApiKey (client may provide), registeredLegacyKey) -> stores securely to OUTPUT_DIR and can optionally call AI or update SERVER_KEYS.
- helper functions: parse_html_file, find_register_legacy_key_calls, safe_write_report
- includes an "injection snippet" string (JS) that you can paste into your HTML to report runtime script ids/session info back to the backend at /sync_report. Use carefully (exposes client-session info only if the page runs it).

Dependencies:
- beautifulsoup4
- (already present dependencies used by backend: fastapi, uvicorn, httpx)

Security notes:
- The server will NOT automatically accept arbitrary keys from clients to become persistent API keys without explicit admin confirmation. The sync endpoints only write reports to OUTPUT_DIR and return findings. If you opt-in to auto-store keys, be aware of the security consequences.

How to use:
1) Place this file alongside your existing backend and import or include the router.
2) Restart backend.
3) To run a server-side parse of an uploaded HTML file:
   POST /sync_html  with JSON {"file_path":"/mnt/data/Stl-udhis_psychology_duo_fixed.html"}

4) To enable client-side reporting, inject the provided INJECTION_SNIPPET into the HTML (paste before </body>) — the snippet posts a minimal report to /sync_report.

This file was generated referencing your uploaded HTML at: /mnt/data/Stl-udhis_psychology_duo_fixed.html
"""

from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger('backend_auto_sync')
router = APIRouter()

OUTPUT_DIR = Path('/mnt/data/quantum_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- heuristics / patterns ---
NAJIB_PATTERNS = [re.compile(r'najib', re.I), re.compile(r'DEV_mga13', re.I), re.compile(r'DEV_mga14', re.I), re.compile(r'__MGA_LEGACY_KEY', re.I), re.compile(r'registerLegacyApiKey', re.I)]
SESSION_KEY_NAME = '__MGA_LEGACY_KEY'

def safe_write_report(name: str, data: dict) -> str:
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    fname = f"sync_report_{name}_{ts}.json"
    path = OUTPUT_DIR / fname
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(path)
    except Exception as e:
        logger.exception('write fail')
        raise

def find_register_legacy_key_calls(js_text: str):
    # crude search for registerLegacyApiKey('...') or .registerLegacyApiKey("...")
    results = []
    try:
        for m in re.finditer(r"registerLegacyApiKey\(\s*(['\"])(.*?)\1\s*\)", js_text, re.I|re.S):
            results.append(m.group(2))
    except Exception:
        pass
    return results

def parse_html_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(text, 'html.parser')

    scripts = []
    for s in soup.find_all('script'):
        sid = s.get('id')
        src = s.get('src')
        inline = s.string or ''
        attrs = {k: v for k, v in s.attrs.items() if k not in ('id', 'src')}
        match_tags = [p.pattern for p in NAJIB_PATTERNS if (sid and p.search(sid)) or (src and p.search(src or '')) or (inline and p.search(inline))]
        register_calls = find_register_legacy_key_calls(inline or '')
        scripts.append({
            'id': sid,
            'src': src,
            'attrs': attrs,
            'has_najib_pattern': bool(match_tags),
            'register_calls': register_calls,
            'inline_preview': (inline or '').strip()[:300]
        })

    # find data-* attributes across DOM that look like ai targets
    ai_targets = []
    for el in soup.find_all(attrs=True):
        for k, v in el.attrs.items():
            if k.startswith('data-') and isinstance(v, str) and re.search(r'mga13|newai|quantum|ai|legacy', v, re.I):
                ai_targets.append({'tag': el.name, 'attr': k, 'value': v, 'outer': str(el)[:300]})

    # find obvious global names
    text_join = text
    legacy_key_literals = re.findall(r'__MGA_LEGACY_KEY', text_join)

    report = {
        'file': str(path),
        'scripts': scripts,
        'ai_targets': ai_targets,
        'legacy_key_literal_found': bool(legacy_key_literals),
        'summary': {
            'script_count': len(scripts),
            'ai_target_count': len(ai_targets)
        }
    }
    return report

# --- Injection snippet to be embedded in HTML (client-side) ---
INJECTION_SNIPPET = """
<!-- AUTO-SYNC INJECTION SNIPPET (minimal, paste before </body>) -->
<script>
(function(){
  try{
    const payload = { scriptIds:[], scripts:[] };
    document.querySelectorAll('script').forEach(s=>{
      payload.scriptIds.push(s.id||null);
      payload.scripts.push({id:s.id||null, src:s.src||null, inline:(s.textContent||'').slice(0,300)});
    });
    // try to read sessionStorage legacy key (if page put it there)
    try{ const v=sessionStorage.getItem('__MGA_LEGACY_KEY'); if(v) payload.legacy_session_key = atob(v); }catch(e){}
    // post report to backend
    if(navigator.sendBeacon){
      try{ navigator.sendBeacon('/sync_report', JSON.stringify(payload)); }catch(e){}
    }
    // fallback fetch
    fetch('/sync_report',{method:'POST',headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)}).catch(()=>{});
  }catch(e){}
})();
</script>
"""

@router.post('/sync_html')
async def sync_html(request: Request):
    try:
        j = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail='invalid json')
    fp = j.get('file_path') or j.get('path')
    if not fp:
        raise HTTPException(status_code=400, detail='file_path required')
    path = Path(fp)
    try:
        report = parse_html_file(path)
        saved = safe_write_report('server_parse', report)
        return JSONResponse({'ok': True, 'report': report, 'saved_to': saved})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='file not found')
    except Exception as e:
        logger.exception('sync_html failed')
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/sync_report')
async def sync_report(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail='invalid json')
    # sanitize payload minimaly
    allowed = { 'scriptIds','scripts','legacy_session_key','meta' }
    clean = {k: v for k, v in payload.items() if k in allowed}
    # store
    try:
        saved = safe_write_report('client_report', clean)
    except Exception as e:
        logger.exception('save client report fail')
        raise HTTPException(status_code=500, detail=str(e))

    # optionally: if legacy_session_key present, do NOT auto-promote to SERVER_KEYS; just return suggestion
    has_key = 'legacy_session_key' in clean and clean['legacy_session_key']
    return JSONResponse({'ok': True, 'saved_to': saved, 'contains_legacy_session_key': bool(has_key)})

# Export quick helper for including into main backend app
def include_router(app):
    app.include_router(router)

# also expose the injection snippet for convenience
def get_injection_snippet():
    return INJECTION_SNIPPET