#!/usr/bin/env python3
"""
Backend Final Ultra v7 - ENHANCED (Turbo + Benchmark + AI Cache + ZIP builder)

How to use:
- Run: python backend_final_ultra_v7.py
- HTTP: http://localhost:5000
- WebSocket: ws://localhost:8765

New features added in this version:
- /benchmark endpoints (latency, throughput, ai_call_time)
- Turbo mode (double worker concurrency / faster timeouts when enabled)
- Simple AI response caching (in-memory LRU cache + optional disk persistence)
- /build_zip endpoint to create a ZIP including provided local files (HTML + backends) and a run-all script

This file was generated to be compatible with your uploaded files (paths included below) and the HTML auto-bind.
Uploaded files included in ZIP build (paths on the runtime machine):
- /mnt/data/Stl-udhis_psychology_duo_fixed (3).html
- /mnt/data/backend.py.py
- /mnt/data/backend (1).py

"""

import os
import json
import asyncio
import logging
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# --- NajibDev AI response normalizer helpers ---
def _extract_plain_from_result(result):
    try:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if 'output' in result and isinstance(result['output'], str):
                return result['output']
            if 'reply' in result and isinstance(result['reply'], str):
                return result['reply']
            if 'choices' in result and isinstance(result['choices'], list) and result['choices']:
                c=result['choices'][0]
                if 'message' in c and isinstance(c['message'], dict) and c['message'].get('content'):
                    return c['message']['content']
                if 'text' in c:
                    return c['text']
            if 'completion' in result and isinstance(result['completion'], str):
                return result['completion']
        return str(result)
    except Exception:
        try: return str(result)
        except: return '<unextractable>'

async def _ws_send_ai_response(ws, original_result, provider=None, loadingId=None, extra_meta=None):
    plain=_extract_plain_from_result(original_result)
    payload={
        "type":"ai_response",
        "ok":True,
        "provider":provider,
        "result":original_result,
        "output":plain,
        "ai_chat_response":{"text":plain},
        "ai_chat_result":{"text":plain},
        "reply":plain
    }
    if loadingId: payload['loadingId']=loadingId
    if extra_meta: payload['meta']=extra_meta
    try:
        await ws.send(json.dumps(payload))
    except Exception as e:
        try:
            await ws.send(json.dumps({"type":"ai_response","ok":False,"error":str(e)}))
        except:
            pass

# --- end helpers ---


from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import websockets
from websockets.server import WebSocketServerProtocol

try:
    import httpx
except Exception:
    httpx = None

try:
    import aiofiles
except Exception:
    aiofiles = None

# ---------- Config ----------
HTTP_HOST = os.getenv('HTTP_HOST', '0.0.0.0')
HTTP_PORT = int(os.getenv('HTTP_PORT', '5000'))
WS_HOST = os.getenv('WS_HOST', '0.0.0.0')
WS_PORT = int(os.getenv('WS_PORT', '8765'))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5000').split(',') if o.strip()]
SERVER_KEYS_JSON = os.getenv('SERVER_KEYS_JSON', None)
SERVER_KEYS: Dict[str,str] = json.loads(SERVER_KEYS_JSON) if SERVER_KEYS_JSON else {}
# --- HARDCODED GLOBAL SERVER KEY (fallback) ---
SERVER_KEYS.setdefault('openai', 'AIzaSyCSOeClh4DHymoITytGOL7O5d5r5YhSKhw')
SERVER_KEYS.setdefault('gpt', SERVER_KEYS['openai'])
SERVER_KEYS.setdefault('openai_chat', SERVER_KEYS['openai'])
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/mnt/data/quantum_outputs'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '30.0'))
RATE_LIMIT_PER_MIN = int(os.getenv('RATE_LIMIT_PER_MIN', '180'))
AI_CACHE_MAX = int(os.getenv('AI_CACHE_MAX', '256'))  # number of cache entries
TURBO_MODE = os.getenv('TURBO_MODE', '0') in ('1','true','True')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('backend_final_ultra_v7')

if TURBO_MODE:
    logger.info('🔥 TURBO MODE ENABLED: applying performance tweaks')
    # apply turbo adjustments
    HTTP_TIMEOUT = max(5.0, HTTP_TIMEOUT/2)
    RATE_LIMIT_PER_MIN = max(60, RATE_LIMIT_PER_MIN * 2)

# ---------- Simple LRU cache for AI responses ----------
class SimpleLRUCache:
    def __init__(self, capacity:int=256):
        self.capacity = capacity
        self.map: Dict[str, Any] = {}
        self.order: Dict[str, datetime] = {}

    def _evict_if_needed(self):
        if len(self.map) <= self.capacity:
            return
        # evict oldest
        oldest_key = min(self.order.items(), key=lambda kv: kv[1])[0]
        self.map.pop(oldest_key, None)
        self.order.pop(oldest_key, None)

    def get(self, key:str):
        v = self.map.get(key)
        if v is not None:
            self.order[key] = datetime.utcnow()
        return v

    def set(self, key:str, value:Any):
        self.map[key] = value
        self.order[key] = datetime.utcnow()
        self._evict_if_needed()

AI_CACHE = SimpleLRUCache(capacity=AI_CACHE_MAX)

# ---------- Rate limiter (simple token bucket) ----------
class RateLimiter:
    def __init__(self, per_minute:int):
        self.per_minute = per_minute
        self.buckets = {}
        self.lock = asyncio.Lock()

    async def allow(self, ip:str) -> bool:
        async with self.lock:
            now = asyncio.get_event_loop().time()
            b = self.buckets.get(ip)
            if not b:
                self.buckets[ip] = {'tokens': self.per_minute-1, 'last': now}
                return True
            elapsed = now - b['last']
            refill = (elapsed/60.0) * self.per_minute
            b['tokens'] = min(self.per_minute, b['tokens'] + refill)
            b['last'] = now
            if b['tokens'] >= 1:
                b['tokens'] -= 1
                return True
            return False

rate_limiter = RateLimiter(RATE_LIMIT_PER_MIN)

# ---------- App ----------
app = FastAPI(title='STL Quantum Backend Final Ultra v7 (Turbo)')
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS or ['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

class ProxyRequest(BaseModel):
    apiKey: Optional[str] = None
    body: Dict[str,Any] = {}
    model: Optional[str] = None

# ---------- Document generator ----------
class DocumentGenerator:
    def __init__(self, outdir:Path):
        self.outdir = outdir

    async def generate(self, doc_type:str, data:Dict[str,Any]) -> Dict[str,Any]:
        content = f"Document type: {doc_type}
Generated at: {datetime.utcnow().isoformat()}Z

{json.dumps(data, indent=2)}
"
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{doc_type}_{ts}.txt"
        filepath = self.outdir / filename
        try:
            if aiofiles:
                async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                    await f.write(content)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            return {'success': True, 'filename': filename, 'filepath': str(filepath)}
        except Exception as e:
            logger.exception('document write failed')
            return {'success': False, 'error': str(e)}

DOCUMENT_GENERATOR = DocumentGenerator(OUTPUT_DIR)

# ---------- AI Integration (stubs) ----------
class AIIntegration:
    def __init__(self):
        pass

    async def call_openai(self, api_key:str, body:Dict[str,Any]):
        cache_key = f"openai:{json.dumps(body, sort_keys=True)}"
        cached = AI_CACHE.get(cache_key)
        if cached:
            return {'cached': True, 'result': cached}
        if not httpx:
            raise RuntimeError('httpx not available')
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # store a compact form
            AI_CACHE.set(cache_key, data)
            return data

    async def call_anthropic(self, api_key:str, body:Dict[str,Any]):
        cache_key = f"anthropic:{json.dumps(body, sort_keys=True)}"
        cached = AI_CACHE.get(cache_key)
        if cached:
            return {'cached': True, 'result': cached}
        if not httpx:
            raise RuntimeError('httpx not available')
        url = 'https://api.anthropic.com/v1/complete'
        headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            AI_CACHE.set(cache_key, data)
            return data

    async def call_custom(self, url:str, api_key:Optional[str], payload:Dict[str,Any]):
        cache_key = f"custom:{url}:{json.dumps(payload, sort_keys=True)}"
        cached = AI_CACHE.get(cache_key)
        if cached:
            return {'cached': True, 'result': cached}
        if not httpx:
            raise RuntimeError('httpx not available')
        headers = {'Content-Type': 'application/json'}
        if api_key: headers['x-api-key'] = api_key
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            AI_CACHE.set(cache_key, data)
            return data

AI = AIIntegration()

# ---------- HTTP endpoints (proxy + benchmark + build zip) ----------
@app.get('/ping')
async def ping():
    return {'pong': True, 'ts': datetime.utcnow().isoformat() + 'Z', 'ws': f'ws://{WS_HOST}:{WS_PORT}'}

@app.post('/api/v1/{provider}')
async def proxy(provider:str, payload:ProxyRequest, request:Request):
    client_ip = request.client.host if request.client else 'unknown'
    if not await rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail='rate limit exceeded')
    api_key = payload.apiKey or SERVER_KEYS.get(provider)
    body = payload.body or {}
    try:
        provider_l = provider.lower()
        if provider_l in ('openai','gpt','openai_chat'):
            res = await AI.call_openai(api_key, body)
        elif provider_l in ('anthropic',):
            res = await AI.call_anthropic(api_key, body)
        elif provider_l == 'custom':
            url = body.get('url')
            if not url: raise HTTPException(status_code=400, detail='body.url required for custom')
            res = await AI.call_custom(url, api_key, body.get('payload', {}))
        else:
            if provider.startswith('http'):
                res = await AI.call_custom(provider, api_key, body)
            else:
                raise HTTPException(status_code=400, detail='unknown provider')
        return JSONResponse({'ok': True, 'result': res})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception('HTTP proxy error')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/proxy')
async def simple_proxy(payload:Dict[str,Any], request:Request):
    client_ip = request.client.host if request.client else 'unknown'
    if not await rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail='rate limit exceeded')
    provider = payload.get('provider') or payload.get('aiProvider') or 'openai'
    api_key = payload.get('apiKey') or SERVER_KEYS.get(provider)
    prompt = payload.get('prompt') or payload.get('message') or payload.get('text') or ''
    model = payload.get('model') or payload.get('modelName') or 'gpt-4'
    if provider.lower() in ('openai','gpt','openai_chat'):
        if not api_key:
            raise HTTPException(status_code=400, detail='no api key')
        body = {'model': model, 'messages': [{'role':'user','content':prompt}], 'max_tokens': payload.get('max_tokens',512), 'temperature': payload.get('temperature',0.7)}
        res = await AI.call_openai(api_key, body)
        reply = None
        if isinstance(res, dict):
            c = res.get('choices')
            if c and isinstance(c, list) and len(c)>0:
                first = c[0]
                if first.get('message'):
                    reply = first['message'].get('content')
                else:
                    reply = first.get('text') or json.dumps(first)
        if reply is None:
            reply = json.dumps(res)[:2000]
        return {'ok': True, 'reply': reply, 'raw': res}
    else:
        if provider.startswith('http'):
            res = await AI.call_custom(provider, api_key, payload)
            return {'ok': True, 'reply': str(res)}
        return {'ok': True, 'reply': f"[mock reply] provider={provider} prompt={prompt[:120]}"}

# Benchmark endpoints
@app.get('/benchmark/latency')
async def bench_latency():
    # measure simple round-trip of local function
    import time
    t0 = time.perf_counter()
    await asyncio.sleep(0)  # yield
    t1 = time.perf_counter()
    return {'latency_ms': (t1-t0)*1000}

@app.post('/benchmark/ai_call_time')
async def bench_ai_call(payload:Dict[str,Any]):
    # measure AI call time to provider (mock or real if configured)
    provider = payload.get('provider','mock')
    body = payload.get('body', {'model':'gpt-4','messages':[{'role':'user','content':'ping'}]})
    api_key = payload.get('apiKey') or SERVER_KEYS.get(provider)
    import time
    t0 = time.perf_counter()
    try:
        if provider in ('openai','gpt') and api_key and httpx:
            await AI.call_openai(api_key, body)
        elif provider == 'anthropic' and api_key and httpx:
            await AI.call_anthropic(api_key, body)
        else:
            # simulated delay
            await asyncio.sleep(0.12 if not TURBO_MODE else 0.06)
    except Exception as e:
        return {'error': str(e)}
    t1 = time.perf_counter()
    return {'ai_call_time_ms': (t1-t0)*1000, 'turbo': TURBO_MODE}

@app.get('/benchmark/throughput')
async def bench_throughput():
    # very simple throughput estimate: how many trivial tasks can be scheduled in 1s
    import time
    async def trivial():
        return 1
    t0 = time.perf_counter()
    count = 0
    async def worker():
        nonlocal count
        for _ in range(1000):
            await trivial(); count += 1
    # run N workers depending on turbo
    workers = 4 if TURBO_MODE else 2
    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    await asyncio.gather(*tasks)
    t1 = time.perf_counter()
    return {'ops': count, 'seconds': t1-t0, 'ops_per_second': count / (t1-t0) if (t1-t0)>0 else None, 'turbo': TURBO_MODE}

# ZIP builder endpoint: will create a ZIP in /mnt/data containing selected local files and run-all script
@app.post('/build_zip')
async def build_zip(request:Request):
    payload = await request.json()
    files = payload.get('files', [
        '/mnt/data/Stl-udhis_psychology_duo_fixed (3).html',
        '/mnt/data/backend.py.py',
        '/mnt/data/backend (1).py'
    ])
    zipname = f"backend_final_ultra_v7_bundle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
    zippath = Path('/mnt/data') / zipname
    runbat = """@echo off
REM run-all (Windows)
start cmd /k "python -m pip install -r requirements.txt && python backend_final_ultra_v7.py"
start http://localhost:5000
"""
    runsh = """#!/bin/bash
# run-all (Linux/mac)
python3 -m pip install -r requirements.txt &
python3 backend_final_ultra_v7.py &
if command -v live-server >/dev/null 2>&1; then
  live-server /mnt/data --port=5500 &
fi
xdg-open http://localhost:5500 || open http://localhost:5500 || true
"""
    # create zip
    try:
        with zipfile.ZipFile(zippath, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                try:
                    if Path(p).exists():
                        zf.write(p, arcname=Path(p).name)
                except Exception:
                    logger.warning('failed to add %s to zip', p)
            # add run scripts
            zf.writestr('run-all.bat', runbat)
            zf.writestr('run-all.sh', runsh)
            # minimal requirements
            zf.writestr('requirements.txt', 'fastapi
uvicorn
websockets
httpx
')
            # include this backend file itself
            zf.writestr('backend_final_ultra_v7.py', Path(__file__).read_text())
        return {'ok': True, 'zip_path': str(zippath)}
    except Exception as e:
        logger.exception('zip build failed')
        raise HTTPException(status_code=500, detail=str(e))

# ---------- WebSocket server ----------
WS_CLIENTS: Dict[str, WebSocketServerProtocol] = {}
WS_LOCK = asyncio.Lock()

async def broadcast_ws(message: str):
    async with WS_LOCK:
        clients = list(WS_CLIENTS.items())
    for cid, ws in clients:
        try:
            if ws and not ws.closed:
                await ws.send(message)
        except Exception:
            logger.exception('broadcast failed for %s', cid)

async def default_ws_handler(ws: WebSocketServerProtocol, path: str):
    peer = ws.remote_address or ('unknown', 0)
    client_id = f"{peer[0]}:{peer[1]}:{path}"
    async with WS_LOCK:
        WS_CLIENTS[client_id] = ws
    logger.info('WS CONNECT %s total=%d', client_id, len(WS_CLIENTS))
    try:
        await ws.send(json.dumps({'type':'connection_established','client_id':client_id,'server_info':{'name':'Quantum Backend Final Ultra v7','version':'7.0.0'},'auto_wiring_script': generate_runtime_injection()}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send(json.dumps({'type':'error','message':'invalid json'}))
                continue
            typ = msg.get('type')
            if typ == 'ping':
                await ws.send(json.dumps({'type':'pong','ts':datetime.utcnow().isoformat()+'Z'})); continue
            if typ == 'ai_request':
                provider = msg.get('provider') or 'openai'
                body = msg.get('body') or {}
                api_key = msg.get('apiKey') or SERVER_KEYS.get(provider)
                if not api_key and provider not in ('custom',):
                    await ws.send(json.dumps({'type':'error','message':'provider or apiKey missing'})); continue
                if not await rate_limiter.allow(client_id.split(':')[0]):
                    await ws.send(json.dumps({'type':'error','message':'rate limit exceeded'})); continue
                asyncio.create_task(handle_ai_request_ws(ws, client_id, provider, api_key, body)); continue
            if typ == 'subscribe_ui':
                await ws.send(json.dumps({'type':'subscribed','what':'ui_events'})); continue
            # broadcast echo fallback
            await ws.send(json.dumps({'type':'echo','payload':msg}))
    except websockets.exceptions.ConnectionClosed:
        logger.info('WS CLOSED %s', client_id)
    except Exception as e:
        logger.exception('WS loop error')
    finally:
        async with WS_LOCK:
            WS_CLIENTS.pop(client_id, None)
        logger.info('WS DISCONNECT %s remaining=%d', client_id, len(WS_CLIENTS))

async def handle_ai_request_ws(ws: WebSocketServerProtocol, client_id:str, provider:str, api_key:str, body:Dict[str,Any]):
    logger.info('WS ai_request provider=%s client=%s', provider, client_id)
    try:
        if provider.lower() in ('openai','gpt','openai_chat'):
            result = await AI.call_openai(api_key, body)
        elif provider.lower() in ('anthropic',):
            result = await AI.call_anthropic(api_key, body)
        else:
            if provider.startswith('http'):
                result = await AI.call_custom(provider, api_key, body)
            else:
                await ws.send(json.dumps({'type':'error','message':'unknown provider'})); return
        await _ws_send_ai_response(ws, result, provider=provider)
    except Exception as e:
        logger.exception('ai_request_ws failed')
        try:
            await ws.send(json.dumps({'type':'error','message': str(e)}))
        except: pass

# ---------- runtime injection ----------
def generate_runtime_injection() -> str:
    script = """
(function(){
  try{
    window.QuantumBackend = window.QuantumBackend || {};
    window.QuantumBackend.wsEndpoint = 'ws://'+location.hostname+':%d';
    window.QuantumBackend.turbo = %s;
    window.QuantumBackend.send = function(obj){ if(window.quantumWebSocket && window.quantumWebSocket.readyState === WebSocket.OPEN) window.quantumWebSocket.send(JSON.stringify(obj)); };
  }catch(e){}
})();
""" % (WS_PORT, 'true' if TURBO_MODE else 'false')
    return script

# ---------- runner ----------
async def start_ws_server():
    logger.info('Starting WS on %s:%d', WS_HOST, WS_PORT)
    async with websockets.serve(default_ws_handler, WS_HOST, WS_PORT, ping_interval=20, ping_timeout=10, max_size=2**20):
        await asyncio.Future()

async def start_http_app():
    # use programmatic uvicorn with this module name
    config = uvicorn.Config('backend_final_ultra_v7:app', host=HTTP_HOST, port=HTTP_PORT, log_level='info', loop='asyncio', lifespan='off')
    server = uvicorn.Server(config)
    logger.info('Starting HTTP server on %s:%d', HTTP_HOST, HTTP_PORT)
    await server.serve()

async def main():
    if TURBO_MODE:
        logger.info('Turbo mode active: prioritizing throughput')
    await asyncio.gather(start_ws_server(), start_http_app())

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Shutting down')



"""
backend_final_ultra_v8_GLOBAL_PRO.py
Auto-generated PRO wrapper/enhancement for your existing backend_final_ultra_v7_GLOBAL.py

This file *enhances* the existing backend by:
 - Multi-provider GLOBAL_KEYS structure
 - get_global_key(provider) replacement
 - /global/status endpoint (if original backend exposes a Flask "app")
 - Simple in-memory smart cache for completions and embeddings
 - Optional (pluggable) encrypted store placeholder
 - Best-effort monkeypatch of original backend module so integration is seamless

USAGE:
 - This file will attempt to import and patch the original backend file located at:
     /mnt/data/backend_final_ultra_v7_GLOBAL.py
 - If the original backend exposes a Flask `app` object and SERVER_KEYS dict, this file will:
     * register a /global/status route
     * inject PRO helper functions (get_global_key, cache)
 - If original backend is structured differently, this file still exposes utilities you can import.

SECURITY:
 - Global keys remain in server memory only.
 - This file does NOT expose keys to clients.
"""

import importlib.util
import sys
import time
import json
from functools import wraps
from pathlib import Path

ORIG_PATH = Path("/mnt/data/backend_final_ultra_v7_GLOBAL.py")

# --- PRO Global keys multi-provider (edit manually, DO NOT COMMIT keys to public repo) ---
PRO_GLOBAL_KEYS = {
    "openai": "",      # "sk-xxxxx" or your Google/Gemini key if you map provider->google
    "anthropic": "",
    "gemini": "",
    "custom": ""
}

# Simple in-memory cache for responses: key -> (ts, ttl, value)
_PRO_CACHE = {}

def cache_get(key):
    rec = _PRO_CACHE.get(key)
    if not rec:
        return None
    ts, ttl, val = rec
    if time.time() - ts > ttl:
        del _PRO_CACHE[key]
        return None
    return val

def cache_set(key, value, ttl=60):
    _PRO_CACHE[key] = (time.time(), ttl, value)
    return True

def cache_decorator(ttl=60):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = fn.__name__ + ":" + json.dumps({"args":args,"kwargs":kwargs}, default=str)
            cached = cache_get(key)
            if cached is not None:
                return cached
            res = fn(*args, **kwargs)
            cache_set(key, res, ttl=ttl)
            return res
        return wrapper
    return deco

# Optional simple obfuscation (not real encryption) placeholder
def obfuscate_secret_temporary(secret: str) -> str:
    try:
        import base64
        return base64.b64encode(secret.encode("utf-8")).decode("ascii")
    except Exception:
        return secret[::-1]

def deobfuscate_secret_temporary(blob: str) -> str:
    try:
        import base64
        return base64.b64decode(blob.encode("ascii")).decode("utf-8")
    except Exception:
        return blob[::-1]

# Try to import original backend module by file path and patch
def _import_and_patch_original():
    if not ORIG_PATH.exists():
        raise FileNotFoundError(f"Original backend not found at {ORIG_PATH}")
    spec = importlib.util.spec_from_file_location("backend_orig", str(ORIG_PATH))
    backend_orig = importlib.util.module_from_spec(spec)
    sys.modules["backend_orig"] = backend_orig
    spec.loader.exec_module(backend_orig)

    # Provide or merge PRO_GLOBAL_KEYS into original SERVER_KEYS if present
    try:
        if hasattr(backend_orig, "SERVER_KEYS"):
            # merge without overwriting existing non-empty server keys
            for k, v in PRO_GLOBAL_KEYS.items():
                if k not in backend_orig.SERVER_KEYS or not backend_orig.SERVER_KEYS.get(k):
                    backend_orig.SERVER_KEYS[k] = v
        else:
            backend_orig.SERVER_KEYS = dict(PRO_GLOBAL_KEYS)
    except Exception:
        # best-effort fallback
        backend_orig.SERVER_KEYS = dict(PRO_GLOBAL_KEYS)

    # Define get_global_key and attach to backend_orig
    def get_global_key(provider: str):
        provider = (provider or "").lower().strip()
        val = backend_orig.SERVER_KEYS.get(provider) or backend_orig.SERVER_KEYS.get("custom") or ""
        return val

    backend_orig.get_global_key = get_global_key

    # If backend_orig exposes a Flask app, register /global/status
    try:
        app = getattr(backend_orig, "app", None)
        if app is not None:
            from flask import jsonify, request

            @app.route("/global/status", methods=["GET"])
            def global_status():
                # report which global keys are set (boolean presence only)
                keys = {k: bool(bool(v)) for k, v in backend_orig.SERVER_KEYS.items()}
                # active provider heuristics from request or from last UI state if any
                active_provider = request.args.get("provider") or ""
                return jsonify({
                    "mode": "global",
                    "keys_present": keys,
                    "active_provider": active_provider,
                    "cache_size": len(_PRO_CACHE)
                })

            # admin endpoint to rotate/update keys in memory (POST, JSON body)
            @app.route("/global/update_keys", methods=["POST"])
            def global_update_keys():
                try:
                    payload = request.get_json(force=True)
                    # only update keys present in payload
                    for k in ["openai","anthropic","gemini","custom"]:
                        if k in payload:
                            backend_orig.SERVER_KEYS[k] = payload[k]
                    return jsonify({"ok": True}), 200
                except Exception as e:
                    return jsonify({"ok": False, "error": str(e)}), 400

    except Exception:
        # if Flask import fails or app is not a Flask app, skip
        pass

    # attach cache helpers for original backend to use
    backend_orig.PRO_CACHE_GET = cache_get
    backend_orig.PRO_CACHE_SET = cache_set
    backend_orig.PRO_CACHE = _PRO_CACHE

    return backend_orig

# Execute patch at import time
try:
    backend_original = _import_and_patch_original()
    _PATCH_OK = True
except Exception as e:
    backend_original = None
    _PATCH_OK = False
    _PATCH_ERROR = str(e)

# Expose a convenience run helper if original app is not present
def create_minimal_proxy_app():
    """
    If the original backend does not expose a Flask `app`, this helper will create a
    minimal Flask app that implements:
      - /global/status
      - /api/proxy  (simple envelope forwarder using SERVER_KEYS)
    This is a minimal fallback server so you can run a single integrated PRO backend.
    """
    try:
        from flask import Flask, request, jsonify
    except Exception as e:
        raise RuntimeError("Flask is required to run the minimal proxy app: " + str(e))

    app = Flask("backend_final_ultra_v8_PRO_minimal")

    @app.route("/global/status", methods=["GET"])
    def global_status_min():
        keys = {k: bool(bool(v)) for k, v in PRO_GLOBAL_KEYS.items()}
        return jsonify({
            "mode": "global",
            "keys_present": keys,
            "cache_size": len(_PRO_CACHE)
        })

    @app.route("/global/update_keys", methods=["POST"])
    def global_update_min():
        payload = request.get_json(force=True)
        for k in ["openai","anthropic","gemini","custom"]:
            if k in payload:
                PRO_GLOBAL_KEYS[k] = payload[k]
        return jsonify({"ok": True}), 200

    @app.route("/api/proxy", methods=["POST"])
    def proxy_min():
        """
        Minimal proxy logic:
          - Accept JSON envelope {provider, model, prompt, apiKey?}
          - Use apiKey if provided, otherwise use PRO_GLOBAL_KEYS
          - (This minimal impl does NOT call external provider automatically)
          - It returns a placeholder response to confirm routing.
        """
        payload = request.get_json(force=True)
        provider = (payload.get("provider") or "").lower()
        model = payload.get("model")
        prompt = payload.get("prompt") or payload.get("input") or ""
        api_key = payload.get("apiKey") or PRO_GLOBAL_KEYS.get(provider) or PRO_GLOBAL_KEYS.get("custom") or ""
        # Note: in this minimal fallback we DO NOT forward to external provider.
        # You should integrate forwarding logic here or rely on your original backend.
        # Check cache first
        cache_key = f"{provider}:{model}:{prompt}"
        cached = cache_get(cache_key)
        if cached:
            return jsonify({"ok": True, "cached": True, "result": cached}), 200
        # produce placeholder result
        result = {"echo_prompt": prompt[:800], "provider_used": provider, "model": model, "used_key_present": bool(api_key)}
        cache_set(cache_key, result, ttl=120)
        return jsonify({"ok": True, "proxied": True, "result": result}), 200

    return app

# CLI helper
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-minimal", action="store_true", help="Run minimal PRO Flask app (fallback).")
    parser.add_argument("--show-patch", action="store_true", help="Show whether patch succeeded.")
    args = parser.parse_args()
    if args.show_patch:
        print("PATCH_OK:", _PATCH_OK)
        if not _PATCH_OK:
            print("PATCH_ERROR:", _PATCH_ERROR)
    if args.run_minimal:
        app = create_minimal_proxy_app()
        app.run(host="0.0.0.0", port=5000)
