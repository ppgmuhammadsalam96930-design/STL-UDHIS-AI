# backend.py — patched for STL Quantum frontend (HTTP + WebSocket, multi-provider)
# Replace your current backend.py with this file (or merge changes).
import os, json, time, asyncio, logging
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import websockets
from websockets.server import WebSocketServerProtocol

# load env
HTTP_HOST = os.getenv("HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("HTTP_PORT", "5000"))
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5000").split(",") if o.strip()]
SERVER_KEYS_JSON = os.getenv("SERVER_KEYS_JSON", None)
SERVER_KEYS: Dict[str,str] = json.loads(SERVER_KEYS_JSON) if SERVER_KEYS_JSON else {}

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "180"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backend")

# ---------- rate limiter ----------
class RateLimiter:
    def __init__(self, per_minute:int):
        self.per_minute = per_minute
        self.buckets = {}
        self.lock = asyncio.Lock()
    async def allow(self, ip:str) -> bool:
        async with self.lock:
            now = time.time()
            b = self.buckets.get(ip)
            if not b:
                self.buckets[ip] = {"tokens": self.per_minute - 1, "last": now}
                return True
            elapsed = now - b["last"]
            refill = (elapsed/60.0) * self.per_minute
            b["tokens"] = min(self.per_minute, b["tokens"] + refill)
            b["last"] = now
            if b["tokens"] >= 1:
                b["tokens"] -= 1
                return True
            return False

rate_limiter = RateLimiter(RATE_LIMIT_PER_MIN)

# ---------- FastAPI ----------
app = FastAPI(title="STL Quantum AI Proxy")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS or ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ProxyRequest(BaseModel):
    apiKey: Optional[str] = None
    body: Dict[str,Any] = {}
    model: Optional[str] = None

# ---------- provider callers ----------
async def call_openai(api_key:str, body:Dict[str,Any], timeout=HTTP_TIMEOUT):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error {resp.status_code}")
        return resp.json()

async def call_anthropic(api_key:str, body:Dict[str,Any], timeout=HTTP_TIMEOUT):
    url = "https://api.anthropic.com/v1/complete"
    headers = {"x-api-key": api_key, "Content-Type":"application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Anthropic error {resp.status_code}")
        return resp.json()

async def call_custom(url:str, api_key:Optional[str], payload:Dict[str,Any], timeout=HTTP_TIMEOUT):
    headers = {"Content-Type":"application/json"}
    if api_key: headers["x-api-key"] = api_key
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail="Custom provider error")
        return resp.json()

# ---------- HTTP endpoints ----------
@app.get("/ping")
async def ping():
    return {"pong": True, "ts": datetime.utcnow().isoformat()+"Z", "ws": f"ws://{WS_HOST}:{WS_PORT}"}

@app.post("/api/v1/{provider}")
async def proxy(provider: str, payload: ProxyRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not await rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    api_key = payload.apiKey or SERVER_KEYS.get(provider)
    if not api_key:
        raise HTTPException(status_code=400, detail="no api key available for provider")
    body = payload.body or {}
    try:
        if provider.lower() in ("openai","gpt","openai_chat"):
            res = await call_openai(api_key, body)
        elif provider.lower() in ("anthropic",):
            res = await call_anthropic(api_key, body)
        elif provider.lower() == "custom":
            url = body.get("url")
            if not url: raise HTTPException(status_code=400, detail="body.url required for custom")
            res = await call_custom(url, api_key, body.get("payload", {}))
        else:
            if provider.startswith("http"):
                res = await call_custom(provider, api_key, body)
            else:
                raise HTTPException(status_code=400, detail="unknown provider")
        return JSONResponse({"ok": True, "result": res})
    except HTTPException as he: raise he
    except Exception as e:
        logger.exception("HTTP proxy error")
        raise HTTPException(status_code=500, detail="internal proxy error")

# Simple proxy used by frontend's Hybrid Proxy Hook (quantum proxy)
@app.post("/api/proxy")
async def simple_proxy(payload: Dict[str,Any], request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not await rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    provider = payload.get("provider") or payload.get("aiProvider") or "openai"
    api_key = payload.get("apiKey") or SERVER_KEYS.get(provider)
    prompt = payload.get("prompt") or payload.get("message") or payload.get("text") or ""
    model = payload.get("model") or payload.get("modelName") or "gpt-4"
    if provider.lower() in ("openai","gpt","openai_chat"):
        if not api_key:
            raise HTTPException(status_code=400, detail="no api key")
        body = {"model": model, "messages": [{"role":"user","content":prompt}], "max_tokens": payload.get("max_tokens",512), "temperature": payload.get("temperature",0.7)}
        res = await call_openai(api_key, body)
        # try to extract reply
        reply = None
        if isinstance(res, dict):
            c = res.get("choices")
            if c and isinstance(c, list) and len(c)>0:
                first = c[0]
                if first.get("message"):
                    reply = first["message"].get("content")
                else:
                    reply = first.get("text") or json.dumps(first)
        if reply is None:
            reply = json.dumps(res)[:2000]
        return {"ok": True, "reply": reply, "raw": res}
    else:
        if provider.startswith("http"):
            res = await call_custom(provider, api_key, payload)
            return {"ok": True, "reply": str(res)}
        return {"ok": True, "reply": f"[mock reply] provider={provider} prompt={prompt[:120]}"}

# ---------- WebSocket server ----------
WS_CLIENTS = {}
WS_LOCK = asyncio.Lock()

async def default_ws_handler(ws: WebSocketServerProtocol, path: str):
    peer = ws.remote_address
    client_id = f"{peer[0]}:{peer[1]}:{path}"
    async with WS_LOCK:
        WS_CLIENTS[client_id] = ws
    logger.info("WS CONNECT %s (path=%s total=%d)", client_id, path, len(WS_CLIENTS))
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send(json.dumps({"type":"error","message":"invalid json"}))
                continue
            typ = msg.get("type")
            if typ == "ping":
                await ws.send(json.dumps({"type":"pong","ts":datetime.utcnow().isoformat()+"Z"})); continue
            if typ == "ai_request":
                provider = msg.get("provider") or "openai"
                body = msg.get("body") or {}
                api_key = msg.get("apiKey") or SERVER_KEYS.get(provider)
                if not api_key:
                    await ws.send(json.dumps({"type":"error","message":"provider or apiKey missing"}))
                    continue
                if not await rate_limiter.allow(client_id.split(":")[0]):
                    await ws.send(json.dumps({"type":"error","message":"rate limit exceeded"})); continue
                # background handler
                asyncio.create_task(handle_ai_request_ws(ws, client_id, provider, api_key, body))
                continue
            if typ == "subscribe_ui":
                await ws.send(json.dumps({"type":"subscribed","what":"ui_events"})); continue
            # echo fallback
            await ws.send(json.dumps({"type":"echo","payload":msg}))
    except websockets.exceptions.ConnectionClosed:
        logger.info("WS CLOSED %s", client_id)
    except Exception as e:
        logger.exception("WS loop error")
    finally:
        async with WS_LOCK:
            WS_CLIENTS.pop(client_id, None)
        logger.info("WS DISCONNECT %s (remaining=%d)", client_id, len(WS_CLIENTS))

async def handle_ai_request_ws(ws: WebSocketServerProtocol, client_id:str, provider:str, api_key:str, body:Dict[str,Any]):
    logger.info("WS ai_request provider=%s client=%s", provider, client_id)
    try:
        if provider.lower() in ("openai","gpt","openai_chat"):
            result = await call_openai(api_key, body)
        elif provider.lower() in ("anthropic",):
            result = await call_anthropic(api_key, body)
        else:
            if provider.startswith("http"):
                result = await call_custom(provider, api_key, body)
            else:
                await ws.send(json.dumps({"type":"error","message":"unknown provider"})); return
        await ws.send(json.dumps({"type":"ai_response","ok": True, "result": result}))
    except Exception as e:
        logger.exception("ai_request_ws failed")
        try:
            await ws.send(json.dumps({"type":"error","message": str(e)}))
        except: pass

async def broadcast_ws(message: str):
    async with WS_LOCK:
        clients = list(WS_CLIENTS.items())
    for cid, ws in clients:
        try:
            if ws and not ws.closed:
                await ws.send(message)
        except Exception:
            logger.exception("broadcast failed for %s", cid)

# ---------- runner ----------
async def start_ws_server_single(host, port, handler):
    logger.info("Starting WS on %s:%d", host, port)
    async with websockets.serve(handler, host, port, ping_interval=20, ping_timeout=10, max_size=2**20):
        await asyncio.Future()

async def start_http_app():
    import uvicorn
    config = uvicorn.Config("backend:app", host=HTTP_HOST, port=HTTP_PORT, log_level="info", loop="asyncio", lifespan="off")
    server = uvicorn.Server(config)
    logger.info("Starting HTTP server on %s:%d", HTTP_HOST, HTTP_PORT)
    await server.serve()

async def start_ws_servers():
    await start_ws_server_single(WS_HOST, WS_PORT, default_ws_handler)

async def main():
    await asyncio.gather(start_ws_servers(), start_http_app())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host'); parser.add_argument('--http-port', type=int); parser.add_argument('--ws-port', type=int)
    args = parser.parse_args()
    if args.host: HTTP_HOST = args.host
    if args.http_port: HTTP_PORT = args.http_port
    if args.ws_port: WS_PORT = args.ws_port
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")
