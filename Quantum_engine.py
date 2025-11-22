#!/usr/bin/env python3
"""
Quantum Engine v2 - Hybrid AI Integration (OpenAI library + HTTP for Anthropic/Gemini + Custom endpoint)
Features:
- WebSocket server for UI integration
- ai_config_update route to receive provider/model/api_key from frontend
- Supports OpenAI via `openai` library (if installed)
- Supports Anthropic & Gemini via HTTP (aiohttp)
- Custom endpoints via HTTP POST
- Per-client config storage (so each websocket can have its own API key / provider)
- Graceful error handling and fallbacks
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import websockets
import aiohttp

# Optional: try to import openai library; if missing, we fallback with clear error message
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumEngineV2")


class QuantumEngineV2:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.connected_clients = set()
        # store per-websocket config keyed by client_id (uuid)
        self.client_configs: Dict[str, Dict[str, Any]] = {}
        # simple message routing map
        self.routes = {
            "ping": self._handle_ping,
            "ai_config_update": self._handle_ai_config_update,
            "ai_chat_message": self._handle_ai_chat,
            "telemetry_request": self._handle_telemetry_request
        }

    async def _handle_ping(self, message, ws, client_id):
        return {"type": "pong", "timestamp": datetime.now().isoformat()}

    async def _handle_ai_config_update(self, message, ws, client_id):
        # message expected: { type: "ai_config_update", provider, model, api_key, custom_endpoint(optional) }
        cfg = {
            "provider": message.get("provider"),
            "model": message.get("model"),
            "api_key": message.get("api_key"),
            "custom_endpoint": message.get("custom_endpoint")
        }
        self.client_configs[client_id] = cfg
        logger.info(f"[{client_id}] Saved AI config: provider={cfg.get('provider')} model={cfg.get('model')}")
        return {"type": "ai_config_ack", "status": "ok", "config": {"provider": cfg.get("provider"), "model": cfg.get("model")}}

    async def _handle_ai_chat(self, message, ws, client_id):
        user_message = message.get("message", "")
        cfg = self.client_configs.get(client_id, {})
        provider = cfg.get("provider", "openai")
        model = cfg.get("model", None)
        api_key = cfg.get("api_key", None)
        custom_endpoint = cfg.get("custom_endpoint", None)

        # simple validation
        if provider is None or api_key is None and provider != "custom":
            return {"type": "ai_chat_response", "error": "AI provider or API key not configured. Use ai_config_update first."}

        try:
            ai_text = await self._generate_ai_response(provider, model, api_key, custom_endpoint, user_message)
            return {"type": "ai_chat_response", "message": ai_text, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.exception("Error generating AI response")
            return {"type": "ai_chat_response", "error": str(e)}

    async def _handle_telemetry_request(self, message, ws, client_id):
        # minimal telemetry example
        return {"type": "telemetry_response", "connected_clients": len(self.connected_clients), "timestamp": datetime.now().isoformat()}

    async def _generate_ai_response(self, provider: str, model: Optional[str], api_key: Optional[str], custom_endpoint: Optional[str], prompt: str) -> str:
        provider = (provider or "openai").lower()

        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("openai library not installed on server. Install 'openai' package or switch to custom/http mode.")
            # configure key and call OpenAI ChatCompletion (synchronous library wrapped in asyncio)
            openai.api_key = api_key
            # prefer chat completions route; fallback to completion if model provided differently
            try:
                # use run_in_executor to avoid blocking if openai is sync-only
                loop = asyncio.get_event_loop()
                def call_openai_sync():
                    # fallback: use ChatCompletion if available
                    try:
                        resp = openai.ChatCompletion.create(
                            model=model or "gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=800
                        )
                        # support both choices/message and text return
                        if isinstance(resp, dict) and "choices" in resp and len(resp["choices"])>0:
                            # ChatCompletion: choices[0].message.content
                            c0 = resp["choices"][0]
                            if "message" in c0 and "content" in c0["message"]:
                                return c0["message"]["content"]
                            if "text" in c0:
                                return c0["text"]
                        # fallback string
                        return str(resp)
                    except Exception as e:
                        raise

                ai_text = await loop.run_in_executor(None, call_openai_sync)
                return ai_text
            except Exception as e:
                logger.exception("OpenAI call failed")
                raise

        elif provider in ("anthropic", "claude"):
            # Call Anthropic Claude over HTTP. API spec sometimes varies by version - we use v1/messages style as used in UI.
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": api_key, "Content-Type": "application/json"}
            body = {"model": model or "claude-2", "messages": [{"role": "user", "content": prompt}]}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body, timeout=30) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"Anthropic API error: {resp.status} {text}")
                    # attempt parse json
                    try:
                        j = await resp.json()
                        # try common paths
                        if "content" in j:
                            return j.get("content")
                        if "choices" in j and len(j["choices"])>0:
                            c = j["choices"][0]
                            if "message" in c and "content" in c["message"]:
                                return c["message"]["content"]
                            if "text" in c:
                                return c["text"]
                        return json.dumps(j)
                    except Exception:
                        return text

        elif provider in ("google", "gemini"):
            # Google Gemini: use custom HTTP endpoint pattern or require user to use custom_endpoint
            # For safety, prefer custom_endpoint for Gemini usage.
            if custom_endpoint:
                url = custom_endpoint
            else:
                url = f"https://generativemodels.googleapis.com/v1/models/{model or 'gemini'}:generate"
            headers = {"Content-Type": "application/json"}
            # If user provided api_key, use it as Bearer token
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {"input": prompt, "maxOutputTokens": 800}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"Gemini API error: {resp.status} {text}")
                    try:
                        j = await resp.json()
                        # parse potential fields
                        if "candidates" in j and len(j["candidates"])>0:
                            return j["candidates"][0].get("content", j["candidates"][0])
                        if "output" in j:
                            return j["output"].get("text", json.dumps(j["output"]))
                        return json.dumps(j)
                    except Exception:
                        return text

        elif provider == "custom":
            # Custom endpoint is required
            if not custom_endpoint:
                raise ValueError("Custom provider requires 'custom_endpoint' in config.")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {"prompt": prompt, "model": model}
            async with aiohttp.ClientSession() as session:
                async with session.post(custom_endpoint, headers=headers, json=payload, timeout=30) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"Custom endpoint error: {resp.status} {text}")
                    # try to parse as json and find common fields
                    try:
                        j = await resp.json()
                        if isinstance(j, dict):
                            if "message" in j: return j["message"]
                            if "output" in j and isinstance(j["output"], str): return j["output"]
                            if "choices" in j and len(j["choices"])>0:
                                c = j["choices"][0]
                                if isinstance(c, dict) and "text" in c: return c["text"]
                        return json.dumps(j)
                    except Exception:
                        return text
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def handler(self, websocket, path):
        client_id = str(uuid.uuid4())
        self.connected_clients.add(websocket)
        # default empty config until client sends ai_config_update
        self.client_configs[client_id] = {}
        logger.info(f"Client connected: {client_id}")

        # send initial connection info
        init_msg = {"type": "connection_established", "client_id": client_id, "timestamp": datetime.now().isoformat()}
        await websocket.send(json.dumps(init_msg))

        try:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except Exception:
                    await websocket.send(json.dumps({"type": "error", "message": "invalid_json"}))
                    continue

                mtype = message.get("type")
                handler = self.routes.get(mtype)
                if not handler:
                    await websocket.send(json.dumps({"type": "error", "message": f"unknown_type:{mtype}"}))
                    continue

                try:
                    response = await handler(message, websocket, client_id)
                    if response is not None:
                        await websocket.send(json.dumps(response))
                except Exception as e:
                    logger.exception("Handler error")
                    await websocket.send(json.dumps({"type": "error", "message": str(e)}))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        finally:
            self.connected_clients.discard(websocket)
            if client_id in self.client_configs:
                del self.client_configs[client_id]

    async def start(self):
        logger.info(f"Starting Quantum Engine v2 on {self.host}:{self.port}")
        server = await websockets.serve(self.handler, self.host, self.port)
        await server.wait_closed()


if __name__ == "__main__":
    engine = QuantumEngineV2(host="0.0.0.0", port=8765)
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        logger.info("Shutting down Quantum Engine V2")
