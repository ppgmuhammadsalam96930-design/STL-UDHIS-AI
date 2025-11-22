# API Routes (WebSocket) — Quantum Engine

This document lists the WebSocket message types supported by `Quantum_engine.py`.

## Message types

### `ping`
- Client: `{ "type": "ping" }`
- Server: `{ "type": "pong", "timestamp": "..." }`

### `component_event`
- Sent by client when UI components fire events.
- Example:
```json
{
  "type": "component_event",
  "payload": {
    "componentId": "auto_button_abc",
    "eventType": "click",
    "data": { "value": "" }
  }
}
```

### `quantum_island_update`
- UI notifications forwarded to server.
- Example:
```json
{ "type": "quantum_island_update", "payload": { "message":"...", "duration": 3000 } }
```

### `telemetry_request`
- Client requests server telemetry.
- Example:
```json
{ "type": "telemetry_request" }
```

### `ai_chat_message`
- Send chat messages for AI processing.
- Example:
```json
{ "type": "ai_chat_message", "message": "Bagaimana membuat RPP tentang bilangan?" }
```

### `theme_change`
- Request theme application.
- Example:
```json
{ "type": "theme_change", "theme": { "name":"Quantum Dark", "vars": { "--quantum-bg":"#000" } } }
```

### `rpp_list_templates`
- Request list of templates (local + github).
- Example:
```json
{ "type": "rpp_list_templates" }
```

### `rpp_generate`
- Generate RPP from template.
- Example:
```json
{
  "type":"rpp_generate",
  "template":"templates/k13/smp_kelas_7.html",
  "context": {
    "subject":"Matematika",
    "grade":"7",
    "topic":"Bilangan Bulat",
    "duration":"2 x 45 menit"
  },
  "options": {
    "prefer_local": true,
    "save_markdown": "outputs/rpp_kelas7.md",
    "save_pdf": "outputs/rpp_kelas7.pdf"
  }
}
```

### Server Responses
Server responds with `{ "type": "<message>_response", ... }` messages, e.g. `rpp_generate_response`.

