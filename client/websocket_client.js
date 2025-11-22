// websocket_client.js — Quantum WS Client Final v3
(function () {
    const WS_URL = "ws://localhost:8765";
    let ws = null;
    let reconnectDelay = 1500;

    function connect() {
        ws = new WebSocket(WS_URL);
        window.quantumWebSocket = ws;

        ws.addEventListener("open", () => {
            console.info("🔌 Connected to Quantum Python Engine");
            document.getElementById && document.getElementById('quantum-conn') && (document.getElementById('quantum-conn').textContent = 'yes');
            ws.send(JSON.stringify({ type: "ping" }));
        });

        ws.addEventListener("message", (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                document.dispatchEvent(
                    new CustomEvent("QuantumEngineMessage", { detail: msg })
                );
            } catch (e) {
                console.warn("WS message parse error:", e);
            }
        });

        ws.addEventListener("close", () => {
            console.warn("Quantum WebSocket closed — reconnecting…");
            document.getElementById && document.getElementById('quantum-conn') && (document.getElementById('quantum-conn').textContent = 'no');
            setTimeout(connect, reconnectDelay);
        });

        ws.addEventListener("error", () => {
            console.warn("WebSocket error — retrying…");
            try{ ws.close(); }catch(e){}
        });
    }

    // Public API
    window.QuantumEngineBridge = {
        send: function (obj) {
            try {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(obj));
                } else {
                    console.warn("WS not ready, skipping:", obj);
                }
            } catch (e) {
                console.error("WS send error:", e);
            }
        }
    };

    // Forward QuantumEngineMessage to chat UI and handle auto-wiring injection
    document.addEventListener("QuantumEngineMessage", (ev) => {
        const msg = ev.detail;
        if(!msg) return;
        try {
            if (msg.type === "connection_established") {
                // inject auto-wiring script if provided
                if (msg.auto_wiring_script) {
                    try { eval(msg.auto_wiring_script); } catch(e){ console.error("Auto-wiring eval error:", e); }
                }
            }
            if (msg.type === "ai_chat_response" && msg.message) {
                if (window.appendAIChatBubble) window.appendAIChatBubble(msg.message);
            }
            // also forward other messages in console for debugging
            console.debug("QuantumEngineMessage processed:", msg.type);
        } catch(e) {}
    });

    connect();
})();
