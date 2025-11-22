// websocket_client.js - simple Quantum WS client
(function(){
  const WS_URL = "ws://localhost:8765";
  let ws = new WebSocket(WS_url=WS_URL);
  window.quantumWebSocket = ws;
  ws.addEventListener('open', ()=> {
    console.info('Connected to Quantum Engine at', WS_URL);
    ws.send(JSON.stringify({type:'ping'}));
  });
  ws.addEventListener('message', ev => {
    try{
      const msg = JSON.parse(ev.data);
      console.debug('QuantumWS <-', msg);
      document.dispatchEvent(new CustomEvent('QuantumEngineMessage', { detail: msg }));
    }catch(e){}
  });
  ws.addEventListener('close', ()=> console.warn('Quantum WS closed'));
  window.QuantumEngineBridge = {
    send(obj){
      if(ws && ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify(obj));
      else console.warn('WS not ready', obj);
    }
  };
})();