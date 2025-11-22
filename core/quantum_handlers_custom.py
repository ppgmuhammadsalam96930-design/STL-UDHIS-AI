
# quantum_handlers_custom.py
# Custom handler extensions for QuantumEngine

from Quantum_engine import QuantumEngine

class CustomQuantumEngine(QuantumEngine):
    async def _handle_button_click(self, component_id, payload):
        # Override example
        if component_id == "btn_special":
            return {"action": "special_button", "message": "Special button clicked!"}
        return await super()._handle_button_click(component_id, payload)

    async def _handle_input_change(self, component_id, payload):
        if component_id == "username_input":
            value = payload.get("data", {}).get("value", "")
            return {"action": "username_updated", "value": value}
        return await super()._handle_input_change(component_id, payload)

if __name__ == "__main__":
    engine = CustomQuantumEngine()
    import asyncio
    asyncio.run(engine.start_server())
