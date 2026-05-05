from langchain_core.callbacks import BaseCallbackHandler
from ..core.tracer import BaseTracer


class LangChainTracer(BaseCallbackHandler, BaseTracer):
    """Attach to any LangChain/LangGraph agent via config={'callbacks': [LangChainTracer(...)]}"""

    def __init__(self, version: str, task_id: str = ""):
        BaseTracer.__init__(self, version, task_id)
        self._turn = 0

    def on_chat_model_start(self, serialized, messages, **kw):
        try:
            content = messages[-1][-1].content
        except (IndexError, AttributeError):
            content = str(messages)
        self.log_llm_start(content)

    def on_llm_end(self, response, **kw):
        try:
            gen = response.generations[0][0]
            text = gen.text or ""
            tool_calls = getattr(gen.message, "additional_kwargs", {}).get("tool_calls", [])
            tool = tool_calls[0]["function"]["name"] if tool_calls else ""
        except (IndexError, AttributeError):
            text, tool = "", ""
        self.log_llm_end(text, tool_selected=tool)

    def on_tool_start(self, serialized, input_str, **kw):
        self._turn += 1
        name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        self.log_tool_start(name, {"input": input_str}, self._turn)

    def on_tool_end(self, output, **kw):
        self.log_tool_end(str(output))

    def on_chain_end(self, outputs, **kw):
        # Only capture root chain end (has 'output' or 'messages' key)
        if isinstance(outputs, dict) and "output" in outputs:
            self.log_final(outputs["output"], self._turn, True)
        elif isinstance(outputs, dict) and "messages" in outputs:
            msgs = outputs["messages"]
            last = msgs[-1] if msgs else None
            if last and hasattr(last, "content"):
                self.log_final(last.content, self._turn, True)
