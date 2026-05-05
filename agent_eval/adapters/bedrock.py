from ..core.tracer import BaseTracer


class BedrockTracer:
    """
    Wraps boto3 bedrock-agent-runtime invoke_agent.

    Usage:
        import boto3
        client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        agent = BedrockTracer(client, agent_id="...", alias_id="...", version="v1")
        result = agent.invoke("What is the GDP of India?", session_id="sess-1")
    """

    def __init__(self, client, agent_id: str, alias_id: str, version: str, task_id: str = ""):
        self._client = client
        self._agent_id = agent_id
        self._alias_id = alias_id
        self._tracer = BaseTracer(version=version, task_id=task_id)

    def invoke(self, input_text: str, session_id: str) -> str:
        self._tracer.log_llm_start(input_text)
        response = self._client.invoke_agent(
            agentId=self._agent_id,
            agentAliasId=self._alias_id,
            sessionId=session_id,
            inputText=input_text,
        )
        chunks = []
        for event in response.get("completion", []):
            if "chunk" in event:
                chunks.append(event["chunk"]["bytes"].decode())
        output = "".join(chunks)
        self._tracer.log_llm_end(output)
        self._tracer.log_final(output, total_turns=1, success=bool(output))
        return output
