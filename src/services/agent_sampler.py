import json
import os
from datetime import datetime
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart, ToolCallPart, ToolReturnPart
from src.services.service_types import SamplerBase, SamplerResponse, MessageList
from src.services.base_agent import BaseAgent

class AgentAsSampler(SamplerBase):
    """
    A wrapper to make BaseAgent compatible with the SamplerBase interface.
    It also handles the conversion of pydantic-ai message formats and counts search calls.
    """
    def __init__(self, agent: BaseAgent, metrics_source=None, query_log_path=None):
        self.agent = agent
        self.metrics_source = metrics_source
        self.query_log_path = query_log_path
        self.evaluation_count = 0

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # The last user message is the input
        user_input = ""
        for msg in reversed(message_list):
            if msg['role'] == 'user':
                user_input = msg['content']
                break
        
        # BaseAgent.run returns a pydantic-ai ModelResponse object
        response = self.agent.run(user_input)
        
        pydantic_ai_messages = response.all_messages()
        
        search_call_count = 0
        for msg in pydantic_ai_messages:
            if isinstance(msg, ModelResponse):
                for p in msg.parts:
                    if isinstance(p, ToolCallPart) and p.tool_name == 'search':
                        search_call_count += 1
        
        converted_messages = self._convert_messages(pydantic_ai_messages)

        # stop_reason is None for normal completions; set to the exception name (e.g.
        # "UsageLimitExceeded") when BaseAgent salvaged a best-effort answer from a question that
        # hit the loop cap or timed out, so those rows can be filtered out of analysis.
        response_metadata = {'search_calls': search_call_count,
                             'stop_reason': getattr(response, 'stop_reason', None)}

        # Increment evaluation counter for every call
        self.evaluation_count += 1

        # Log query information if available
        if self.metrics_source and hasattr(self.metrics_source, 'get_and_clear_query_log'):
            query_logs = self.metrics_source.get_and_clear_query_log()
            if self.query_log_path:
                # Always create a log entry, even if no queries were made
                log_entry = {
                    'evaluation_iteration': self.evaluation_count,
                    'timestamp': datetime.now().isoformat(),
                    'queries': query_logs  # Will be empty list if no searches made
                }
                
                # Append to log file
                if os.path.exists(self.query_log_path):
                    with open(self.query_log_path, 'r') as f:
                        all_logs = json.load(f)
                else:
                    all_logs = []
                
                all_logs.append(log_entry)
                
                with open(self.query_log_path, 'w') as f:
                    json.dump(all_logs, f, indent=2)

        return SamplerResponse(
            response_text=response,
            actual_queried_message_list=converted_messages,
            response_metadata=response_metadata
        )

    def _convert_messages(self, messages: list) -> MessageList:
        message_list = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        message_list.append({'role': 'user', 'content': str(part.content)})
                    elif isinstance(part, ToolReturnPart):
                        message_list.append({
                            'role': 'tool', 
                            'content': str(part.content), 
                            'name': part.tool_name
                        })
            elif isinstance(msg, ModelResponse):
                text_content = "\n".join([p.content for p in msg.parts if isinstance(p, TextPart) and p.content])
                tool_calls = []
                for p in msg.parts:
                    if isinstance(p, ToolCallPart):
                        tool_calls.append({
                            'id': p.tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': p.tool_name,
                                'arguments': str(p.args)
                            }
                        })
                
                assistant_message = {'role': 'assistant', 'content': text_content or None}
                if tool_calls:
                    assistant_message['tool_calls'] = tool_calls
                
                if assistant_message['content'] or 'tool_calls' in assistant_message:
                    message_list.append(assistant_message)
        return message_list

    async def acall(self, message_list: MessageList) -> SamplerResponse:
        *history_msgs, last = message_list
        user_input = last['content'] if last['role'] == 'user' else ""
        if not user_input:
            for msg in reversed(message_list):
                if msg['role'] == 'user':
                    user_input = msg['content']
                    break

        # Prior conversation turns (e.g. an unrelated chit-chat prefix, or a mocked search
        # exchange) get converted to pydantic-ai message history so the model actually sees
        # them, not just the final question. Turns come in two shapes: plain {role, content}
        # text, or {role, parts} for a turn that made/received a tool call (role "assistant"
        # for a tool_call/text part, role "tool" for the matching tool_response part).
        message_history = None
        if history_msgs:
            message_history = []
            for msg in history_msgs:
                role = msg['role']
                parts = msg.get('parts')
                if parts is None:
                    if role == 'user':
                        message_history.append(ModelRequest(parts=[UserPromptPart(content=msg['content'])]))
                    elif role == 'assistant':
                        message_history.append(ModelResponse(parts=[TextPart(content=msg['content'])]))
                    continue
                for p in parts:
                    if p['type'] == 'tool_call' and role == 'assistant':
                        message_history.append(ModelResponse(parts=[ToolCallPart(
                            tool_name=p['tool_name'], args=p['arguments'], tool_call_id=p['tool_call_id'])]))
                    elif p['type'] == 'tool_response' and role == 'tool':
                        message_history.append(ModelRequest(parts=[ToolReturnPart(
                            tool_name=p.get('tool_name', 'search'), content=p['content'], tool_call_id=p['tool_call_id'])]))
                    elif p['type'] == 'text' and role == 'assistant':
                        message_history.append(ModelResponse(parts=[TextPart(content=p['content'])]))

        response = await self.agent.arun(user_input, message_history=message_history)

        pydantic_ai_messages = response.all_messages()

        search_call_count = 0
        for msg in pydantic_ai_messages:
            if isinstance(msg, ModelResponse):
                for p in msg.parts:
                    if isinstance(p, ToolCallPart) and p.tool_name == 'search':
                        search_call_count += 1

        converted_messages = self._convert_messages(pydantic_ai_messages)
        response_metadata = {'search_calls': search_call_count,
                             'stop_reason': getattr(response, 'stop_reason', None)}
        self.evaluation_count += 1

        return SamplerResponse(
            response_text=response,
            actual_queried_message_list=converted_messages,
            response_metadata=response_metadata
        )

    def _pack_message(self, content, role="user"):
        return {"content": content, "role": role}
