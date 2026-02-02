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

        response_metadata = {'search_calls': search_call_count}

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

    def _pack_message(self, content, role="user"):
        return {"content": content, "role": role}
