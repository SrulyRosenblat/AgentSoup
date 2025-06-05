from typing import get_type_hints
from litellm import completion
import os
import base64
from functools import wraps
from pyclbr import Function
import inspect
import re
from typing import Generic, NamedTuple, TypeVar, get_origin, get_args
import json

T = TypeVar("T")

class CompleteResponse(NamedTuple, Generic[T]):
    """Container returned when you also want the raw ChatCompletion."""
    parsed_response: T                                  # the parsed pydantic model
    completion: object 

def _get_header_and_docstring(func):
    source = inspect.getsource(func)
    lines = source.splitlines()

    # Skip decorators
    lines = [line for line in lines if not line.strip().startswith('@')]

    # Find the header (starts with def)
    header_index = next(i for i, line in enumerate(lines) if line.strip().startswith('def'))
    header_line = lines[header_index]

    # Remove return type annotation (e.g., '-> str')
    header_line = re.sub(r'->\s*[^:]+', '', header_line)

    result = [header_line]

    # If there's a docstring right after
    if len(lines) > header_index + 1 and lines[header_index + 1].strip().startswith(('"""', "'''")):
        result.append(lines[header_index + 1])
        for line in lines[header_index + 2:]:
            result.append(line)
            if line.strip().endswith(('"""', "'''")):
                break

    return '\n'.join(result)

# Function to encode the image
def encode_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



class MessagePart:
    def __init__(self, **kwargs):
        self.content = kwargs
    def to_openai_format(self):
        return self.content
    def __str__(self):
        return str(self.content)
    def __repr__(self):
        return str(self.content)

class Text(MessagePart):
    def __init__(self, text: str):
        
        super().__init__(type="text", text=text)

class RemoteImage(MessagePart):
    def __init__(self, url: str):
        super().__init__(type="image_url", image_url={"url": url})

class LocalImage(MessagePart):
    def __init__(self, image_path: str):
        super().__init__(type="image_url", image_url={"url": f"data:image/jpeg;base64,{encode_base64(image_path)}"})
        
class LocalPDF(MessagePart):
    def __init__(self, file_path: str):
            encoded_pdf = encode_base64(file_path)
            super().__init__(type="file", file={"filename": file_path, "file_data": f"data:application/pdf;base64,{encoded_pdf}"})

class RemotePDF(MessagePart):
    def __init__(self, url: str):
        super().__init__(type="file", file={"file_id": url})



class Message:
    def __init__(self, role: str, content: list[MessagePart]):
        self.role = role
        self.content = content
        
    def to_openai_format(self):
        message_content = []
   
        for part in self.content:
            message_content.append(part.to_openai_format())
        return {"role": self.role, "content": message_content}
        
    def __str__(self):
        return str(self.to_openai_format())
    def __repr__(self):
        return str(self.to_openai_format())


class UserMessage(Message):
    def __init__(self, content: list[MessagePart]):
        super().__init__("user", content)

class UserMessageText(UserMessage):
    def __init__(self, content: str):
        super().__init__( [Text(content)])

class SystemMessage(Message):
    def __init__(self, content: list[MessagePart]):
        super().__init__("system", content)

class SystemMessageText(SystemMessage):
    def __init__(self, content: str):
        super().__init__([Text(content)])

class AssistantMessage(Message):
    def __init__(self, content: list[MessagePart]):
        super().__init__("assistant", content)

class AssistantMessageText(AssistantMessage):
    def __init__(self, content: str):
        super().__init__([Text(content)])

class AgentSystemPrompt(SystemMessage):
    def __init__(self, prompt: str, tools: list[Function] = []):
        if len(tools) > 0 and "{tools}" not in prompt:
            raise ValueError("Prompt must contain {tools} if tools are provided")

        super().__init__([Text(prompt.format(tools=self._format_tools(tools)))])
    def _format_tools(self, tools: list[Function]):
        text = "```\n"
        for tool in tools:
            text += _get_header_and_docstring(tool)
            text += "\n---\n"
        text += "```"
        return text

    



def _generate_response(messages: list[Message], return_type: type, model: str, **llm_kwargs):
    if isinstance(messages, list) and all(isinstance(message, Message) for message in messages):
        messages = [message.to_openai_format() for message in messages]
    complete_response = get_origin(return_type) == CompleteResponse
    return_type = get_args(return_type)[0] if complete_response else return_type
    
    if return_type is not None and return_type != str:
        response = completion(
            model=model,
            messages=messages,
            response_format=return_type,
            **llm_kwargs
        )
        parsed_response = return_type(**json.loads(response.choices[0].message.content))
        if complete_response:
            return CompleteResponse(parsed_response=parsed_response, completion=response)
        else:
            return parsed_response
    else:
        response = completion(
            model=model,
            messages=messages,
            **llm_kwargs
        )
        parsed_response = response.choices[0].message.content
        if complete_response:
            return CompleteResponse(parsed_response=parsed_response, completion=response)
        else:
            return parsed_response

def llm(model="gpt-4.1", **llm_kwargs):
    "Decorator to wrap a function with a LLM call and return the result in the correct format" 
    def _llm(f):
        return_type = get_type_hints(f).get('return', None)
        @wraps(f)
        def _wrap(*args, **kwargs): 
            function_result = f(*args, **kwargs)
            if type(function_result) == str:
                messages = [UserMessageText(function_result)]
            elif isinstance(function_result, Message):
                messages = [function_result]
            elif isinstance(function_result, list):
                messages = function_result
            else:
                raise ValueError(f"Unsupported return type: {type(function_result)}")
            return _generate_response(
                messages,
                return_type,
                model,
                **llm_kwargs
            )
         
        return _wrap  
    
    return _llm  


