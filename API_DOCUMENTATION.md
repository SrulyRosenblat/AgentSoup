# AgentSoup API Documentation

## Overview

AgentSoup is a lightweight Python library that transforms regular functions into structured, prompt-driven LLM tools. This documentation covers all public APIs, functions, and components with detailed usage examples.

## Table of Contents

- [Core Decorator](#core-decorator)
- [Response Types](#response-types)
- [Message System](#message-system)
- [Message Parts](#message-parts)
- [Message Types](#message-types)
- [Utility Functions](#utility-functions)
- [Complete Usage Examples](#complete-usage-examples)

---

## Core Decorator

### `@llm(model="gpt-4o-mini", **llm_kwargs)`

The main decorator that transforms a regular function into an LLM-powered function.

**Parameters:**
- `model` (str, optional): The LLM model to use. Defaults to `"gpt-4o-mini"`. Supports all models available through litellm (OpenAI, Gemini, Claude, Mistral, etc.)
- `**llm_kwargs`: Additional keyword arguments passed to the underlying litellm completion call

**Returns:**
- Decorated function that returns structured output based on the function's return type annotation

**Usage Examples:**

```python
from pydantic import BaseModel
from agentsoup import llm

# Basic string return
@llm(model="gpt-4o-mini")
def simple_completion(prompt: str) -> str:
    """Simple text completion."""
    return f"Complete this: {prompt}"

result = simple_completion("The weather today is")
print(result)  # Returns string response from LLM

# Structured output with Pydantic
class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

@llm(model="gpt-4o-mini")
def summarize_text(text: str) -> Summary:
    """Summarizes text with structured output."""
    return f"Summarize the following text: {text}"

summary = summarize_text("Long article text here...")
print(summary.title)  # Access structured fields
print(summary.key_points)
print(summary.sentiment)

# With additional LLM parameters
@llm(model="gpt-4o-mini", temperature=0.7, max_tokens=500)
def creative_writing(prompt: str) -> str:
    """Creative writing with custom parameters."""
    return f"Write a creative story about: {prompt}"
```

---

## Response Types

### `CompleteResponse[T]`

A container that provides both the parsed response and the raw completion object when you need access to metadata like token usage.

**Type Parameters:**
- `T`: The type of the parsed response

**Attributes:**
- `parsed_response: T`: The parsed Pydantic model or string response
- `completion: object`: The raw ChatCompletion object from litellm

**Usage Example:**

```python
from pydantic import BaseModel
from agentsoup import llm, CompleteResponse

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]

@llm(model="gpt-4o-mini")
def analyze_text(text: str) -> CompleteResponse[Analysis]:
    """Analyze text sentiment with full response metadata."""
    return f"Analyze the sentiment of: {text}"

response = analyze_text("I love this product!")

# Access parsed response
print(response.parsed_response.sentiment)  # "positive"
print(response.parsed_response.confidence)  # 0.95

# Access raw completion metadata
print(response.completion.usage.total_tokens)  # Token count
print(response.completion.model)  # Model used
print(response.completion.choices[0].finish_reason)  # Completion reason
```

---

## Message System

The message system provides structured ways to compose complex prompts with multiple content types.

### `MessagePart`

Base class for all message content parts.

**Methods:**
- `to_openai_format()`: Converts the message part to OpenAI API format
- `__str__()`, `__repr__()`: String representations

**Note:** This is a base class - use the specific subclasses below.

---

## Message Parts

### `Text(text: str)`

Represents text content in a message.

**Parameters:**
- `text` (str): The text content

**Usage Example:**

```python
from agentsoup import Text

text_part = Text("Hello, world!")
print(text_part.to_openai_format())
# Output: {'type': 'text', 'text': 'Hello, world!'}
```

### `RemoteImage(url: str)`

Represents an image from a URL.

**Parameters:**
- `url` (str): The URL of the image

**Usage Example:**

```python
from agentsoup import RemoteImage

image_part = RemoteImage("https://example.com/image.jpg")
print(image_part.to_openai_format())
# Output: {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.jpg'}}
```

### `LocalImage(image_path: str)`

Represents a local image file that gets base64 encoded.

**Parameters:**
- `image_path` (str): Path to the local image file

**Usage Example:**

```python
from agentsoup import LocalImage

image_part = LocalImage("./my_image.jpg")
# Automatically encodes the image as base64
```

### `LocalPDF(file_path: str)`

Represents a local PDF file that gets base64 encoded.

**Parameters:**
- `file_path` (str): Path to the local PDF file

**Usage Example:**

```python
from agentsoup import LocalPDF

pdf_part = LocalPDF("./document.pdf")
# Automatically encodes the PDF as base64
```

### `RemotePDF(url: str)`

Represents a PDF from a URL.

**Parameters:**
- `url` (str): The URL of the PDF file

**Usage Example:**

```python
from agentsoup import RemotePDF

pdf_part = RemotePDF("https://example.com/document.pdf")
```

---

## Message Types

### `Message(role: str, content: list[MessagePart])`

Base class for all message types.

**Parameters:**
- `role` (str): The role of the message sender ("user", "system", "assistant")
- `content` (list[MessagePart]): List of message parts that make up the content

**Methods:**
- `to_openai_format()`: Converts to OpenAI API message format

### `UserMessage(content: list[MessagePart])`

Represents a user message with mixed content types.

**Parameters:**
- `content` (list[MessagePart]): List of message parts

**Usage Example:**

```python
from agentsoup import UserMessage, Text, LocalImage, llm
from pydantic import BaseModel

class ImageDescription(BaseModel):
    description: str
    objects: list[str]

@llm(model="gpt-4o-mini")
def describe_image(image_path: str, question: str) -> ImageDescription:
    """Describe an image with a specific question."""
    return [
        UserMessage(content=[
            Text(f"Question: {question}"),
            LocalImage(image_path=image_path),
            Text("Please provide a detailed description.")
        ])
    ]

result = describe_image("./photo.jpg", "What objects are in this image?")
```

### `UserMessageText(content: str)`

Convenience class for simple text-only user messages.

**Parameters:**
- `content` (str): The text content

**Usage Example:**

```python
from agentsoup import UserMessageText

message = UserMessageText("Hello, how are you?")
# Equivalent to: UserMessage([Text("Hello, how are you?")])
```

### `SystemMessage(content: list[MessagePart])`

Represents a system message that sets context or instructions.

**Parameters:**
- `content` (list[MessagePart]): List of message parts

**Usage Example:**

```python
from agentsoup import SystemMessage, Text, llm

@llm(model="gpt-4o-mini")
def helpful_assistant(user_query: str) -> str:
    """Assistant with system context."""
    return [
        SystemMessage([Text("You are a helpful assistant specialized in Python programming.")]),
        UserMessageText(user_query)
    ]
```

### `SystemMessageText(content: str)`

Convenience class for simple text-only system messages.

**Parameters:**
- `content` (str): The system message text

### `AssistantMessage(content: list[MessagePart])`

Represents an assistant message (useful for conversation history).

**Parameters:**
- `content` (list[MessagePart]): List of message parts

### `AssistantMessageText(content: str)`

Convenience class for simple text-only assistant messages.

**Parameters:**
- `content` (str): The assistant message text

### `AgentSystemPrompt(prompt: str, tools: list[Function] = [])`

Specialized system message for agent-style prompts with tool integration.

**Parameters:**
- `prompt` (str): The system prompt template (must contain `{tools}` placeholder if tools are provided)
- `tools` (list[Function], optional): List of functions to include as available tools

**Usage Example:**

```python
from agentsoup import AgentSystemPrompt, llm

def calculate(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}"

@llm(model="gpt-4o-mini")
def agent_response(user_input: str) -> str:
    """Agent with available tools."""
    return [
        AgentSystemPrompt(
            "You are a helpful agent. Available tools: {tools}",
            tools=[calculate, get_weather]
        ),
        UserMessageText(user_input)
    ]
```

---

## Utility Functions

### `encode_base64(image_path: str) -> str`

Encodes a local file to base64 string.

**Parameters:**
- `image_path` (str): Path to the file to encode

**Returns:**
- `str`: Base64 encoded string of the file

**Usage Example:**

```python
from agentsoup import encode_base64

encoded = encode_base64("./my_image.jpg")
print(f"data:image/jpeg;base64,{encoded}")
```

---

## Complete Usage Examples

### Example 1: Multi-Modal Image Analysis

```python
from pydantic import BaseModel
from agentsoup import llm, UserMessage, Text, LocalImage, CompleteResponse

class ImageAnalysis(BaseModel):
    description: str
    objects: list[str]
    colors: list[str]
    mood: str

@llm(model="gpt-4o-mini")
def analyze_image(image_path: str) -> CompleteResponse[ImageAnalysis]:
    """Comprehensive image analysis."""
    return [
        UserMessage(content=[
            Text("Please analyze this image in detail:"),
            LocalImage(image_path=image_path),
            Text("Provide description, objects, dominant colors, and overall mood.")
        ])
    ]

# Usage
response = analyze_image("./vacation_photo.jpg")
print(f"Description: {response.parsed_response.description}")
print(f"Objects found: {response.parsed_response.objects}")
print(f"Colors: {response.parsed_response.colors}")
print(f"Mood: {response.parsed_response.mood}")
print(f"Tokens used: {response.completion.usage.total_tokens}")
```

### Example 2: Document Processing with PDF

```python
from pydantic import BaseModel
from agentsoup import llm, UserMessage, Text, LocalPDF

class DocumentSummary(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    page_count: int

@llm(model="gpt-4o-mini")
def summarize_pdf(pdf_path: str) -> DocumentSummary:
    """Summarize a PDF document."""
    return [
        UserMessage(content=[
            Text("Please summarize this PDF document:"),
            LocalPDF(file_path=pdf_path),
            Text("Include title, summary, key points, and estimated page count.")
        ])
    ]

# Usage
summary = summarize_pdf("./report.pdf")
print(f"Title: {summary.title}")
print(f"Summary: {summary.summary}")
for i, point in enumerate(summary.key_points, 1):
    print(f"{i}. {point}")
```

### Example 3: Conversation with Context

```python
from agentsoup import llm, SystemMessage, UserMessage, AssistantMessage, Text

@llm(model="gpt-4o-mini")
def continue_conversation(conversation_history: list, new_message: str) -> str:
    """Continue a conversation with full context."""
    messages = [
        SystemMessage([Text("You are a helpful assistant with memory of the conversation.")])
    ]
    
    # Add conversation history
    for msg in conversation_history:
        if msg['role'] == 'user':
            messages.append(UserMessage([Text(msg['content'])]))
        elif msg['role'] == 'assistant':
            messages.append(AssistantMessage([Text(msg['content'])]))
    
    # Add new message
    messages.append(UserMessage([Text(new_message)]))
    
    return messages

# Usage
history = [
    {'role': 'user', 'content': 'My name is Alice'},
    {'role': 'assistant', 'content': 'Nice to meet you, Alice!'},
    {'role': 'user', 'content': 'I like programming'},
    {'role': 'assistant', 'content': 'Programming is great! What languages do you enjoy?'}
]

response = continue_conversation(history, "What was my name again?")
print(response)  # Should remember the name is Alice
```

### Example 4: Structured Data Extraction

```python
from pydantic import BaseModel
from typing import Optional
from agentsoup import llm

class Contact(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None

class ContactList(BaseModel):
    contacts: list[Contact]

@llm(model="gpt-4o-mini")
def extract_contacts(text: str) -> ContactList:
    """Extract contact information from text."""
    return f"""
    Extract all contact information from the following text.
    For each person, find their name, email, phone, and company if available:
    
    {text}
    """

# Usage
text = """
John Smith works at TechCorp and can be reached at john@techcorp.com or 555-0123.
Sarah Johnson from DataSoft is available at sarah.j@datasoft.io.
Mike Brown's phone is (555) 987-6543.
"""

contacts = extract_contacts(text)
for contact in contacts.contacts:
    print(f"Name: {contact.name}")
    if contact.email:
        print(f"  Email: {contact.email}")
    if contact.phone:
        print(f"  Phone: {contact.phone}")
    if contact.company:
        print(f"  Company: {contact.company}")
    print()
```

### Example 5: Custom Model Parameters

```python
from pydantic import BaseModel
from agentsoup import llm

class CreativeStory(BaseModel):
    title: str
    story: str
    genre: str
    word_count: int

@llm(
    model="gpt-4o-mini",
    temperature=0.9,  # High creativity
    max_tokens=1000,
    top_p=0.95
)
def write_story(prompt: str, genre: str) -> CreativeStory:
    """Write a creative story with custom parameters."""
    return f"""
    Write a {genre} story based on this prompt: {prompt}
    Make it creative and engaging, around 200-300 words.
    """

# Usage
story = write_story("A robot discovers emotions", "science fiction")
print(f"Title: {story.title}")
print(f"Genre: {story.genre}")
print(f"Word Count: {story.word_count}")
print(f"\nStory:\n{story.story}")
```

---

## Error Handling

### Common Errors and Solutions

1. **Unsupported Return Type**
   ```python
   # Wrong: Unsupported return type
   @llm()
   def bad_function() -> dict:  # dict is not supported
       return "some prompt"
   
   # Correct: Use Pydantic models or str
   class MyModel(BaseModel):
       data: dict
   
   @llm()
   def good_function() -> MyModel:
       return "some prompt"
   ```

2. **Invalid Function Return**
   ```python
   # Wrong: Function must return str, Message, or list of Messages
   @llm()
   def bad_function() -> str:
       return 123  # Invalid return type
   
   # Correct: Return appropriate types
   @llm()
   def good_function() -> str:
       return "This is a valid prompt string"
   ```

3. **Missing Tools Placeholder**
   ```python
   # Wrong: Tools provided but no {tools} placeholder
   AgentSystemPrompt("You are helpful", tools=[some_function])  # Error!
   
   # Correct: Include {tools} placeholder
   AgentSystemPrompt("You are helpful. Tools: {tools}", tools=[some_function])
   ```

---

## Best Practices

1. **Use Type Hints**: Always provide return type annotations for proper parsing
2. **Structured Output**: Use Pydantic models for complex, structured responses
3. **Message Composition**: Use the message system for complex, multi-modal prompts
4. **Error Handling**: Wrap LLM calls in try-catch blocks for production use
5. **Token Management**: Use `CompleteResponse` when you need to monitor token usage
6. **Model Selection**: Choose appropriate models for your use case (speed vs capability)

---

## Integration with LiteLLM

AgentSoup is built on top of [LiteLLM](https://github.com/BerriAI/litellm), which means you can use any model supported by LiteLLM:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`
- **Google**: `gemini-2.5-flash`, `gemini-1.5-pro`
- **Mistral**: `mistral-large-latest`, `mistral-small-latest`
- **And many more...**

Simply change the `model` parameter in the `@llm` decorator to use different models.