# AgentSoup Quick Start Guide

## Installation

```bash
pip install agentsoup
```

## 5-Minute Quick Start

### 1. Basic Text Completion

```python
from agentsoup import llm

@llm(model="gpt-4o-mini")
def complete_text(prompt: str) -> str:
    """Complete a text prompt."""
    return f"Complete this text: {prompt}"

# Usage
result = complete_text("The future of AI is")
print(result)
```

### 2. Structured Output with Pydantic

```python
from pydantic import BaseModel
from agentsoup import llm

class TaskAnalysis(BaseModel):
    priority: str  # "high", "medium", "low"
    estimated_hours: int
    category: str

@llm(model="gpt-4o-mini")
def analyze_task(task_description: str) -> TaskAnalysis:
    """Analyze a task and return structured information."""
    return f"Analyze this task: {task_description}"

# Usage
task = analyze_task("Implement user authentication system with OAuth2")
print(f"Priority: {task.priority}")
print(f"Estimated hours: {task.estimated_hours}")
print(f"Category: {task.category}")
```

### 3. Image Analysis

```python
from pydantic import BaseModel
from agentsoup import llm, UserMessage, Text, LocalImage

class ImageDescription(BaseModel):
    main_subject: str
    setting: str
    colors: list[str]

@llm(model="gpt-4o-mini")
def describe_image(image_path: str) -> ImageDescription:
    """Describe an image."""
    return [
        UserMessage(content=[
            Text("Describe this image focusing on the main subject, setting, and dominant colors:"),
            LocalImage(image_path=image_path)
        ])
    ]

# Usage
description = describe_image("./my_photo.jpg")
print(f"Main subject: {description.main_subject}")
print(f"Setting: {description.setting}")
print(f"Colors: {description.colors}")
```

### 4. Getting Complete Response (with metadata)

```python
from pydantic import BaseModel
from agentsoup import llm, CompleteResponse

class Summary(BaseModel):
    title: str
    key_points: list[str]

@llm(model="gpt-4o-mini")
def summarize_with_metadata(text: str) -> CompleteResponse[Summary]:
    """Summarize text and get response metadata."""
    return f"Summarize this text: {text}"

# Usage
response = summarize_with_metadata("Long article text here...")

# Access structured data
print(f"Title: {response.parsed_response.title}")
print(f"Key points: {response.parsed_response.key_points}")

# Access metadata
print(f"Tokens used: {response.completion.usage.total_tokens}")
print(f"Model: {response.completion.model}")
```

### 5. Different Models

```python
from agentsoup import llm

# OpenAI GPT-4
@llm(model="gpt-4o")
def gpt4_completion(prompt: str) -> str:
    return prompt

# Google Gemini
@llm(model="gemini-2.5-flash")
def gemini_completion(prompt: str) -> str:
    return prompt

# Anthropic Claude
@llm(model="claude-3-5-sonnet-20241022")
def claude_completion(prompt: str) -> str:
    return prompt

# Usage
result1 = gpt4_completion("Explain quantum computing")
result2 = gemini_completion("Write a haiku about coding")
result3 = claude_completion("Analyze this data trend")
```

## Common Patterns

### Pattern 1: Data Extraction

```python
from pydantic import BaseModel
from typing import Optional
from agentsoup import llm

class Person(BaseModel):
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    location: Optional[str] = None

class People(BaseModel):
    people: list[Person]

@llm(model="gpt-4o-mini")
def extract_people(text: str) -> People:
    """Extract people information from text."""
    return f"Extract all people mentioned in this text with their details: {text}"

# Usage
text = "John Smith, 30, is a software engineer in San Francisco. Mary Johnson works as a teacher."
people = extract_people(text)
for person in people.people:
    print(f"{person.name} - {person.age} - {person.occupation} - {person.location}")
```

### Pattern 2: Classification

```python
from pydantic import BaseModel
from agentsoup import llm

class Classification(BaseModel):
    category: str
    confidence: float
    reasoning: str

@llm(model="gpt-4o-mini")
def classify_email(email_content: str) -> Classification:
    """Classify email as spam, important, or normal."""
    return f"""
    Classify this email as 'spam', 'important', or 'normal'.
    Provide confidence (0-1) and reasoning.
    
    Email: {email_content}
    """

# Usage
email = "Congratulations! You've won $1,000,000! Click here to claim..."
result = classify_email(email)
print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### Pattern 3: Content Generation

```python
from pydantic import BaseModel
from agentsoup import llm

class BlogPost(BaseModel):
    title: str
    content: str
    tags: list[str]
    word_count: int

@llm(model="gpt-4o-mini", temperature=0.7)
def generate_blog_post(topic: str, target_audience: str) -> BlogPost:
    """Generate a blog post for a specific audience."""
    return f"""
    Write a blog post about {topic} for {target_audience}.
    Make it engaging and informative, around 300-500 words.
    """

# Usage
post = generate_blog_post("machine learning", "beginners")
print(f"Title: {post.title}")
print(f"Content: {post.content}")
print(f"Tags: {post.tags}")
print(f"Word count: {post.word_count}")
```

## Environment Setup

### Setting API Keys

AgentSoup uses LiteLLM, so you can set API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google
export GOOGLE_API_KEY="your-google-key"

# Mistral
export MISTRAL_API_KEY="your-mistral-key"
```

### Using Configuration Files

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

Then load it in your Python script:

```python
from dotenv import load_dotenv
load_dotenv()

from agentsoup import llm
# Now you can use the models
```

## Next Steps

1. **Read the full [API Documentation](API_DOCUMENTATION.md)** for detailed information about all classes and functions
2. **Explore advanced patterns** like multi-modal inputs and conversation handling
3. **Experiment with different models** to find the best fit for your use case
4. **Check out the examples** in the main README for more complex scenarios

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: API key not found
   ```
   Solution: Set the appropriate environment variable for your model provider.

2. **Model Not Found**
   ```
   Error: Model 'invalid-model' not found
   ```
   Solution: Check the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for valid model names.

3. **Invalid Return Type**
   ```
   Error: Unsupported return type
   ```
   Solution: Use `str` or Pydantic `BaseModel` subclasses for return types.

4. **JSON Parsing Error**
   ```
   Error: Failed to parse JSON response
   ```
   Solution: Make your Pydantic models more flexible with `Optional` fields and default values.

### Getting Help

- Check the [API Documentation](API_DOCUMENTATION.md) for detailed usage
- Look at the examples in the main README
- Review the source code in `agentsoup/__init__.py`
- Create an issue on the GitHub repository for bugs or feature requests