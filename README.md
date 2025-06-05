

# 🥣 AgentSoup

**Mix prompts, models, and logic — cook up LLM-powered functions with ease.**

AgentSoup is a lightweight Python library for turning regular functions into structured, prompt-driven LLM tools. It’s flexible enough for creative or structured outputs, while keeping your code readable and maintainable.

## Installation

```
pip install agentsoup
```

> ✨ Built on top of [`litellm`](https://github.com/BerriAI/litellm), `pydantic`, and composable message objects.

---

## 🧠 Why AgentSoup?

* 🥄 **Simple to start** – just decorate a function and return a prompt.
* 🍲 **Structured responses** – get fully typed outputs using `pydantic`.
* 🔬 **Complete response mode** – get raw completions *and* parsed data.
* 🧩 **Composable messages** – use structured message types for flexibility.
* 🔄 **Model-flexible** – works with OpenAI, Gemini, Mistral, Claude, and more via `litellm`.

---

## 📘 Example 1: Sorting Books

```python
from pydantic import BaseModel
from agentsoup import llm

class Book(BaseModel):
    title: str

class BooksList(BaseModel):
    books: list[Book]

@llm(model="gpt-4o-mini")
def sort_books_by_popularity(book_list: list[str]) -> BooksList:
    """
    Sorts books by popularity.
    """
    return 'sort the following books by popularity {book_list}'

# Call it like a regular function:
result = sort_books_by_popularity([
    'The Great Gatsby', 'To Kill a Mockingbird', '1984',
    'Pride and Prejudice', 'The Hobbit'
])
print(result)
```

---

## 🖼️ Example 2: Image Captioning with Full Response

```python
from pydantic import BaseModel
from agentsoup import llm, UserMessage, Text, LocalImage, CompleteResponse

class ImageDescription(BaseModel):
    caption: str
    long_description: str

@llm(model="gpt-4o-mini")
def describe_image(path: str) -> CompleteResponse[ImageDescription]:
    """
    Describes an image.
    """
    return [
        UserMessage(
            content=[
                Text('describe the following image:'),
                LocalImage(image_path=path)
            ]
        )
    ]

response = describe_image('./penguin.jpeg')
print("Caption:", response.parsed_response.caption)
print("Description:", response.parsed_response.long_description)
print("Tokens used:", response.completion.usage.total_tokens)

```

---

## 🧪 Example 3: Regex Generator with Gemini

```python
import re
from pydantic import BaseModel
from agentsoup import llm

class Regex(BaseModel):
    regex_str: str

@llm(model="gemini-2.5-flash")
def build_regex(text: str) -> Regex:
    """
    Builds a regex from a text description.
    """
    return f'give regex that does the following: {text}'

text_blob = """
poe 435-435-4354
holmes (435) 435-4354
bob 435.435.4354
"""

regex = build_regex('extracts all phone numbers from a text in any format').regex_str
print("Generated Regex:", regex)
print("Matches:", re.findall(regex, text_blob))
```

---

## 📦 Message Types (Optional)

AgentSoup supports structured message composition via:

* `UserMessage`
* `SystemMessage`
* `Text`
* `LocalImage`
* and more..

This allows full control over the message content while keeping function logic clean and focused.

---

## 🧠 Coming Soon

* 🕵️ Agent mode for goal-driven agents
* 🧰 Tool & plugin support
* 🔁 Streaming + iterative refinement modes
* 🧠 Full Documentation

---

## 📜 License

MIT



