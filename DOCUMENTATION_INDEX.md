# AgentSoup Documentation Index

Welcome to the comprehensive documentation for AgentSoup! This index provides easy navigation to all documentation resources.

## 🚀 Getting Started

### New to AgentSoup?
Start here for a quick introduction and basic usage:

- **[Quick Start Guide](QUICK_START_GUIDE.md)** - 5-minute introduction with basic examples
- **[README](README.md)** - Project overview and installation instructions

### Want to dive deeper?
Explore the complete API reference and advanced patterns:

- **[API Documentation](API_DOCUMENTATION.md)** - Complete reference for all classes and functions
- **[Advanced Usage Guide](ADVANCED_USAGE.md)** - Complex patterns and production-ready implementations

---

## 📚 Documentation Structure

### 1. [Quick Start Guide](QUICK_START_GUIDE.md)
**Perfect for beginners** - Get up and running in 5 minutes
- Installation instructions
- Basic examples for common use cases
- Environment setup
- Troubleshooting common issues

**Key Topics:**
- Text completion
- Structured output with Pydantic
- Image analysis
- Different model usage
- Common patterns (data extraction, classification, content generation)

### 2. [API Documentation](API_DOCUMENTATION.md)
**Complete reference** - Detailed documentation of all public APIs
- Core decorator (`@llm`)
- Response types (`CompleteResponse`)
- Message system (all message types and parts)
- Utility functions
- Complete usage examples

**Key Components:**
- `@llm` decorator with all parameters
- `CompleteResponse[T]` for metadata access
- Message parts: `Text`, `LocalImage`, `RemoteImage`, `LocalPDF`, `RemotePDF`
- Message types: `UserMessage`, `SystemMessage`, `AssistantMessage`, `AgentSystemPrompt`
- Error handling and best practices

### 3. [Advanced Usage Guide](ADVANCED_USAGE.md)
**For experienced users** - Complex patterns and production deployment
- Multi-modal applications
- Advanced message composition
- Error handling and resilience
- Performance optimization
- Production patterns

**Key Topics:**
- Complex document analysis with PDFs and images
- Interactive conversation systems
- Dynamic message building
- Robust error handling with retries and fallbacks
- Batch processing and caching
- Monitoring and logging
- Configuration management

---

## 🔍 Quick Reference

### Core Concepts

| Concept | Description | Documentation |
|---------|-------------|---------------|
| `@llm` decorator | Main decorator to create LLM-powered functions | [API Docs](API_DOCUMENTATION.md#core-decorator) |
| `CompleteResponse` | Container for parsed response + raw completion | [API Docs](API_DOCUMENTATION.md#response-types) |
| Message System | Structured way to compose complex prompts | [API Docs](API_DOCUMENTATION.md#message-system) |
| Pydantic Integration | Structured output using Pydantic models | [Quick Start](QUICK_START_GUIDE.md#structured-output-with-pydantic) |

### Message Parts

| Class | Purpose | Example Usage |
|-------|---------|---------------|
| `Text` | Text content | `Text("Hello world")` |
| `LocalImage` | Local image file | `LocalImage("./photo.jpg")` |
| `RemoteImage` | Image from URL | `RemoteImage("https://example.com/img.jpg")` |
| `LocalPDF` | Local PDF file | `LocalPDF("./document.pdf")` |
| `RemotePDF` | PDF from URL | `RemotePDF("https://example.com/doc.pdf")` |

### Message Types

| Class | Purpose | Example Usage |
|-------|---------|---------------|
| `UserMessage` | User input with mixed content | `UserMessage([Text("Analyze"), LocalImage("img.jpg")])` |
| `SystemMessage` | System instructions | `SystemMessage([Text("You are a helpful assistant")])` |
| `AssistantMessage` | Assistant response (for history) | `AssistantMessage([Text("I can help with that")])` |
| `AgentSystemPrompt` | Agent prompt with tools | `AgentSystemPrompt("You have tools: {tools}", tools=[func])` |

---

## 🎯 Use Case Navigation

### By Use Case

| What do you want to do? | Start Here | Advanced Patterns |
|-------------------------|------------|-------------------|
| **Basic text completion** | [Quick Start](QUICK_START_GUIDE.md#basic-text-completion) | [API Docs](API_DOCUMENTATION.md#core-decorator) |
| **Structured data extraction** | [Quick Start](QUICK_START_GUIDE.md#pattern-1-data-extraction) | [Advanced](ADVANCED_USAGE.md#complex-multi-modal-applications) |
| **Image analysis** | [Quick Start](QUICK_START_GUIDE.md#image-analysis) | [Advanced](ADVANCED_USAGE.md#document-analysis-with-images-and-text) |
| **Document processing** | [API Docs](API_DOCUMENTATION.md#example-2-document-processing-with-pdf) | [Advanced](ADVANCED_USAGE.md#document-analysis-with-images-and-text) |
| **Conversation systems** | [API Docs](API_DOCUMENTATION.md#example-3-conversation-with-context) | [Advanced](ADVANCED_USAGE.md#interactive-conversation-system) |
| **Error handling** | [Quick Start](QUICK_START_GUIDE.md#troubleshooting) | [Advanced](ADVANCED_USAGE.md#error-handling-and-resilience) |
| **Production deployment** | [API Docs](API_DOCUMENTATION.md#best-practices) | [Advanced](ADVANCED_USAGE.md#production-patterns) |
| **Performance optimization** | - | [Advanced](ADVANCED_USAGE.md#performance-optimization) |

### By Complexity Level

#### 🟢 Beginner
- [Quick Start Guide](QUICK_START_GUIDE.md) - Complete beginner tutorial
- [API Documentation - Basic Examples](API_DOCUMENTATION.md#core-decorator) - Simple usage patterns

#### 🟡 Intermediate  
- [API Documentation - Complete Examples](API_DOCUMENTATION.md#complete-usage-examples) - Multi-modal applications
- [Advanced Usage - Message Composition](ADVANCED_USAGE.md#advanced-message-composition) - Dynamic prompts

#### 🔴 Advanced
- [Advanced Usage - Error Handling](ADVANCED_USAGE.md#error-handling-and-resilience) - Production resilience
- [Advanced Usage - Performance](ADVANCED_USAGE.md#performance-optimization) - Scaling and optimization
- [Advanced Usage - Production](ADVANCED_USAGE.md#production-patterns) - Enterprise deployment

---

## 🛠️ Development Workflow

### 1. **Learning Phase**
1. Read [Quick Start Guide](QUICK_START_GUIDE.md)
2. Try basic examples
3. Explore [API Documentation](API_DOCUMENTATION.md) for your use case

### 2. **Development Phase**
1. Use [API Documentation](API_DOCUMENTATION.md) as reference
2. Implement error handling from [Advanced Guide](ADVANCED_USAGE.md#error-handling-and-resilience)
3. Add monitoring from [Production Patterns](ADVANCED_USAGE.md#monitoring-and-logging)

### 3. **Production Phase**
1. Implement [Configuration Management](ADVANCED_USAGE.md#configuration-management)
2. Add [Performance Optimization](ADVANCED_USAGE.md#performance-optimization)
3. Set up [Monitoring and Logging](ADVANCED_USAGE.md#monitoring-and-logging)

---

## 📖 Code Examples by Category

### Basic Usage
```python
# Simple completion
@llm(model="gpt-4o-mini")
def complete_text(prompt: str) -> str:
    return f"Complete: {prompt}"

# Structured output
class Analysis(BaseModel):
    sentiment: str
    confidence: float

@llm(model="gpt-4o-mini")
def analyze(text: str) -> Analysis:
    return f"Analyze: {text}"
```

### Multi-Modal
```python
# Image + text analysis
@llm(model="gpt-4o-mini")
def analyze_image(image_path: str, question: str) -> str:
    return [
        UserMessage([
            Text(f"Question: {question}"),
            LocalImage(image_path=image_path)
        ])
    ]
```

### Production Ready
```python
# With error handling and monitoring
@monitored_llm(model="gpt-4o-mini")
@with_retry(RetryConfig(max_retries=3))
def production_analyze(data: str) -> CompleteResponse[Analysis]:
    return f"Analyze: {data}"
```

---

## 🔗 External Resources

### AgentSoup Project
- **GitHub Repository**: [https://github.com/SrulyRosenblat/AgentSoup](https://github.com/SrulyRosenblat/AgentSoup)
- **PyPI Package**: [https://pypi.org/project/agentsoup/](https://pypi.org/project/agentsoup/)

### Dependencies
- **LiteLLM Documentation**: [https://docs.litellm.ai/](https://docs.litellm.ai/)
- **Pydantic Documentation**: [https://docs.pydantic.dev/](https://docs.pydantic.dev/)

### Model Providers
- **OpenAI**: [https://platform.openai.com/docs](https://platform.openai.com/docs)
- **Anthropic**: [https://docs.anthropic.com/](https://docs.anthropic.com/)
- **Google AI**: [https://ai.google.dev/docs](https://ai.google.dev/docs)
- **Mistral AI**: [https://docs.mistral.ai/](https://docs.mistral.ai/)

---

## 🆘 Getting Help

### Documentation Issues
1. **Can't find what you're looking for?** Check the [API Documentation](API_DOCUMENTATION.md) search functionality
2. **Need a specific example?** Browse the [Advanced Usage Guide](ADVANCED_USAGE.md) examples
3. **Getting started issues?** Follow the [Quick Start Guide](QUICK_START_GUIDE.md) step by step

### Common Problems
| Problem | Solution | Documentation |
|---------|----------|---------------|
| API key not working | Check environment variables | [Quick Start - Environment Setup](QUICK_START_GUIDE.md#environment-setup) |
| Model not found | Verify model name with LiteLLM | [Quick Start - Troubleshooting](QUICK_START_GUIDE.md#troubleshooting) |
| JSON parsing errors | Use Optional fields in Pydantic | [API Docs - Error Handling](API_DOCUMENTATION.md#error-handling) |
| Performance issues | Implement caching/batching | [Advanced - Performance](ADVANCED_USAGE.md#performance-optimization) |

### Support Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community support
- **Documentation**: Start with this index and navigate to specific guides

---

## 📝 Contributing to Documentation

Found an issue or want to improve the documentation? 

1. **Typos/Small fixes**: Submit a pull request
2. **Missing examples**: Add them to the appropriate guide
3. **New patterns**: Consider adding to [Advanced Usage Guide](ADVANCED_USAGE.md)
4. **Beginner confusion**: Improve [Quick Start Guide](QUICK_START_GUIDE.md)

---

## 🏷️ Version Information

This documentation covers AgentSoup v0.1.0 and is compatible with:
- Python 3.8+
- LiteLLM (latest)
- Pydantic v2.x

Last updated: 2024

---

**Happy coding with AgentSoup! 🥣✨**