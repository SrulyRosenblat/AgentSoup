# AgentSoup Advanced Usage Guide

## Table of Contents

- [Complex Multi-Modal Applications](#complex-multi-modal-applications)
- [Advanced Message Composition](#advanced-message-composition)
- [Error Handling and Resilience](#error-handling-and-resilience)
- [Performance Optimization](#performance-optimization)
- [Custom Model Configuration](#custom-model-configuration)
- [Production Patterns](#production-patterns)
- [Integration Patterns](#integration-patterns)

---

## Complex Multi-Modal Applications

### Document Analysis with Images and Text

```python
from pydantic import BaseModel
from typing import List, Optional
from agentsoup import llm, UserMessage, Text, LocalImage, LocalPDF, CompleteResponse

class DocumentAnalysis(BaseModel):
    document_type: str
    summary: str
    key_findings: List[str]
    images_described: List[str]
    confidence_score: float
    recommendations: List[str]

@llm(model="gpt-4o", temperature=0.1)
def analyze_mixed_document(pdf_path: str, image_paths: List[str], context: str) -> CompleteResponse[DocumentAnalysis]:
    """Analyze a document with both PDF and image components."""
    content = [
        Text(f"Context: {context}"),
        Text("Please analyze this document package:"),
        LocalPDF(file_path=pdf_path),
        Text("Related images:")
    ]
    
    for i, image_path in enumerate(image_paths):
        content.extend([
            Text(f"Image {i+1}:"),
            LocalImage(image_path=image_path)
        ])
    
    content.append(Text("Provide comprehensive analysis including document type, summary, key findings, image descriptions, confidence score, and recommendations."))
    
    return [UserMessage(content=content)]

# Usage
response = analyze_mixed_document(
    pdf_path="./financial_report.pdf",
    image_paths=["./chart1.png", "./chart2.png"],
    context="Q3 2024 financial performance review"
)

analysis = response.parsed_response
print(f"Document Type: {analysis.document_type}")
print(f"Summary: {analysis.summary}")
print(f"Confidence: {analysis.confidence_score}")
print(f"Token Usage: {response.completion.usage.total_tokens}")
```

### Interactive Conversation System

```python
from pydantic import BaseModel
from typing import List, Dict, Any
from agentsoup import llm, SystemMessage, UserMessage, AssistantMessage, Text, CompleteResponse

class ConversationResponse(BaseModel):
    response: str
    emotion: str
    topics: List[str]
    follow_up_questions: List[str]
    conversation_summary: str

class ConversationManager:
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.system_prompt = system_prompt
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
    
    @llm(model="gpt-4o-mini")
    def _generate_response(self, messages: List) -> CompleteResponse[ConversationResponse]:
        return messages
    
    def chat(self, user_input: str) -> ConversationResponse:
        """Continue the conversation with context."""
        messages = [SystemMessage([Text(self.system_prompt)])]
        
        # Add conversation history
        for msg in self.conversation_history:
            if msg['role'] == 'user':
                messages.append(UserMessage([Text(msg['content'])]))
            else:
                messages.append(AssistantMessage([Text(msg['content'])]))
        
        # Add new user message
        messages.append(UserMessage([Text(user_input)]))
        
        # Generate response
        response = self._generate_response(messages)
        
        # Update history
        self.conversation_history.extend([
            {'role': 'user', 'content': user_input},
            {'role': 'assistant', 'content': response.parsed_response.response}
        ])
        
        return response.parsed_response
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the entire conversation."""
        if not self.conversation_history:
            return "No conversation yet."
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])
        
        @llm(model=self.model)
        def summarize_conversation(conv: str) -> str:
            return f"Summarize this conversation: {conv}"
        
        return summarize_conversation(conversation_text)

# Usage
bot = ConversationManager(
    system_prompt="You are a helpful AI assistant that provides thoughtful responses and asks engaging follow-up questions.",
    model="gpt-4o-mini"
)

response1 = bot.chat("I'm thinking about starting a new hobby")
print(f"Response: {response1.response}")
print(f"Emotion: {response1.emotion}")
print(f"Follow-ups: {response1.follow_up_questions}")

response2 = bot.chat("I like creative activities")
print(f"Response: {response2.response}")

summary = bot.get_conversation_summary()
print(f"Conversation Summary: {summary}")
```

---

## Advanced Message Composition

### Dynamic Message Building

```python
from pydantic import BaseModel
from typing import List, Optional, Union
from agentsoup import llm, UserMessage, SystemMessage, Text, LocalImage, RemoteImage

class MessageBuilder:
    def __init__(self):
        self.messages = []
    
    def add_system_context(self, context: str) -> 'MessageBuilder':
        self.messages.append(SystemMessage([Text(context)]))
        return self
    
    def add_text(self, text: str) -> 'MessageBuilder':
        if not self.messages or not isinstance(self.messages[-1], UserMessage):
            self.messages.append(UserMessage([Text(text)]))
        else:
            self.messages[-1].content.append(Text(text))
        return self
    
    def add_image(self, image_path: str, is_local: bool = True) -> 'MessageBuilder':
        if not self.messages or not isinstance(self.messages[-1], UserMessage):
            self.messages.append(UserMessage([]))
        
        if is_local:
            self.messages[-1].content.append(LocalImage(image_path=image_path))
        else:
            self.messages[-1].content.append(RemoteImage(url=image_path))
        return self
    
    def build(self) -> List:
        return self.messages

class AnalysisResult(BaseModel):
    analysis: str
    confidence: float
    recommendations: List[str]

@llm(model="gpt-4o-mini")
def flexible_analysis(builder: MessageBuilder) -> AnalysisResult:
    """Perform analysis using dynamically built messages."""
    return builder.build()

# Usage
builder = (MessageBuilder()
    .add_system_context("You are an expert analyst")
    .add_text("Analyze the following data:")
    .add_image("./chart.png")
    .add_text("Consider market trends and provide recommendations"))

result = flexible_analysis(builder)
print(f"Analysis: {result.analysis}")
print(f"Confidence: {result.confidence}")
```

### Conditional Message Composition

```python
from pydantic import BaseModel
from typing import List, Optional
from agentsoup import llm, UserMessage, SystemMessage, Text, LocalImage

class ConditionalAnalysis(BaseModel):
    primary_analysis: str
    visual_analysis: Optional[str] = None
    comparative_analysis: Optional[str] = None
    recommendations: List[str]

def build_analysis_prompt(
    data: str,
    include_image: bool = False,
    image_path: Optional[str] = None,
    compare_with: Optional[str] = None,
    expertise_level: str = "general"
) -> List:
    """Build analysis prompt based on conditions."""
    
    # System message based on expertise level
    system_prompts = {
        "general": "You are a helpful analyst.",
        "technical": "You are a technical expert with deep domain knowledge.",
        "executive": "You are a senior executive advisor focused on strategic insights."
    }
    
    messages = [SystemMessage([Text(system_prompts[expertise_level])])]
    
    # Build user message content
    content = [Text(f"Analyze this data: {data}")]
    
    # Add image if requested
    if include_image and image_path:
        content.extend([
            Text("Visual data:"),
            LocalImage(image_path=image_path)
        ])
    
    # Add comparison if requested
    if compare_with:
        content.append(Text(f"Compare with: {compare_with}"))
    
    content.append(Text("Provide comprehensive analysis and recommendations."))
    messages.append(UserMessage(content=content))
    
    return messages

@llm(model="gpt-4o-mini")
def conditional_analysis(
    data: str,
    include_image: bool = False,
    image_path: Optional[str] = None,
    compare_with: Optional[str] = None,
    expertise_level: str = "general"
) -> ConditionalAnalysis:
    """Perform analysis with conditional message building."""
    return build_analysis_prompt(data, include_image, image_path, compare_with, expertise_level)

# Usage examples
result1 = conditional_analysis("Sales data Q3 2024", expertise_level="executive")
result2 = conditional_analysis(
    "Performance metrics", 
    include_image=True, 
    image_path="./metrics.png",
    compare_with="Previous quarter data",
    expertise_level="technical"
)
```

---

## Error Handling and Resilience

### Robust Error Handling

```python
from pydantic import BaseModel, ValidationError
from typing import Optional, Union
from agentsoup import llm, CompleteResponse
import logging
import time
import random

class SafeResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    retry_count: int = 0

class RetryConfig:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

def with_retry(config: RetryConfig = None):
    """Decorator to add retry logic to LLM functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = min(
                            config.base_delay * (2 ** attempt) + random.uniform(0, 1),
                            config.max_delay
                        )
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logging.error(f"All {config.max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

class RobustAnalyzer:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    
    @with_retry(RetryConfig(max_retries=3, base_delay=2.0))
    def safe_analyze(self, data: str) -> SafeResponse:
        """Analyze data with comprehensive error handling."""
        try:
            @llm(model=self.model, temperature=0.1)
            def analyze_data(text: str) -> dict:
                return f"Analyze this data and return structured JSON: {text}"
            
            result = analyze_data(data)
            
            return SafeResponse(
                success=True,
                data=result,
                retry_count=0
            )
            
        except ValidationError as e:
            logging.error(f"Validation error: {e}")
            return SafeResponse(
                success=False,
                error=f"Data validation failed: {str(e)}"
            )
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return SafeResponse(
                success=False,
                error=f"Analysis failed: {str(e)}"
            )
    
    def analyze_with_fallback(self, data: str, fallback_model: str = "gpt-3.5-turbo") -> SafeResponse:
        """Try primary model, fallback to secondary if it fails."""
        try:
            return self.safe_analyze(data)
        except Exception as primary_error:
            logging.warning(f"Primary model failed: {primary_error}. Trying fallback.")
            
            try:
                old_model = self.model
                self.model = fallback_model
                result = self.safe_analyze(data)
                self.model = old_model  # Restore original model
                return result
            except Exception as fallback_error:
                self.model = old_model  # Restore original model
                return SafeResponse(
                    success=False,
                    error=f"Both models failed. Primary: {primary_error}, Fallback: {fallback_error}"
                )

# Usage
analyzer = RobustAnalyzer()

response = analyzer.analyze_with_fallback("Complex data analysis request")
if response.success:
    print(f"Analysis successful: {response.data}")
else:
    print(f"Analysis failed: {response.error}")
```

### Graceful Degradation

```python
from pydantic import BaseModel
from typing import Optional, List, Union
from agentsoup import llm

class AnalysisLevel(BaseModel):
    level: str  # "basic", "detailed", "comprehensive"
    features: List[str]

class GradedResponse(BaseModel):
    analysis: str
    level_achieved: str
    features_included: List[str]
    limitations: Optional[List[str]] = None

class GradedAnalyzer:
    def __init__(self):
        self.analysis_levels = {
            "comprehensive": AnalysisLevel(
                level="comprehensive",
                features=["detailed_analysis", "recommendations", "risk_assessment", "forecasting"]
            ),
            "detailed": AnalysisLevel(
                level="detailed", 
                features=["detailed_analysis", "recommendations"]
            ),
            "basic": AnalysisLevel(
                level="basic",
                features=["basic_analysis"]
            )
        }
    
    def analyze_with_degradation(self, data: str, preferred_level: str = "comprehensive") -> GradedResponse:
        """Attempt analysis at preferred level, degrade gracefully if needed."""
        
        levels_to_try = ["comprehensive", "detailed", "basic"]
        start_index = levels_to_try.index(preferred_level) if preferred_level in levels_to_try else 0
        
        for level in levels_to_try[start_index:]:
            try:
                result = self._analyze_at_level(data, level)
                return result
            except Exception as e:
                if level == "basic":  # Last resort failed
                    return GradedResponse(
                        analysis="Analysis failed at all levels",
                        level_achieved="none",
                        features_included=[],
                        limitations=[f"Complete failure: {str(e)}"]
                    )
                continue
        
        return GradedResponse(
            analysis="Unexpected error in degradation logic",
            level_achieved="error",
            features_included=[],
            limitations=["System error"]
        )
    
    def _analyze_at_level(self, data: str, level: str) -> GradedResponse:
        """Perform analysis at specific level."""
        level_config = self.analysis_levels[level]
        
        @llm(model="gpt-4o-mini", max_tokens=self._get_token_limit(level))
        def level_analysis(text: str, features: List[str]) -> str:
            feature_prompt = ", ".join(features)
            return f"Analyze this data with focus on: {feature_prompt}. Data: {text}"
        
        try:
            analysis_result = level_analysis(data, level_config.features)
            
            return GradedResponse(
                analysis=analysis_result,
                level_achieved=level,
                features_included=level_config.features,
                limitations=self._get_limitations(level) if level != "comprehensive" else None
            )
        except Exception as e:
            raise Exception(f"Level {level} analysis failed: {str(e)}")
    
    def _get_token_limit(self, level: str) -> int:
        limits = {"comprehensive": 2000, "detailed": 1000, "basic": 500}
        return limits.get(level, 500)
    
    def _get_limitations(self, level: str) -> List[str]:
        limitations = {
            "detailed": ["No forecasting", "Limited risk assessment"],
            "basic": ["No recommendations", "No risk assessment", "No forecasting", "Surface-level analysis only"]
        }
        return limitations.get(level, [])

# Usage
analyzer = GradedAnalyzer()

# Try comprehensive analysis first
response = analyzer.analyze_with_degradation("Complex market data...", "comprehensive")
print(f"Level achieved: {response.level_achieved}")
print(f"Features included: {response.features_included}")
if response.limitations:
    print(f"Limitations: {response.limitations}")
```

---

## Performance Optimization

### Batch Processing

```python
from pydantic import BaseModel
from typing import List, Dict, Any
from agentsoup import llm
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchResult(BaseModel):
    item_id: str
    result: Any
    success: bool
    error: Optional[str] = None

class BatchProcessor:
    def __init__(self, model: str = "gpt-4o-mini", max_workers: int = 5):
        self.model = model
        self.max_workers = max_workers
    
    def process_batch_sync(self, items: List[Dict[str, Any]], processor_func) -> List[BatchResult]:
        """Process items in batch using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._safe_process_item, item, processor_func): item 
                for item in items
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(BatchResult(
                        item_id=item.get('id', 'unknown'),
                        result=None,
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def _safe_process_item(self, item: Dict[str, Any], processor_func) -> BatchResult:
        """Safely process a single item."""
        try:
            result = processor_func(item)
            return BatchResult(
                item_id=item.get('id', 'unknown'),
                result=result,
                success=True
            )
        except Exception as e:
            return BatchResult(
                item_id=item.get('id', 'unknown'),
                result=None,
                success=False,
                error=str(e)
            )

# Usage example
class TextSummary(BaseModel):
    summary: str
    key_points: List[str]
    word_count: int

@llm(model="gpt-4o-mini")
def summarize_text(text: str) -> TextSummary:
    return f"Summarize this text: {text}"

def process_item(item: Dict[str, Any]) -> TextSummary:
    return summarize_text(item['text'])

# Process multiple texts in batch
processor = BatchProcessor(max_workers=3)
items = [
    {'id': '1', 'text': 'Long article 1...'},
    {'id': '2', 'text': 'Long article 2...'},
    {'id': '3', 'text': 'Long article 3...'},
]

results = processor.process_batch_sync(items, process_item)
for result in results:
    if result.success:
        print(f"Item {result.item_id}: {result.result.summary}")
    else:
        print(f"Item {result.item_id} failed: {result.error}")
```

### Caching and Memoization

```python
from pydantic import BaseModel
from typing import Dict, Any, Optional
from agentsoup import llm
import hashlib
import json
import time
import pickle
import os

class CacheEntry:
    def __init__(self, result: Any, timestamp: float, ttl: Optional[float] = None):
        self.result = result
        self.timestamp = timestamp
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

class LLMCache:
    def __init__(self, cache_dir: str = "./llm_cache", default_ttl: float = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict, model: str) -> str:
        """Generate cache key from function parameters."""
        cache_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs,
            'model': model
        }
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                return entry.result
            else:
                del self.memory_cache[key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    if not entry.is_expired():
                        # Load back to memory cache
                        self.memory_cache[key] = entry
                        return entry.result
                    else:
                        os.remove(cache_file)
            except Exception:
                pass  # Ignore corrupted cache files
        
        return None
    
    def set(self, key: str, result: Any, ttl: Optional[float] = None) -> None:
        """Set cached result."""
        if ttl is None:
            ttl = self.default_ttl
        
        entry = CacheEntry(result, time.time(), ttl)
        
        # Store in memory cache
        self.memory_cache[key] = entry
        
        # Store in disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception:
            pass  # Ignore disk cache errors

# Global cache instance
_cache = LLMCache()

def cached_llm(model: str = "gpt-4o-mini", ttl: Optional[float] = None, **llm_kwargs):
    """LLM decorator with caching."""
    def decorator(func):
        original_llm_func = llm(model=model, **llm_kwargs)(func)
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _cache._get_cache_key(func.__name__, args, kwargs, model)
            
            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Not in cache, call original function
            result = original_llm_func(*args, **kwargs)
            
            # Store in cache
            _cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Usage
class Analysis(BaseModel):
    summary: str
    sentiment: str
    keywords: List[str]

@cached_llm(model="gpt-4o-mini", ttl=1800)  # Cache for 30 minutes
def analyze_cached(text: str) -> Analysis:
    """Cached analysis function."""
    return f"Analyze this text: {text}"

# First call - will hit the LLM
result1 = analyze_cached("Sample text for analysis")

# Second call with same text - will use cache
result2 = analyze_cached("Sample text for analysis")  # Much faster!
```

---

## Production Patterns

### Configuration Management

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from agentsoup import llm
import os
import yaml
import json

class ModelConfig(BaseModel):
    name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class LLMServiceConfig(BaseModel):
    primary_model: ModelConfig
    fallback_model: Optional[ModelConfig] = None
    retry_attempts: int = 3
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 60
    cache_ttl_seconds: int = 3600

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get('LLM_CONFIG_PATH', './llm_config.yaml')
        self.config = self._load_config()
    
    def _load_config(self) -> LLMServiceConfig:
        """Load configuration from file or environment."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
                return LLMServiceConfig(**config_data)
        else:
            # Default configuration
            return LLMServiceConfig(
                primary_model=ModelConfig(name="gpt-4o-mini"),
                fallback_model=ModelConfig(name="gpt-3.5-turbo"),
                retry_attempts=3
            )
    
    def get_llm_kwargs(self, use_fallback: bool = False) -> Dict[str, Any]:
        """Get LLM kwargs from configuration."""
        model_config = self.config.fallback_model if use_fallback and self.config.fallback_model else self.config.primary_model
        
        return {
            'model': model_config.name,
            'temperature': model_config.temperature,
            'max_tokens': model_config.max_tokens,
            'top_p': model_config.top_p,
            'frequency_penalty': model_config.frequency_penalty,
            'presence_penalty': model_config.presence_penalty
        }

# Usage
config_manager = ConfigManager()

class ProductionAnalysis(BaseModel):
    result: str
    confidence: float
    model_used: str

def create_production_analyzer():
    """Factory function to create configured analyzer."""
    
    @llm(**config_manager.get_llm_kwargs())
    def analyze_production(data: str) -> ProductionAnalysis:
        return f"Analyze this production data: {data}"
    
    return analyze_production

# Create analyzer with configuration
analyzer = create_production_analyzer()
result = analyzer("Production metrics data...")
```

### Monitoring and Logging

```python
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from agentsoup import llm, CompleteResponse
import logging
import time
import json
from datetime import datetime
import uuid

class LLMMetrics(BaseModel):
    request_id: str
    function_name: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime

class MetricsCollector:
    def __init__(self):
        self.metrics: List[LLMMetrics] = []
        self.logger = logging.getLogger('llm_metrics')
    
    def record_metrics(self, metrics: LLMMetrics):
        """Record metrics for analysis."""
        self.metrics.append(metrics)
        self.logger.info(f"LLM Call: {metrics.model_dump_json()}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics:
            return {"total_calls": 0}
        
        successful_calls = [m for m in self.metrics if m.success]
        failed_calls = [m for m in self.metrics if not m.success]
        
        return {
            "total_calls": len(self.metrics),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "success_rate": len(successful_calls) / len(self.metrics),
            "avg_duration": sum(m.duration_seconds for m in self.metrics) / len(self.metrics),
            "total_tokens": sum(m.total_tokens or 0 for m in self.metrics),
            "models_used": list(set(m.model for m in self.metrics))
        }

# Global metrics collector
_metrics = MetricsCollector()

def monitored_llm(model: str = "gpt-4o-mini", **llm_kwargs):
    """LLM decorator with monitoring."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            try:
                # Create the LLM function
                llm_func = llm(model=model, **llm_kwargs)(func)
                
                # Call it and measure
                result = llm_func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract token usage if available
                tokens_used = None
                input_tokens = None
                output_tokens = None
                
                if isinstance(result, CompleteResponse):
                    if hasattr(result.completion, 'usage'):
                        tokens_used = result.completion.usage.total_tokens
                        input_tokens = getattr(result.completion.usage, 'prompt_tokens', None)
                        output_tokens = getattr(result.completion.usage, 'completion_tokens', None)
                
                # Record metrics
                metrics = LLMMetrics(
                    request_id=request_id,
                    function_name=func.__name__,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=tokens_used,
                    duration_seconds=duration,
                    success=True,
                    timestamp=datetime.now()
                )
                _metrics.record_metrics(metrics)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metrics
                metrics = LLMMetrics(
                    request_id=request_id,
                    function_name=func.__name__,
                    model=model,
                    duration_seconds=duration,
                    success=False,
                    error_message=str(e),
                    timestamp=datetime.now()
                )
                _metrics.record_metrics(metrics)
                
                raise e
        
        return wrapper
    return decorator

# Usage
class MonitoredAnalysis(BaseModel):
    analysis: str
    confidence: float

@monitored_llm(model="gpt-4o-mini")
def monitored_analyze(text: str) -> CompleteResponse[MonitoredAnalysis]:
    """Monitored analysis function."""
    return f"Analyze: {text}"

# Make some calls
result1 = monitored_analyze("Sample text 1")
result2 = monitored_analyze("Sample text 2")

# Get metrics summary
summary = _metrics.get_summary()
print(f"Metrics Summary: {json.dumps(summary, indent=2)}")
```

This comprehensive documentation covers all the public APIs, functions, and components of AgentSoup with detailed examples, usage patterns, and best practices. The documentation is organized into:

1. **API_DOCUMENTATION.md** - Complete API reference with all classes, methods, and usage examples
2. **QUICK_START_GUIDE.md** - Step-by-step guide for beginners to get started quickly
3. **ADVANCED_USAGE.md** - Advanced patterns, error handling, performance optimization, and production-ready implementations

Each document provides:
- Detailed parameter descriptions
- Return type information
- Comprehensive usage examples
- Best practices and common patterns
- Error handling strategies
- Production-ready code examples

The documentation covers all aspects of the library from basic usage to advanced production patterns, making it easy for developers to understand and effectively use AgentSoup in their projects.