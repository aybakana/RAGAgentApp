from dataclasses import dataclass, field
from typing import Optional, List

# Logging levels can be DEBUG, INFO, WARNING, ERROR, or CRITICAL.
LOG_LEVEL = "DEBUG"            # Overall log level for the application.
CONSOLE_LOG_LEVEL = "DEBUG"      # Log level for the console handler.
FILE_LOG_LEVEL = "INFO"          # Log level for the file handler.

# Log file configuration
LOG_TO_FILE = True
LOG_FILE_PATH = "./logs/rag_app.log"   # Ensure this directory exists or is creatable.
ENABLE_ERROR_LOGGING = True
ERROR_LOG_FILE_PATH = "./logs/rag_app_errors.log"

# Default error message templates for user friendly display.
ERROR_MESSAGE_TEMPLATE = "Something went wrong. Please try again later."

@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    max_length: int = 512

@dataclass
class LLMConfig:
    model_name: str = "llama3.2:1b"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.95
    request_timeout: int = 600

@dataclass
class SplitterConfig:
    # Code splitter settings
    code_chunk_size: int = 1000
    code_chunk_overlap: int = 20
    code_language: str = "python"
    
    # Text splitter settings
    text_chunk_size: int = 1024
    text_chunk_overlap: int = 20
    
    # File extensions to process
    code_extensions: List[str] = field(default_factory=lambda: [".py", ".js", ".ts", ".java", ".cpp", ".cs"])
    text_extensions: List[str] = field(default_factory=lambda: [".txt", ".md", ".rst", ".json", ".yaml", ".yml"])

@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    
    # Default directories for document processing
    default_dirs: List[str] = field(default_factory=lambda: ["docs", "src", "tests"])

# Create default configuration instance
config = AppConfig()