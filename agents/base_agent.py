from abc import ABC, abstractmethod
from typing import List, Optional, Any
from llama_index.core import Document

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    initialized = False

    @abstractmethod
    def init_models(self) -> None:
        """Initialize required models (embedding, LLM, etc.)."""
        pass


    @abstractmethod
    def init_agent(self) -> None:
        """Initialize the agent workflow with tools and configurations."""
        pass

    @abstractmethod
    def query(self, question: str, **kwargs: Any) -> Any:
        """Process a query and return a response.
        
        Args:
            question: The question to answer
            **kwargs: Additional arguments for query processing
            
        Returns:
            Response from the agent
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get statistics about the agent's state.
        
        Returns:
            Dictionary containing statistics (e.g., number of loaded documents,
            index size, etc.)
        """
        pass