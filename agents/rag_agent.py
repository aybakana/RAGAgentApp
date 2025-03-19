from typing import List, Optional, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from config.settings import config
from processors.code_splitter import CustomCodeSplitter
from processors.text_splitter import CustomTextSplitter
from .base_agent import BaseAgent

class RAGAgent(BaseAgent):
    def __init__(self):
        self.nodes = []
        self.embedding_model = None
        self.llm = None
        self.llmGemini = None
        self.query_engine = None
        self.agent = None
        self.index = None
        self.code_splitter = CustomCodeSplitter()
        self.text_splitter = CustomTextSplitter()
        
    def init_models(self) -> None:
        """Initialize the embedding model and LLM."""
        self.embedding_model = HuggingFaceEmbedding(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )
        self.llm15flashGemini = Gemini(
            model=config.gemini15FlashLLM.model_name,
        )
        self.llm20FlashGemini = Gemini(
            model=config.gemini20FlashLLM.model_name,
        )

    def load_documents(self, directories: str, extensions: Optional[List[str]] = None) -> None:
        """Load and process documents from specified directories."""
        if extensions is None:
            code_exts = config.splitter.code_extensions
            text_exts = config.splitter.text_extensions
        else:
            code_exts = [".py"]
            text_exts = [".txt", ".md"]

        # Load code files
        if code_exts:
            code_documents = SimpleDirectoryReader(
                input_dir=directories, 
                recursive=True, 
                #required_exts=code_exts,
                required_exts=[".py"]
            ).load_data()
            code_nodes = self.code_splitter.split_documents(code_documents)
            self.nodes.extend(code_nodes)

        # Load text files
        if text_exts:
            text_documents = SimpleDirectoryReader(
                input_dir=directories, 
                recursive=True, 
                #required_exts=text_exts,
                required_exts=[".txt", ".md"]
            ).load_data()
            text_nodes = self.text_splitter.split_documents(text_documents)
            self.nodes.extend(text_nodes)


    def build_index(self) -> None:
        """Build the vector store index and initialize retrievers."""
        if not self.nodes:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        if not self.embedding_model or not self.llm15flashGemini:
            raise ValueError("Models not initialized. Call init_models first.")
        
        # Create vector store index
        self.index = VectorStoreIndex(
            nodes=self.nodes, 
            embed_model=self.embedding_model
        )
        
        # Create query engine with post-processing
        self.query_engine = self.index.as_query_engine(
            llm=self.llm15flashGemini,
            similarity_top_k=5,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )

    def init_agent(self) -> None:
        """Initialize the RAG agent with query engine tool."""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call build_index first.")
        
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="QueryEngineTool",
            description="Retrieves relevant code snippets and best practices for software development queries.",
            return_direct=False,
        )
        
        self.agent = ReActAgent.from_tools(
            tools=[query_engine_tool],
            llm=self.llm20FlashGemini,
            verbose=True
        )

    def query(self, question: str, **kwargs: Any) -> Any:
        """Process a query using the agent.
        
        Args:
            question: The question to answer
            **kwargs: Additional arguments for query processing
            
        Returns:
            Response from the agent
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call init_agent first.")
        
        response = self.agent.chat(question)
        return response

    def get_stats(self) -> dict:
        """Get statistics about the agent's state."""
        return {
            "total_nodes": len(self.nodes) if self.nodes else 0,
            "index_initialized": self.index is not None if hasattr(self, 'index') else False,
            "models_initialized": bool(self.embedding_model and self.llm),
            "agent_initialized": bool(self.agent)
        }