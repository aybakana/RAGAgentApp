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
        self.indexed_directories = set()
        self.embedding_model = None
        self.llmQuery = None # Ollama or Gemini with standard capabilities
        self.llmMain = None  # Gemini with Flash capabilities
        self.agent = None
        self.query_engines: Dict[str, Any] = {}
        self.indexes: Dict[str, VectorStoreIndex] = {}        
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

    def load_directory(self, directory: str, extensions: Optional[List[str]] = None) -> List[Any]:
        """Load and process documents from specified directories."""
        if directories in self.indexed_directories:
            print(f"Directory '{directories}' has already been indexed.")
            return []

        if extensions is None:
            code_exts = config.splitter.python_extensions
            text_exts = config.splitter.text_extensions
        else:
            code_exts = [".py"]
            text_exts = [".txt", ".md"]

        nodes = []

        # Load code files
        if code_exts:
            code_documents = SimpleDirectoryReader(
                input_dir=directories, 
                recursive=True, 
                required_exts=code_exts
            ).load_data()
            code_nodes = self.code_splitter.split_documents(code_documents)
            nodes.extend(code_nodes)

        # Load text files
        if text_exts:
            text_documents = SimpleDirectoryReader(
                input_dir=directories, 
                recursive=True, 
                required_exts=text_exts
            ).load_data()
            text_nodes = self.text_splitter.split_documents(text_documents)
            nodes.extend(text_nodes)

        self.indexed_directories.add(directories)
        return nodes

    def build_index(self, directory: str, nodes: List[Any]) -> None:
        """Build the vector store index for a specific directory."""
        if not nodes:
            raise ValueError("No documents loaded. Call load_directory first.")
        
        if not self.embedding_model or not self.llm15flashGemini:
            raise ValueError("Models not initialized. Call init_models first.")
        
        # Create vector store index
        index = VectorStoreIndex(
            nodes=nodes, 
            embed_model=self.embedding_model
        )
        self.indexes[directory] = index
        
        # Create query engine with post-processing
        query_engine = index.as_query_engine(
            llm=self.llm15flashGemini,
            similarity_top_k=5,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        self.query_engines[directory] = query_engine

    def save_index(self, directory: str, storage_path: str) -> None:
        """Save the index for a specific directory to the specified storage path."""
        if directory not in self.indexes:
            raise ValueError(f"Index for directory '{directory}' not found.")
        self.indexes[directory].save(storage_path)

    def save_all_indexes(self, storage_base_path: str) -> None:
        """Save all indexes to the specified base storage path."""
        for directory, index in self.indexes.items():
            storage_path = f"{storage_base_path}/{directory.replace('/', '_')}_index.json"
            index.save(storage_path)

    def load_index_and_query_engine(self, directory: str, storage_path: str) -> None:
        """Load the index for a specific directory from the specified storage path."""
        index = VectorStoreIndex.load(storage_path)
        self.indexes[directory] = index
        
        # Create query engine with post-processing
        query_engine = index.as_query_engine(
            llm=self.llm15flashGemini,
            similarity_top_k=5,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        self.query_engines[directory] = query_engine

    def load_all_indexes(self, storage_base_path: str) -> None:
        """Load all indexes from the specified base storage path."""
        for directory in self.indexed_directories:
            storage_path = f"{storage_base_path}/{directory.replace('/', '_')}_index.json"
            self.load_index_and_query_engine(directory, storage_path)        


    def add_directory_query_engine(self, directory: str, extensions: Optional[List[str]] = None) -> None:
        """Add a directory to the index and create a query engine for it."""
        nodes = self.load_directory(directory, extensions)
        self.build_index(directory, nodes)

    def remove_directory(self, directory: str) -> None:
        """Remove a directory from the index."""
        if directory not in self.indexes:
            raise ValueError(f"Directory '{directory}' not found in indexes.")
                
        # Remove index and query engine for the directory
        del self.indexes[directory]
        del self.query_engines[directory]
        self.indexed_directories.remove(directory)

    def get_query_engine(self, directory: str) -> Any:
        """Retrieve the query engine for a specific directory."""
        if directory not in self.query_engines:
            raise ValueError(f"Query engine for directory '{directory}' not found.")
        return self.query_engines[directory]        

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

    def query_engine_query(self, directory: str, question: str) -> Any:
        """Process a query using the agent for a specific directory.
        
        From the UI, The user can ask a question and the agent will return a response from the specified directory.
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call init_agent first.")
        directories = self.indexed_directories
        responses = []
        for d in directories:
            query_engine = self.get_query_engine(directory)
            responses.append(query_engine.query(question))
        return responses

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