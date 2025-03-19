from typing import List, Optional, Any, Dict
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.core.postprocessor import SimilarityPostprocessor
from config.settings import config
from processors.code_splitter import CustomCodeSplitter
from processors.text_splitter import CustomTextSplitter
from agents.base_agent import BaseAgent

class RAGAgent(BaseAgent):
    def __init__(self):
        self.indexed_directories = set()
        self.embedding_model = None
        self.llmQuery = None # Ollama or Gemini with standard capabilities
        self.llmMain = None  # Gemini with Flash capabilities
        self.agent = None
        self.nodes = []
        self.query_engines: Dict[str, Any] = {}
        self.indexes: Dict[str, VectorStoreIndex] = {}        
        self.code_splitter = CustomCodeSplitter()
        self.text_splitter = CustomTextSplitter()
        self.init_models()
        
    def init_models(self) -> None:
        """Initialize the embedding model and LLM."""
        self.embedding_model = HuggingFaceEmbedding(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )
        self.llmOllama = Ollama(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            request_timeout=config.llm.request_timeout
        )
        self.llm15flashGemini = Gemini(
            model=config.gemini15FlashLLM.model_name,
        )
        self.llm20FlashGemini = Gemini(
            model=config.gemini20FlashLLM.model_name,
        )

    def load_directory(self, directory: str, extensions: Optional[List[str]] = None) -> List[Any]:
        """Load and process documents from specified directories."""
        if directory in self.indexed_directories:
            print(f"Directory '{directory}' has already been indexed.")
            return []

        if extensions is None:
            code_exts = config.splitter.python_extensions
            text_exts = config.splitter.text_extensions
        else:
            code_exts = [".py"]
            text_exts = [".txt", ".md"]

        self.nodes = []

        # Load code files
        if code_exts:
            code_documents = SimpleDirectoryReader(
                input_dir=directory,
                recursive=True,
                required_exts=code_exts
            ).load_data()
            code_nodes = self.code_splitter.split_documents(code_documents)
            self.nodes.extend(code_nodes)

        # Load text files
        if text_exts:
            text_documents = SimpleDirectoryReader(
                input_dir=directory, 
                recursive=True, 
                required_exts=text_exts
            ).load_data()
            text_nodes = self.text_splitter.split_documents(text_documents)
            #self.nodes.extend(text_nodes)

        self.indexed_directories.add(directory)
        return self.nodes

    def build_index_and_query_engine(self, directory: str, nodes: List[Any]) -> None:
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
        self.indexes[directory].persist(persist_dir=storage_path)

    def save_all_indexes(self, storage_base_path: str) -> None:
        """Save all indexes to the specified base storage path."""
        for directory, index in self.indexes.items():
            storage_path = f"{storage_base_path}/{directory.replace('/', '_')}_index.json"
            index.storage_context.persist(persist_dir=storage_path)

    def load_index_and_query_engine(self, directory: str, storage_path: str) -> bool:
        """Load the index for a specific directory from the specified storage path."""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            index = load_index_from_storage(storage_context)
        except:
            return False
        
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
        return True

    def set_directories(self, directories: List[str]) -> None:
        """Set the directories to be indexed."""
        self.indexed_directories = set(directories)

    def load_all_indexes(self, storage_base_path: str, directories) -> None:
        """Load all indexes from the specified base storage path."""
        for directory in directories:
            storage_path = f"{storage_base_path}/{directory.replace('/', '_')}_index.json"
            self.load_index_and_query_engine(directory, storage_path)


    def add_directory_query_engine(self, directory: str, extensions: Optional[List[str]] = None) -> None:
        """Add a directory to the index and create a query engine for it."""
        nodes = self.load_directory(directory, extensions)
        self.build_index_and_query_engine(directory, nodes)

    def remove_directory(self, directory: str) -> None:
        """Remove a directory from the index."""
        if directory not in self.indexes:
            print(f"Directory '{directory}' not found in indexes.")
            self.indexed_directories.remove(directory)
            return
                
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
        if len(self.query_engines.items()) < 1:
            raise ValueError("No query engines available. Add directories and initialize the agent first.")
        
        query_engine_tools = []
        for directory, query_engine in self.query_engines.items():
            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=f"QueryEngineTool_{directory}",
                description=f"Retrieves relevant code snippets and best practices for software development queries from {directory}.",
                return_direct=False,
            )
            query_engine_tools.append(tool)
        
        self.agent = ReActAgent.from_tools(
            tools=query_engine_tools,
            llm=self.llm20FlashGemini,
            verbose=True
        )

    def query_engine_query(self, question: str) -> Any:
        """Process a query using query engines from all indexed directories.
        
        Args:
            question: The question to ask across all indexed directories
            
        Returns:
            Dict containing responses from each directory and combined metadata
        """
        if not self.query_engines:
            raise ValueError("No query engines available. Add directories and initialize the agent first.")

        responses = []
        source_nodes = []
        
        # Query each directory's query engine
        for directory, query_engine in self.query_engines.items():
            response = query_engine.query(question)
            if hasattr(response, 'source_nodes'):
                source_nodes.extend([
                    {
                        'file_path': node.metadata.get('file_path', 'Unknown'),
                        'score': getattr(node, 'score', 0.0),
                        'text': node.text
                    }
                    for node in response.source_nodes
                ])
            responses.append({
                'directory': directory,
                'response': str(response)
            })

        # Combine responses into a structured format
        combined_response = {
            'response_text': '\n\n'.join([
                f"From {r['directory']}:\n{r['response']}"
                for r in responses
            ]),
            'debug_info': {
                'source_nodes': sorted(source_nodes, key=lambda x: x['score'], reverse=True),
                'response_metadata': {
                    'total_directories': len(responses),
                    'directories': [r['directory'] for r in responses]
                }
            }
        }
        
        return combined_response

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
            "models_initialized": bool(self.embedding_model and self.llmOllama and self.llm15flashGemini),
            "agent_initialized": bool(self.agent)
        }