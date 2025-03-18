# agent.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.core.retrievers import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from config.settings import config

class RAGAgent:
    def __init__(self):
        self.nodes = []
        self.embedding_model = None
        self.llm = None
        self.query_engine = None
        self.query_engine_agent = None
        self.bm25_retriever = None

    def init_models(self):
        """
        Initialize the embedding model and LLM.
        """
        self.embedding_model = HuggingFaceEmbedding(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )
        self.llm = Ollama(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            request_timeout=config.llm.request_timeout
        )

    def load_documents(self, directories, extensions=[".py", ".txt", ".md"]):
        """
        Load documents from specified directories and split them into nodes.
        """
        # Load code files
        python_documents = SimpleDirectoryReader(input_dir=directories, recursive=True, required_exts=[".py"]).load_data()
        code_splitter = CodeSplitter(language="python", max_chars=1000)
        code_nodes = code_splitter.get_nodes_from_documents(python_documents, show_progress=True)

        # Load text files
        text_documents = SimpleDirectoryReader(input_dir=directories, recursive=True, required_exts=[".txt", ".md"]).load_data()
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        text_nodes = text_splitter.get_nodes_from_documents(text_documents, show_progress=True)

        # Combine nodes
        self.nodes.extend(code_nodes + text_nodes)
        print(f"Loaded {len(self.nodes)} nodes from {directories}.")

    def build_index(self):
        """
        Build the VectorStore index and BM25 retriever from the loaded nodes.
        """
        if not self.nodes:
            raise ValueError("No nodes loaded. Please load documents first.")
        
        # Create vector store index
        self.index = VectorStoreIndex(self.nodes, embed_model=self.embedding_model)
        
        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=5)
        
        # Create hybrid retriever combining vector and BM25
        vector_retriever = self.index.as_retriever(similarity_top_k=5)
        retrievers = [vector_retriever, self.bm25_retriever]
        
        # Create query engine with post-processing
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            retriever=vector_retriever,  # Default to vector retriever
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )

    def init_agent(self):
        """
        Initialize the RAG agent workflow.
        """
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="QueryEngineTool",
            description="Retrieves relevant code snippets and best practices for software development queries.",
            return_direct=False,
        )
        self.query_engine_agent = AgentWorkflow.from_tools_or_functions(
            [query_engine_tool],
            llm=self.llm,
            system_prompt="You are a specialized assistant that retrieves structured and relevant code knowledge.",
            verbose=True
        )

    def query(self, question, use_bm25=False):
        """
        Query the RAG agent with a question.
        Args:
            question (str): The question to ask
            use_bm25 (bool): Whether to use BM25 retriever instead of vector retriever
        """
        if not self.query_engine_agent:
            raise ValueError("Agent not initialized. Call `init_agent` first.")
        
        if use_bm25 and self.bm25_retriever:
            # Use BM25 retriever for this query
            retrieved_nodes = self.bm25_retriever.retrieve(question)
            # Update query engine with retrieved nodes
            response = self.query_engine.query(question, retrieved_nodes=retrieved_nodes)
        else:
            # Use default vector retriever
            response = self.query_engine_agent.query(question)
            
        return response

from .rag_agent import RAGAgent

# Re-export RAGAgent as the default agent
__all__ = ['RAGAgent']