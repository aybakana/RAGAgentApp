# agent.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini

class RAGAgent:
    def __init__(self):
        self.nodes = []
        self.embedding_model = None
        self.llm = None
        self.query_engine = None
        self.query_engine_agent = None

    def init_models(self):
        """
        Initialize the embedding model and LLM.
        """
        self.embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model="gemma3:1b", temperature=0.7, request_timeout=600)

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
        Build the VectorStore index from the loaded nodes.
        """
        if not self.nodes:
            raise ValueError("No nodes loaded. Please load documents first.")
        self.index = VectorStoreIndex(self.nodes, embed_model=self.embedding_model)
        self.query_engine = self.index.as_query_engine(llm=self.llm, similarity_top_k=5)

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

    def query(self, question):
        """
        Query the RAG agent with a question.
        """
        if not self.query_engine_agent:
            raise ValueError("Agent not initialized. Call `init_agent` first.")
        return self.query_engine_agent.query(question)