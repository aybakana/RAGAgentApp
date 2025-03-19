# agent.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
    Context
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.core.postprocessor import SimilarityPostprocessor
from config.settings import config

class RAGAgent:
    def __init__(self):
        self.nodes = []
        self.embedding_model = None
        self.llm = None
        self.context = None
        self.query_engine = None
        self.query_engine_agent = None
        self.initialized = False

    def init_models(self):
        """
        Initialize the embedding model and LLM.
        """
        self.embedding_model = HuggingFaceEmbedding(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )
        self.retrieval_llm = Ollama(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            request_timeout=config.llm.request_timeout
        )
        self.gemini_llm = Gemini(
            model=config.geminiLLM.model_name,
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
        Build the VectorStore index retriever from the loaded nodes.
        """
        if not self.nodes:
            raise ValueError("No nodes loaded. Please load documents first.")
        
        # Create vector store index
        self.index = VectorStoreIndex(self.nodes, embed_model=self.embedding_model)
        
        vector_retriever = self.index.as_retriever(similarity_top_k=5)
        
        # Create query engine with post-processing
        self.query_engine = self.index.as_query_engine(
            llm=self.retrieval_llm,  # Default to retrieval LLM
            retriever=vector_retriever,  # Default to vector retriever
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )

    def init_agent(self, additional_tools=None):
        """
        Initialize the RAG agent workflow.
        """
        if not self.nodes or not self.query_engine:
            raise ValueError("Documents must be loaded and index built before initializing agent.")

        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="search_context",
            description=(
                "Search through the loaded documents and code to find relevant information. "
                "Input should be a specific search query. "
                "Returns relevant code examples, documentation, or text passages."
            ),
            return_direct=False,
        )

        system_prompt = """You are an AI assistant designed to answer queries efficiently by leveraging a search_context tool. You follow a structured approach to think critically, retrieve relevant data, and respond clearly.

Capabilities:
You have access to a search_context tool that retrieves up-to-date information.
You follow the Thought-Action-Input format to ensure structured execution.
You analyze the query, decide if a search is required, and provide a relevant response.
Instructions:

Process User Query Thoughtfully:

If the answer is known, respond directly with clear, concise information.
If additional information is needed, use the search_context tool before responding.
Follow the Thought-Action-Input Format:

Thought: Analyze the query and decide if a search is required.
Action: Choose the best action (search_context) if needed.
Input: Provide the search query in a structured format.
Ensure Output is Well-Formatted:

If using the search_context tool, ensure output follows:
Thought: [Reasoning behind search]  
Action: search_context  
Input: [Query to retrieve relevant data]  
After receiving search results, process them into a user-friendly answer.
Example Execution:
User Query: "What are the latest advancements in quantum computing?"

Correct Response Flow:
Thought: The field of quantum computing evolves rapidly, and up-to-date information is necessary. I should perform a search.  
Action: search_context  
Input: "Latest advancements in quantum computing 2025"  
(After retrieving search results, the AI processes the information and provides a detailed, clear response.)

Additional Constraints:

Do not generate responses before processing search results.
If a search query returns no results, inform the user and ask for clarifications.
Ensure responses remain accurate, concise, and informative."""

        # Include additional tools if provided
        tools = [query_engine_tool]
        if additional_tools:
            tools.extend(additional_tools)

        self.query_engine_agent = AgentWorkflow.from_tools_or_functions(
            tools,
            llm=self.gemini_llm,
            system_prompt=system_prompt,
            verbose=True
        )
        # init the Context from the AgentWorkflow
        self.context = Context(self.query_engine_agent)
        self.initialized = True

    def query(self, question):
        """
        Query the RAG agent with a question.
        Args:
            question (str): The question to ask
        Returns:
            dict: Dictionary containing response text and debug info
        """
        if not self.initialized:
            raise ValueError("Agent not initialized. Call `init_agent` first.")
        
        # Use default vector retriever
        raw_response = self.query_engine_agent.query(question)
        
        # Extract useful information from the raw response
        response_dict = {
            'response_text': str(raw_response),
            'debug_info': {
                'source_nodes': [],
                'response_metadata': {}
            }
        }

        # Try to extract metadata and source nodes if available
        try:
            if hasattr(raw_response, 'metadata'):
                response_dict['debug_info']['response_metadata'] = raw_response.metadata
            
            # Extract source nodes if available
            if hasattr(raw_response, 'source_nodes'):
                for node in raw_response.source_nodes:
                    node_info = {
                        'text': node.text,
                        'score': node.score if hasattr(node, 'score') else None,
                        'file_path': node.metadata.get('file_path', 'Unknown'),
                    }
                    response_dict['debug_info']['source_nodes'].append(node_info)
        except Exception as e:
            print(f"Warning: Could not extract debug info: {str(e)}")
            
        return response_dict

    def get_stats(self):
        """
        Get statistics about the loaded documents and nodes.
        Returns:
            dict: Dictionary containing statistics like total_nodes
        """
        return {
            "total_nodes": len(self.nodes),
            "is_initialized": self.initialized
        }

    def is_initialized(self):
        """Check if the agent is initialized."""
        return self.initialized