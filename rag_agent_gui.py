import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt6.QtGui import QFont
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
#from custom_code_splitter import process_python_files
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
import asyncio
import time

# Option 1: Using a callback handler (if supported by your LLM implementation)
# This handler prints out all inputs (prompts) and outputs of the LLM.
from langchain_core.callbacks import BaseCallbackHandler

class PromptResponseCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to log the LLM prompt and response."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Called when LLM starts processing, receiving the list of prompt(s).
        print("LLM Prompt(s):")
        for idx, prompt in enumerate(prompts, start=1):
            print(f"Prompt {idx}: {prompt}")

    def on_llm_end(self, response, **kwargs):
        # Called when the LLM finishes processing, with the response data.
        print("LLM Response:")
        # Depending on the LLM, the response might be structured; adapt as needed.
        print(response.generations)


class CodeQueryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_agents()        

    def init_ui(self):
        self.setWindowTitle("Code Query Assistant")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()

        # Title Label
        self.title_label = QLabel("🔍 Code Query Assistant")
        self.title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))

        # Input Field
        self.query_input = QTextEdit(self)
        self.query_input.setPlaceholderText("Enter your programming-related question...")
        self.query_input.setText("Develop a query engine ai agent with llama_index")

        # Execute Button
        self.query_button = QPushButton("Run Query", self)
        self.query_button.clicked.connect(self.run_query)

        # Output Area
        self.output_display = QTextEdit(self)
        self.output_display.setReadOnly(True)

        # Add widgets to layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.query_input)
        layout.addWidget(self.query_button)
        layout.addWidget(self.output_display)

        self.setLayout(layout)

    def init_agents(self):
        """
        Initialize the RAG Agent with the necessary tools and models.
        """
        # Load code files with improved chunking
        python_documents = SimpleDirectoryReader("./",recursive=True, required_exts=[".py"]).load_data()
        splitter = CodeSplitter(language="python", max_chars=1000)
        nodes = splitter.get_nodes_from_documents(python_documents, show_progress=True)
        print(f"Extracted {len(nodes)} nodes from code files.")

        # Load other text files with improved chunking
        text_documents = SimpleDirectoryReader(input_dir="./", required_exts=[".txt",".md"], recursive=True).load_data()
        sentence_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)  # Use an appropriate splitter for text files
        #text_nodes = text_splitter.get_nodes_from_documents(text_documents)       
        text_nodes = sentence_splitter.get_nodes_from_documents(
            text_documents, show_progress=True
        )    

        all_nodes = nodes + text_nodes


        # Use HuggingFace embeddings BAAI/bge-small-en-v1.5
        # embedding_model = HuggingFaceEmbedding(
        #     model_name="all-MiniLM-L6-v2"  # A good general-purpose embedding model
        # )
        embedding_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"  # A good general-purpose embedding model
        )        

        llmGemma = Ollama(model="gemma3:1b", temperature=0.7, request_timeout=600)
        #llmGemma = Gemini(model="models/gemini-1.5-flash")        
        #llm = Ollama(model="llama3.2:latest", temperature=0.7, request_timeout=600)
        llm = Gemini(
            model="models/gemini-2.0-flash",
        ) # gemini-2.0-flash-thinking-exp-01-21
        # models/gemini-1.5-flash
        # models/gemini-2.0-flash
        # gemini-2.0-pro-exp-02-05

        # Create a VectorStore index for efficient retrieval
        index = VectorStoreIndex(all_nodes, embed_model=embedding_model)
        query_engine = index.as_query_engine(llm=llmGemma, similarity_top_k=5)  # Replace None with your LLM instance

        # Define RAG Agent Tool
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="QueryEngineTool",
            description="Retrieves relevant code snippets and best practices for software development queries.",
            return_direct=False,
        )

        # Create Agent Workflow
        self.query_engine_agent = AgentWorkflow.from_tools_or_functions(
            [query_engine_tool],
            llm=llm,  # Replace None with an LLM instance
            system_prompt="You are a specialized assistant that retrieves structured and relevant code knowledge.",
            verbose=True
        )


    def run_query(self):
        """Runs the query through the RAG Agent and displays results."""
        user_query = self.query_input.toPlainText().strip()
        if not user_query:
            self.output_display.setText("❌ Please enter a query!")
            return

        # Run query through RAG Agent


        start = time.time()
     
        response = asyncio.run(self.run_query_async(user_query)) # AgentOutput class object
        end = time.time()
        print(f"Time taken: {end - start}")



        self.output_display.setText(str(response))

    async def run_query_async(self, user_query):
        """Asynchronous function to run the query through the RAG Agent."""
        try:
            response = await self.query_engine_agent.run(user_msg=user_query)
        except Exception as e:
            response = f"An error occurred: {str(e)}"
        return response


if __name__ == "__main__":
   

    app = QApplication(sys.argv)
    window = CodeQueryApp()
    window.show()
    
    sys.exit(app.exec())

