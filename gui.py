# gui.py
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QMessageBox
)
from agents.query_engine_agent import RAGAgent

class AgentAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Agent App")
        self.agent = RAGAgent()

        # Initialize GUI components
        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Add Directory Button
        self.add_dir_button = QPushButton("Add Directory")
        self.add_dir_button.clicked.connect(self.add_directory)
        self.layout.addWidget(self.add_dir_button)

        # Initialize Agent Button
        self.init_agent_button = QPushButton("Initialize Agent")
        self.init_agent_button.clicked.connect(self.initialize_agent)
        self.layout.addWidget(self.init_agent_button)

        # Question Input
        self.question_label = QLabel("Your Question:")
        self.question_input = QLineEdit()
        self.layout.addWidget(self.question_label)
        self.layout.addWidget(self.question_input)

        # Ask Button
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask_question)
        self.layout.addWidget(self.ask_button)

        # Response Display
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.layout.addWidget(self.response_text)

    def add_directory(self):
        """
        Open a directory dialog and add the selected directory to the agent.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.agent.load_documents([directory])
            QMessageBox.information(self, "Success", f"Added directory: {directory}")

    def initialize_agent(self):
        """
        Initialize the RAG agent.
        """
        try:
            self.agent.init_models()
            self.agent.build_index()
            self.agent.init_agent()
            QMessageBox.information(self, "Success", "Agent initialized successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize agent: {str(e)}")

    def ask_question(self):
        """
        Ask a question to the RAG agent and display the response.
        """
        question = self.question_input.text()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return

        try:
            response = self.agent.query(question)
            self.response_text.setPlainText(str(response))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to query agent: {str(e)}")