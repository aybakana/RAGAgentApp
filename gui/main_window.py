from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QMessageBox, QProgressBar,
    QLineEdit, QStatusBar, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from agents.query_engine_agent import RAGAgent
from .components.response_display import ResponseDisplay
from config.settings import config

class AgentWorker(QThread):
    """Worker thread for running agent operations."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, agent, task, *args):
        super().__init__()
        self.agent = agent
        self.task = task
        self.args = args

    def run(self):
        try:
            if self.task == "load_documents":
                self.agent.load_documents(self.args[0])
                self.progress.emit(100)
                self.finished.emit(None)
            elif self.task == "init_agent":
                self.agent.init_models()
                self.progress.emit(33)
                self.agent.build_index()
                self.progress.emit(66)
                self.agent.init_agent()
                self.progress.emit(100)
                self.finished.emit(None)
            elif self.task == "query":
                response = self.agent.query(self.args[0])
                self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.agent = RAGAgent()
        self.init_ui()
        self.setWindowTitle("RAG Agent Assistant")
        self.resize(800, 600)

    def init_ui(self):
        """Initialize the user interface."""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("🤖 RAG Agent Assistant")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Directory Selection
        dir_layout = QHBoxLayout()
        self.dir_button = QPushButton("Add Directory")
        self.dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.dir_button)
        
        self.init_button = QPushButton("Initialize Agent")
        self.init_button.clicked.connect(self.initialize_agent)
        dir_layout.addWidget(self.init_button)
        layout.addLayout(dir_layout)

        # Query Input and Controls
        query_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your question here...")
        query_layout.addWidget(self.query_input)
        
        # Model Selection
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "models/gemini-2.0-flash", 
            "models/gemini-2.0-flash-thinking-exp-01-21",
            "models/gemini-1.5-flash",
            "models/gemini-2.0-pro-exp-02-05",
            "models/gemini-2.0-flash-lite",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-flash-8b"])
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        query_layout.addWidget(self.model_selector)
        
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask_question)
        query_layout.addWidget(self.ask_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_query)
        self.cancel_button.setEnabled(False)
        query_layout.addWidget(self.cancel_button)
        
        layout.addLayout(query_layout)

        # Response Display
        self.response_display = ResponseDisplay()
        layout.addWidget(self.response_display)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Set initial button states
        self.init_button.setEnabled(False)
        self.ask_button.setEnabled(False)

    def show_progress_bar(self):
        """Show and reset the progress bar."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def hide_progress_bar(self):
        """Hide the progress bar."""
        self.progress_bar.setVisible(False)

    def update_progress(self, value):
        """Update progress bar value."""
        self.progress_bar.setValue(value)

    def select_directory(self):
        """Handle directory selection."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Index",
            options=QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.show_progress_bar()
            self.status_bar.showMessage("Loading documents...")
            self.dir_button.setEnabled(False)
            
            # Start worker thread
            self.worker = AgentWorker(self.agent, "load_documents", directory)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_documents_loaded)
            self.worker.error.connect(self.on_error)
            self.worker.start()

    def initialize_agent(self):
        """Initialize the agent with loaded documents."""
        self.show_progress_bar()
        self.status_bar.showMessage("Initializing agent...")
        self.init_button.setEnabled(False)
        
        # Start worker thread
        self.worker = AgentWorker(self.agent, "init_agent")
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_agent_initialized)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def ask_question(self):
        """Process a user question."""
        question = self.query_input.text().strip()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return

        self.show_progress_bar()
        self.status_bar.showMessage("Processing question...")
        self.ask_button.setEnabled(False)
        self.query_input.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # Clear previous response
        self.response_display.clear()
        
        # Start worker thread
        self.worker = AgentWorker(self.agent, "query", question)
        self.worker.finished.connect(self.on_response_received)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def cancel_query(self):
        """Cancel the current query operation."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.on_query_cancelled()

    def on_query_cancelled(self):
        """Handle query cancellation."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Query cancelled")
        self.ask_button.setEnabled(True)
        self.query_input.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.response_display.set_text("Query cancelled by user")

    def on_model_changed(self, model_name):
        """Handle LLM model change."""
        if self.agent.initialized:
            self.show_progress_bar()
            self.status_bar.showMessage(f"Reinitializing agent with {model_name}...")
            # Update config
            config.geminiLLM.model_name = model_name
            # Reinitialize agent
            self.initialize_agent()

    def on_documents_loaded(self, _):
        """Handle completion of document loading."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Documents loaded successfully")
        self.dir_button.setEnabled(True)
        self.init_button.setEnabled(True)
        
        # Show stats
        stats = self.agent.get_stats()
        QMessageBox.information(
            self,
            "Documents Loaded",
            f"Successfully loaded {stats['total_nodes']} document nodes."
        )

    def on_agent_initialized(self, _):
        """Handle completion of agent initialization."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Agent initialized successfully")
        self.ask_button.setEnabled(True)
        self.query_input.setEnabled(True)
        
        QMessageBox.information(
            self,
            "Agent Initialized",
            "The agent has been initialized and is ready for questions."
        )

    def on_response_received(self, response):
        """Handle agent response."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Response received")
        self.ask_button.setEnabled(True)
        self.query_input.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # Display response
        self.response_display.set_text(response)

    def on_error(self, error_message):
        """Handle errors from worker threads."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Error occurred")
        self.dir_button.setEnabled(True)
        self.init_button.setEnabled(True)
        self.ask_button.setEnabled(True)
        self.query_input.setEnabled(True)
        
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred: {error_message}"
        )