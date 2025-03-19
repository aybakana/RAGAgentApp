from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QMessageBox, QProgressBar,
    QLineEdit, QStatusBar, QComboBox, QListWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from agents.rag_agent import RAGAgent
from .components.response_display import ResponseDisplay
from config.settings import config
import json
import os

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
            if self.task == "load_directory":
                self.progress.emit(10)
                nodes = self.agent.load_directory(self.args[0])
                self.agent.build_index_and_query_engine(self.args[0], nodes)
                self.progress.emit(100)
                self.finished.emit(None)
            elif self.task == "init_agent":
                self.progress.emit(20)
                self.agent.init_agent()
                self.progress.emit(100)
                self.finished.emit(None)
            elif self.task == "query":
                self.progress.emit(10)
                response = self.agent.query(self.args[0])
                self.progress.emit(100)
                self.finished.emit(response)
            elif self.task == "query_engine_query":
                self.progress.emit(10)
                response = self.agent.query_engine_query(self.args[0])
                self.progress.emit(100)
                self.finished.emit(response)
            elif self.task == "save_all_indexes":
                self.progress.emit(10)
                self.agent.save_all_indexes(self.args[0])
                self.progress.emit(100)
                self.finished.emit(None)
            elif self.task == "load_all_indexes":
                self.progress.emit(10)
                self.agent.load_all_indexes(self.args[0])
                self.progress.emit(100)
                self.finished.emit(None)
            elif self.task == "remove_directory":
                self.progress.emit(10)
                self.agent.remove_directory(self.args[0])
                self.progress.emit(100)
                self.finished.emit(None)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.agent = RAGAgent()
        self.directories = set()  # Track indexed directories
        self.storage_path = None  # Last used storage path for indexes
        self.load_directories_state()  # Load saved state
        self.init_ui()
        self.setWindowTitle("RAG Agent Assistant")
        self.resize(800, 600)

    def load_directories_state(self):
        """Load saved directories and storage path from state file"""
        try:
            with open('directories_state.json', 'r') as f:
                state = json.loads(f.read())
                self.directories = set(state.get('directories', []))
                self.storage_path = state.get('storage_path', None)
                # Try to load indexes if storage path exists
                if self.storage_path and os.path.exists(self.storage_path):
                    self.agent.load_all_indexes(self.storage_path, self.directories)
        except FileNotFoundError:
            pass

    def save_directories_state(self):
        """Save current directories and storage path to state file"""
        state = {
            'directories': list(self.directories),
            'storage_path': self.storage_path
        }
        with open('directories_state.json', 'w') as f:
            json.dump(state, f)

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

        # Directory Management Section
        dir_section = QVBoxLayout()
        dir_header = QHBoxLayout()
        
        # Directory buttons
        self.dir_button = QPushButton("Add Directory")
        self.dir_button.clicked.connect(self.select_directory)
        dir_header.addWidget(self.dir_button)
        
        self.remove_dir_button = QPushButton("Remove Directory")
        self.remove_dir_button.clicked.connect(self.remove_directory)
        dir_header.addWidget(self.remove_dir_button)
        
        # Index management buttons
        self.save_indexes_button = QPushButton("Save Indexes")
        self.save_indexes_button.clicked.connect(self.save_indexes)
        dir_header.addWidget(self.save_indexes_button)
        
        self.load_indexes_button = QPushButton("Load Indexes")
        self.load_indexes_button.clicked.connect(self.load_indexes)
        dir_header.addWidget(self.load_indexes_button)
        
        dir_section.addLayout(dir_header)
        
        # Directory list
        self.dir_list = QListWidget()
        self.dir_list.setMaximumHeight(100)
        self.update_directory_list()
        dir_section.addWidget(self.dir_list)
        
        layout.addLayout(dir_section)
        
        # Initialize Agent button moved after directory section
        self.init_agent_button = QPushButton("Initialize Agent")
        self.init_agent_button.clicked.connect(self.initialize_agent)
        layout.addWidget(self.init_agent_button)

        # Query Section
        query_section = QVBoxLayout()
        
        # Query type selection
        query_type_layout = QHBoxLayout()
        self.query_type_label = QLabel("Query Type:")
        self.query_type_selector = QComboBox()
        self.query_type_selector.addItems(["Agent Query", "Direct Query Engine"])
        query_type_layout.addWidget(self.query_type_label)
        query_type_layout.addWidget(self.query_type_selector)
        query_section.addLayout(query_type_layout)
        
        # Query input and controls
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
        
        query_section.addLayout(query_layout)
        layout.addLayout(query_section)

        # Response Display
        self.response_display = ResponseDisplay()
        layout.addWidget(self.response_display)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Set initial button states
        self.init_agent_button.setEnabled(False)
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
            self.directories.add(directory)
            self.show_progress_bar()
            self.status_bar.showMessage("Loading documents...")
            self.dir_button.setEnabled(False)
            
            # Start worker thread
            self.worker = AgentWorker(self.agent, "load_directory", directory)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_directory_loaded)
            self.worker.error.connect(self.on_error)
            self.worker.start()

    def initialize_agent(self):
        """Initialize the agent with loaded documents."""
        self.show_progress_bar()
        self.status_bar.showMessage("Initializing agent...")
        self.init_agent_button.setEnabled(False)
        
        # Start worker thread
        self.worker = AgentWorker(self.agent, "init_agent")
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_agent_initialized)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def ask_question(self):
        """Process a user question based on selected query type"""
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
        
        # Start worker thread based on query type
        query_type = self.query_type_selector.currentText()
        if query_type == "Direct Query Engine":
            self.worker = AgentWorker(self.agent, "query_engine_query", question)
        else:
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

    def on_directory_loaded(self, _):
        """Handle completion of document loading."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Documents loaded successfully")
        self.dir_button.setEnabled(True)
        self.init_agent_button.setEnabled(True)   # enable the init agent button
        
        
        # Show stats
        stats = self.agent.get_stats()
        QMessageBox.information(
            self,
            "Documents Loaded",
            f"Successfully loaded {stats['total_nodes']} document nodes."
        )
        self.update_directory_list()
        self.save_directories_state()

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
        self.init_agent_button.setEnabled(True)
        self.ask_button.setEnabled(True)
        self.query_input.setEnabled(True)
        
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred: {error_message}"
        )

    def update_directory_list(self):
        """Update the directory list widget with current directories"""
        self.dir_list.clear()
        for directory in sorted(self.directories):
            self.dir_list.addItem(directory)

    def remove_directory(self):
        """Remove selected directory from the index"""
        current_item = self.dir_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "Selection Error", "Please select a directory to remove.")
            return
        
        directory = current_item.text()
        
        # Start worker thread
        self.worker = AgentWorker(self.agent, "remove_directory", directory)
        self.worker.finished.connect(self.on_directory_removed)
        self.worker.error.connect(self.on_error)
        self.worker.start()

        self.directories.remove(directory)
        self.show_progress_bar()
        self.status_bar.showMessage("Removing directory...")        

    def on_directory_removed(self, _):
        """Handle completion of directory removal."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Directory removed successfully")
        self.update_directory_list()
        self.save_directories_state()

    def save_indexes(self):
        """Save all indexes to a selected directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Indexes",
            options=QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.storage_path = directory
            self.show_progress_bar()
            self.status_bar.showMessage("Saving indexes...")
            
            # Start worker thread
            self.worker = AgentWorker(self.agent, "save_all_indexes", directory)
            self.worker.finished.connect(self.on_indexes_saved)
            self.worker.error.connect(self.on_error)
            self.worker.start()

    def on_indexes_saved(self, _):
        """Handle completion of index saving."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Indexes saved successfully")
        self.save_directories_state()
        QMessageBox.information(self, "Success", "Indexes saved successfully")

    def load_indexes(self):
        """Load all indexes from a selected directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Load Indexes",
            options=QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.storage_path = directory
            self.show_progress_bar()
            self.status_bar.showMessage("Loading indexes...")
            
            # Start worker thread
            self.worker = AgentWorker(self.agent, "load_all_indexes", directory)
            self.worker.finished.connect(self.on_indexes_loaded)
            self.worker.error.connect(self.on_error)
            self.worker.start()

    def on_indexes_loaded(self, _):
        """Handle completion of index loading."""
        self.hide_progress_bar()
        self.status_bar.showMessage("Indexes loaded successfully")
        self.save_directories_state()
        QMessageBox.information(self, "Success", "Indexes loaded successfully")