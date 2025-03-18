# gui.py
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QTextEdit, QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal
from agents.query_engine_agent import RAGAgent

class AgentWorker(QThread):
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
                self.agent.load_documents(self.args[0], progress_callback=self.progress.emit)
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

class AgentAppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Agent App")
        self.agent = RAGAgent()
        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

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

    def show_progress_bar(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def hide_progress_bar(self):
        self.progress_bar.setVisible(False)

    def add_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.show_progress_bar()
            self.worker = AgentWorker(self.agent, "load_documents", directory)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_load_complete)
            self.worker.error.connect(self.on_error)
            self.worker.start()

    def initialize_agent(self):
        self.show_progress_bar()
        self.worker = AgentWorker(self.agent, "init_agent")
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_init_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def ask_question(self):
        question = self.question_input.text()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return

        self.show_progress_bar()
        self.worker = AgentWorker(self.agent, "query", question)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_query_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_load_complete(self, _):
        self.hide_progress_bar()
        QMessageBox.information(self, "Success", "Documents loaded successfully.")

    def on_init_complete(self, _):
        self.hide_progress_bar()
        QMessageBox.information(self, "Success", "Agent initialized successfully.")

    def on_query_complete(self, response):
        self.hide_progress_bar()
        self.response_text.setPlainText(str(response))

    def on_error(self, error_message):
        self.hide_progress_bar()
        QMessageBox.critical(self, "Error", error_message)