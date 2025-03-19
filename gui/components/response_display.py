from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

class ResponseDisplay(QWidget):
    """A widget for displaying streaming responses from the RAG agent."""
    
    # Signal emitted when response is fully displayed
    response_complete = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self._response_buffer = []
        self._display_timer = QTimer()
        self._display_timer.timeout.connect(self._update_display)
        self._display_timer.setInterval(50)  # 50ms update interval

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Response tab
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.tab_widget.addTab(self.response_text, "Response")
        
        # Source nodes tab
        self.source_nodes_table = QTableWidget()
        self.source_nodes_table.setColumnCount(3)
        self.source_nodes_table.setHorizontalHeaderLabels(["File Path", "Score", "Text"])
        self.source_nodes_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tab_widget.addTab(self.source_nodes_table, "Source Nodes")
        
        # Debug info tab
        self.debug_info_text = QTextEdit()
        self.debug_info_text.setReadOnly(True)
        self.tab_widget.addTab(self.debug_info_text, "Debug Info")
        
        layout.addWidget(self.tab_widget)

    def clear(self):
        """Clear all displays"""
        self.response_text.clear()
        self.source_nodes_table.setRowCount(0)
        self.debug_info_text.clear()
        self._response_buffer.clear()
        self._display_timer.stop()

    def append_text(self, text: str):
        """Add text to the response buffer."""
        self._response_buffer.append(text)
        if not self._display_timer.isActive():
            self._display_timer.start()

    def set_text(self, response_data):
        """
        Set the response text and debug information
        Args:
            response_data (dict): Dictionary containing response text and debug info
        """
        self.clear()
        if isinstance(response_data, dict):
            # Set main response text
            self.response_text.setText(response_data['response_text'])
            
            # Update source nodes table
            source_nodes = response_data['debug_info']['source_nodes']
            self.source_nodes_table.setRowCount(len(source_nodes))
            
            for row, node in enumerate(source_nodes):
                self.source_nodes_table.setItem(row, 0, QTableWidgetItem(str(node['file_path'])))
                self.source_nodes_table.setItem(row, 1, QTableWidgetItem(str(node['score'])))
                self.source_nodes_table.setItem(row, 2, QTableWidgetItem(str(node['text'])))
            
            # Update debug info
            debug_text = "Response Metadata:\n"
            debug_text += str(response_data['debug_info']['response_metadata'])
            self.debug_info_text.setText(debug_text)
        else:
            # Handle legacy string responses
            self.response_text.setText(str(response_data))
            self.source_nodes_table.setRowCount(0)
            self.debug_info_text.clear()
        self.response_complete.emit()

    def _update_display(self):
        """Update the display with buffered content."""
        if self._response_buffer:
            current_text = self.response_text.toPlainText()
            next_chunk = self._response_buffer.pop(0)
            self.response_text.setPlainText(current_text + next_chunk)
            
            # Scroll to bottom
            scrollbar = self.response_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # If buffer is empty, stop timer and emit completion
            if not self._response_buffer:
                self._display_timer.stop()
                self.response_complete.emit()

    def get_text(self) -> str:
        """Get the current display text."""
        return self.response_text.toPlainText()