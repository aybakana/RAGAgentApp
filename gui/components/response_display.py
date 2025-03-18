from PyQt6.QtWidgets import QTextEdit, QWidget, QVBoxLayout
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
        
        # Create text display area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Response will appear here...")
        
        # Add to layout
        layout.addWidget(self.text_display)

    def clear(self):
        """Clear the display and buffer."""
        self.text_display.clear()
        self._response_buffer.clear()
        self._display_timer.stop()

    def append_text(self, text: str):
        """Add text to the response buffer."""
        self._response_buffer.append(text)
        if not self._display_timer.isActive():
            self._display_timer.start()

    def set_text(self, text: str):
        """Set the complete response text."""
        self.clear()
        self.text_display.setPlainText(text)
        self.response_complete.emit()

    def _update_display(self):
        """Update the display with buffered content."""
        if self._response_buffer:
            current_text = self.text_display.toPlainText()
            next_chunk = self._response_buffer.pop(0)
            self.text_display.setPlainText(current_text + next_chunk)
            
            # Scroll to bottom
            scrollbar = self.text_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # If buffer is empty, stop timer and emit completion
            if not self._response_buffer:
                self._display_timer.stop()
                self.response_complete.emit()

    def get_text(self) -> str:
        """Get the current display text."""
        return self.text_display.toPlainText()