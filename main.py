# main.py
import sys
from PyQt6.QtWidgets import QApplication
from gui import AgentAppGUI

def main():
    app = QApplication(sys.argv)
    gui = AgentAppGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()