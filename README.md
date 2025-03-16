# Agents App

## Description

The Agents App is a Python-based GUI application that allows users to interact with an intelligent Agent. By loading and indexing documents and code files, users can ask questions and receive context-aware responses in real time using a Retrieval-Augmented Generation (RAG) approach.

## Features

- 🤖 Intelligent RAG Agent for context-aware responses
- 📁 Support for multiple document types (.txt, .md, .py)
- 🔍 Advanced document indexing and retrieval
- 💻 User-friendly GUI interface
- 🔄 Real-time response streaming
- ⚙️ Configurable embedding and language models
- 🎯 Precise code and text splitting capabilities

## Requirements

### Functional Requirements
Agent Management:

The application shall allow users to load and initialize different types of agents (e.g., RAGAgent).
Currently, only the RAGAgent is available, but the system shall be extensible to support additional agent types in the future.

Document and Code Loading:

The RAGAgent shall be able to load documents (e.g., .txt, .md) and Python code files (e.g., .py) from specified directories.
The application shall support recursive directory traversal to load files from nested folders.
The application shall allow users to add multiple directories dynamically through the GUI.

Vector Store Indexing:

The RAGAgent shall process loaded documents and code files into nodes using appropriate splitters:

Code files shall be split using a CodeSplitter with configurable parameters (e.g., max_chars).
Text files shall be split using a SentenceSplitter with configurable parameters (e.g., chunk_size, chunk_overlap).
The RAGAgent shall build a VectorStoreIndex from the processed nodes for efficient retrieval.

Query Handling:

The RAGAgent shall allow users to ask questions and retrieve context-aware responses using the indexed data.
The application shall display the agent's response in the GUI.

Embedding and LLM Configuration:

The RAGAgent shall use a configurable embedding model (e.g., BAAI/bge-small-en-v1.5) for generating embeddings.
The RAGAgent shall use a configurable LLM (e.g., Ollama, Gemini) for generating responses.

Error Handling:

The application shall handle errors gracefully (e.g., invalid directories, missing files, initialization failures) and display appropriate error messages to the user.


### Technical Requirements
Dependencies:

The application shall use the following Python libraries:
llama-index for document processing, indexing, and retrieval.
PyQt6 for the GUI.
HuggingFaceEmbedding for generating embeddings.
Ollama or Gemini for the LLM.

Code Structure:
The application shall follow a modular design with separate modules for:
GUI implementation (gui.py).
Main application entry point (main.py).

Configuration:

The application shall allow users to configure the following parameters:
Embedding model (e.g., BAAI/bge-small-en-v1.5).
LLM model (e.g., Ollama, Gemini).
Splitting parameters for code and text files.


### User Interaction Requirements
GUI Components:
The GUI shall include the following components:
A button to add directories.
A button to initialize the agent.
A text input field for user questions.
A button to submit questions.
A text area to display agent responses.
A status bar or progress indicator for long-running operations.

Dynamic Updates:
The GUI shall dynamically update to reflect the state of the application (e.g., loaded directories, initialized agents).
Streaming of response shall be available on the GUI.

Feedback:
The application shall provide feedback to the user for all actions (e.g., success messages, error messages).

## Modules

### Core Components
- `agents/`: Contains agent implementations (RAGAgent)
- `gui/`: PyQt6-based user interface components
- `utils/`: Helper functions and utilities
- `config/`: Configuration handlers
- `processors/`: Document and code processors

## File Structure
```
agents_app/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   └── rag_agent.py
├── gui/
│   ├── __init__.py
│   ├── main_window.py
│   └── components/
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── config/
│   └── settings.py
├── processors/
│   ├── __init__.py
│   ├── code_splitter.py
│   └── text_splitter.py
└── main.py
```

## UML Diagram in PlantUML Format
[Add UML diagram here]

## Future Improvements

### Short-term Improvements
1. Add support for more document formats (PDF, DOCX)
2. Implement agent configuration persistence
3. Add progress bars for long-running operations
4. Create detailed logging system
5. Add unit tests and integration tests

### Medium-term Improvements
1. Support for multiple concurrent agents
2. Document comparison capabilities
3. Export/Import functionality for indexed data
4. Custom splitting rules configuration
5. Memory management for large document sets

### Long-term Improvements
1. Implement collaborative features
2. Add support for remote document repositories
3. Create plugin system for extensibility
4. Implement advanced caching mechanisms
5. Add visualization tools for document relationships

## Contributing
[Add contribution guidelines]

## License
[Add license information]

