from typing import List
from llama_index.core.node_parser import CodeSplitter
from llama_index.core import Document
from config.settings import config

class CustomCodeSplitter:
    def __init__(self,
                 language: str = None,
                 max_chars: int = None):
        self.language = language or config.splitter.code_language
        self.max_chars = max_chars or config.splitter.code_chunk_size
        
        self._splitter = CodeSplitter(
            language=self.language,
            max_chars=self.max_chars
        )

    def split_documents(self, documents: List[Document], show_progress: bool = True) -> List[Document]:
        """Split code documents into nodes using llama-index CodeSplitter.
        
        Args:
            documents: List of Document objects containing code
            show_progress: Whether to show progress bar during splitting
            
        Returns:
            List of Document nodes after splitting
        """
        return self._splitter.get_nodes_from_documents(
            documents,
            show_progress=show_progress
        )

    @staticmethod
    def is_code_file(file_path: str) -> bool:
        """Check if a file is a code file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file extension matches code extensions
        """
        return any(file_path.endswith(ext) for ext in config.splitter.code_extensions)