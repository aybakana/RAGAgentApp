from typing import List
from llama_index.core.node_parser import SentenceSplitter, NodeParser
from llama_index.core import Document
from config.settings import config

class CustomTextSplitter:
    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.splitter.text_chunk_size
        self.chunk_overlap = chunk_overlap or config.splitter.text_chunk_overlap
        
        self._splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def split_documents(self, documents: List[Document], show_progress: bool = True) -> List[Document]:
        """Split text documents into nodes using llama-index SentenceSplitter.
        
        Args:
            documents: List of Document objects containing text
            show_progress: Whether to show progress bar during splitting
            
        Returns:
            List of Document nodes after splitting
        """
        return self._splitter.get_nodes_from_documents(
            documents,
            show_progress=show_progress
        )

    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """Check if a file is a text file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file extension matches text extensions
        """
        return any(file_path.endswith(ext) for ext in config.splitter.text_extensions)