import unittest
from pathlib import Path
from llama_index.core import Document
from processors.code_splitter import CustomCodeSplitter
from processors.text_splitter import CustomTextSplitter


class TestDocumentProcessors(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create test files
        self.code_file = self.test_data_dir / "test.py"
        self.text_file = self.test_data_dir / "test.txt"
        
        # Sample code content
        code_content = '''
def test_function():
    """Test function docstring."""
    print("Hello world")
    
class TestClass:
    def __init__(self):
        self.value = 42
        
    def test_method(self):
        """Test method docstring."""
        return self.value
'''
        
        # Sample text content
        text_content = '''
This is a test document.
It has multiple sentences.
Each sentence should be processed correctly.
The splitter should handle this text appropriately.
'''
        
        # Write test files
        self.code_file.write_text(code_content)
        self.text_file.write_text(text_content)
        
        # Initialize splitters
        self.code_splitter = CustomCodeSplitter()
        self.text_splitter = CustomTextSplitter()

    def test_code_splitter_initialization(self):
        """Test code splitter initialization with default parameters."""
        self.assertEqual(self.code_splitter.max_chars, 1000)
        self.assertEqual(self.code_splitter.language, "python")

    def test_text_splitter_initialization(self):
        """Test text splitter initialization with default parameters."""
        self.assertEqual(self.text_splitter.chunk_size, 1024)
        self.assertEqual(self.text_splitter.chunk_overlap, 20)

    def test_code_file_detection(self):
        """Test code file extension detection."""
        self.assertTrue(CustomCodeSplitter.is_code_file("test.py"))
        self.assertTrue(CustomCodeSplitter.is_code_file("test.js"))
        self.assertFalse(CustomCodeSplitter.is_code_file("test.txt"))

    def test_text_file_detection(self):
        """Test text file extension detection."""
        self.assertTrue(CustomTextSplitter.is_text_file("test.txt"))
        self.assertTrue(CustomTextSplitter.is_text_file("test.md"))
        self.assertFalse(CustomTextSplitter.is_text_file("test.py"))

    def test_code_document_splitting(self):
        """Test splitting code documents."""
        doc = Document(text=self.code_file.read_text())
        nodes = self.code_splitter.split_documents([doc])
        self.assertTrue(len(nodes) > 0)
        for node in nodes:
            self.assertIsNotNone(node.get_content())

    def test_text_document_splitting(self):
        """Test splitting text documents."""
        doc = Document(text=self.text_file.read_text())
        nodes = self.text_splitter.split_documents([doc])
        self.assertTrue(len(nodes) > 0)
        for node in nodes:
            self.assertIsNotNone(node.get_content())

    def tearDown(self):
        # Clean up test files
        self.code_file.unlink()
        self.text_file.unlink()
        self.test_data_dir.rmdir()


if __name__ == '__main__':
    unittest.main()