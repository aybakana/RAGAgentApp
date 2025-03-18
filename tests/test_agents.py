import unittest
import tempfile
import os
from pathlib import Path
from agents.base_agent import BaseAgent
from agents.rag_agent import RAGAgent

class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing abstract class."""
    
    def init_models(self): pass
    def load_documents(self, directories, extensions=None): pass
    def build_index(self): pass
    def init_agent(self): pass
    def query(self, question, **kwargs): return "mock response"
    def get_stats(self): return {"mock": True}

class TestBaseAgent(unittest.TestCase):
    def test_base_agent_interface(self):
        """Test that BaseAgent interface can be implemented."""
        try:
            mock_agent = MockAgent()
            self.assertIsInstance(mock_agent, BaseAgent)
        except TypeError:
            self.fail("Failed to implement BaseAgent interface")

class TestRAGAgent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.agent = RAGAgent()
        
        # Create test files
        self.code_file = Path(self.test_dir) / "test.py"
        self.text_file = Path(self.test_dir) / "test.txt"
        
        self.code_file.write_text('''
def test_function():
    """Test function docstring."""
    return "Hello World"
''')
        
        self.text_file.write_text('''
This is a test document.
It contains multiple sentences.
Each sentence should be processed correctly.
''')

    def test_init_models(self):
        """Test model initialization."""
        self.agent.init_models()
        self.assertIsNotNone(self.agent.embedding_model)
        self.assertIsNotNone(self.agent.llm)

    def test_load_documents(self):
        """Test document loading and processing."""
        # Load documents
        self.agent.load_documents(self.test_dir)
        self.assertTrue(len(self.agent.nodes) > 0)

    def test_build_index_without_documents(self):
        """Test that build_index fails without documents."""
        self.agent.init_models()
        with self.assertRaises(ValueError):
            self.agent.build_index()

    def test_build_index_without_models(self):
        """Test that build_index fails without initialized models."""
        self.agent.load_documents(self.test_dir)
        with self.assertRaises(ValueError):
            self.agent.build_index()

    def test_init_agent_without_query_engine(self):
        """Test that init_agent fails without query engine."""
        with self.assertRaises(ValueError):
            self.agent.init_agent()

    def test_query_without_initialization(self):
        """Test that query fails without initialization."""
        with self.assertRaises(ValueError):
            self.agent.query("test question")

    def test_get_stats(self):
        """Test statistics reporting."""
        stats = self.agent.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_nodes", stats)
        self.assertIn("models_initialized", stats)
        self.assertIn("agent_initialized", stats)

    def test_full_workflow(self):
        """Test complete agent workflow."""
        # Initialize models
        self.agent.init_models()
        self.assertTrue(self.agent.get_stats()["models_initialized"])

        # Load documents
        self.agent.load_documents(self.test_dir)
        self.assertTrue(self.agent.get_stats()["total_nodes"] > 0)

        # Build index
        self.agent.build_index()
        self.assertTrue(self.agent.get_stats()["index_initialized"])

        # Initialize agent
        self.agent.init_agent()
        self.assertTrue(self.agent.get_stats()["agent_initialized"])

    def tearDown(self):
        # Clean up test files
        self.code_file.unlink()
        self.text_file.unlink()
        os.rmdir(self.test_dir)