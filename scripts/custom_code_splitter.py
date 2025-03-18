import os
import ast
from typing import List, Dict, Any, Optional
from collections import defaultdict
from typing import List, Union

class ASTParentTracker(ast.NodeVisitor):
    """Tracks parent nodes to associate methods with their respective classes."""
    
    def __init__(self):
        self.class_methods = defaultdict(list)
        self.functions = []
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Tracks class definitions and their methods."""
        class_code = ast.get_source_segment(self.source_code, node)
        if class_code:
            self.class_methods[node.name].append(class_code)

        # Visit children
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_code = ast.get_source_segment(self.source_code, child)
                if method_code:
                    self.class_methods[node.name].append(method_code)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Tracks standalone functions."""
        function_code = ast.get_source_segment(self.source_code, node)
        if function_code:
            self.functions.append(function_code)

# custom chunking method
def chunk_python_file(filename: str, chunk_size: int = 20) -> List[str]:
    """
    Extracts functions and classes from a Python file and chunks them into
    groups of a specified minimum size.
    
    Args:
        filename (str): Path to the Python file.
        chunk_size (int): Minimum number of lines per chunk.
        
    Returns:
        List[str]: A list of chunked Python code segments.
    """
    if not os.path.isfile(filename):
        print(f"Error: File not found: {filename}")
        return []
    
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []

    if not content.strip():
        print(f"File {filename} is empty.")
        return []

    # Parse the content into an AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Error: Syntax error in file {filename}. {e}")
        return []

    # Track class and function definitions
    tracker = ASTParentTracker()
    tracker.source_code = content
    tracker.visit(tree)

    # Chunking logic
    chunks = []
    current_chunk = []
    current_chunk_line_count = 0

    # Process class methods together
    for class_name, methods in tracker.class_methods.items():
        total_class_lines = sum(method.count('\n') + 1 for method in methods)
        if current_chunk_line_count + total_class_lines > chunk_size:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = methods
            current_chunk_line_count = total_class_lines
        else:
            current_chunk.extend(methods)
            current_chunk_line_count += total_class_lines

    # Process standalone functions
    for function in tracker.functions:
        function_lines = function.count('\n') + 1
        if current_chunk_line_count + function_lines > chunk_size:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [function]
            current_chunk_line_count = function_lines
        else:
            current_chunk.append(function)
            current_chunk_line_count += function_lines

    # Add any remaining chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


if __name__ == '__main__':
    filename = 'E:\\05_Repos\\03_Sesame_CSM_Voice_Cloning\\generator.py'
    chunks = chunk_python_file(filename)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")        