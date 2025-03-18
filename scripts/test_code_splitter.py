from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter


documents = SimpleDirectoryReader("./").load_data()
splitter = CodeSplitter(language="python", max_chars=1000)
nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(f"code node: {node.embedding} \n\n{node.metadata} ")
    print(f"node Content: {node.get_content()}")
    print("-------------------------------------------------\n\n")