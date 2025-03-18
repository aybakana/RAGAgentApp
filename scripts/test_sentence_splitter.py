from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Load other text files with improved chunking
text_documents = SimpleDirectoryReader(input_dir="./", required_exts=[".txt",".md"], recursive=True).load_data()
sentence_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)  # Use an appropriate splitter for text files
#text_nodes = text_splitter.get_nodes_from_documents(text_documents)       
text_nodes = sentence_splitter.get_nodes_from_documents(
    text_documents, show_progress=False
)

for node in text_nodes:
    print(f"code node: {node.embedding} \n\n{node.metadata} ")
    print(f"node Content: \n{node.get_content()}")
    print("-------------------------------------------------\n\n")
