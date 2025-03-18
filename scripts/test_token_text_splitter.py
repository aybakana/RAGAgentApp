from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.callbacks.base import CallbackManager

# Initialize the TokenTextSplitter with default parameters
token_text_splitter = TokenTextSplitter(
    chunk_size=100,  # Adjust the chunk size as needed
    chunk_overlap=10,  # Adjust the chunk overlap as needed
    separator=" ",  # Default separator for splitting into words
    backup_separators=["\n"],  # Additional separators for splitting
    keep_whitespaces=False,  # Whether to keep leading/trailing whitespaces in the chunk
    callback_manager=CallbackManager([])  # Optional callback manager
)

# read a file
with open("test/README.md", "r") as file:
    text = file.read()

# Split the text into chunks
chunks = token_text_splitter.split_text(text)

# Print the resulting chunks
for chunk in chunks:
    print(chunk)
    print("-------------------------------------------------\n\n")