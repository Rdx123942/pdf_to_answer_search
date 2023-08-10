from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load PDF using PyPDF2
pdfreader = PdfReader("./budget_speech.pdf")

# Extract and concatenate text from PDF pages
raw_text = ''
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Split concatenated text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Initialize OpenAIEmbeddings
embedding = OpenAIEmbeddings()

# Create FAISS index for similarity search
document_search = FAISS.from_texts(texts, embedding)

# Load question answering chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Define query and perform similarity search
query = ": Inclusive Development"
docs = document_search.similarity_search(query)

# Run the question answering chain
chain_result = chain.run(input_documents=docs, question=query)

# Print the chain result
print(chain_result)
