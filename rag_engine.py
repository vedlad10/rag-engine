

!pip install langchain-core

from langchain_core.documents import Document

doc = Document(
    page_content="This is the main text content used to create RAG.",
    metadata={
        "source": "example.txt",
        "page": 1,
        "author": "Ved-Lad",
        "created_at": "2025-01-01"
    }
)

## Create a simple txt file
import os
os.makedirs("content/text_files",exist_ok=True)

sample_texts={
    "content/text_files/python_intro.txt":"""Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",

    "content/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems
    """

}

for filepath,content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8") as f:
        f.write(content)

print("✅ Sample text files created!")

!pip install langchain_community

### TextLoader
from langchain_community.document_loaders import TextLoader

loader=TextLoader("content/text_files/python_intro.txt",encoding="utf-8")
document=loader.load()
print(document)

### Directory Loader
from langchain_community.document_loaders import DirectoryLoader

## load all the text files from the directory
dir_loader=DirectoryLoader(
    "content/text_files",
    glob="**/*.txt", ## Pattern to match files
    loader_cls= TextLoader, ##loader class to use
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=False

)

documents=dir_loader.load()
documents

!pip install pymupdf

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

## load a single PDF file
loader=PyMuPDFLoader(
    "/content/attention.pdf" # Path to the PDF file
)

pdf_documents=loader.load()
pdf_documents

type(pdf_documents[0])

!pip install sentence-transformers

!pip install faiss-cpu

!pip install chromadb

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import FAISS
docs = documents + pdf_documents
vectorstore = FAISS.from_documents(docs, embeddings)
print(vectorstore)

!pip install langchain langchain_community
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(documents + pdf_documents)

query = "What is Python?"
docs_found = vectorstore.similarity_search(query, k=2)

docs_found

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# 1. Retrieve documents
docs = retriever.invoke("what is augmentation")

# 2. Build context
context = "\n\n".join(doc.page_content for doc in docs)

# 3. Prompt LLM
prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
What is used for?
"""

response = llm.invoke(prompt)
print(response)

!pip install langchain-groq

vectorstore.save_local("rag_faiss_index")

