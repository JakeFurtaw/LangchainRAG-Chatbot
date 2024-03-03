from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import shutil

# path to the data
DATA_PATH = 'data'
CHROMA_PATH = 'chroma'

def main():
    documents = load_docs()
    chunks = split_pages(documents)
    save_to_db(chunks)


# load the .txt files
def load_docs():
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


# split the documents into chunks of text
def split_pages(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


# save files to database
def save_to_db(chunks):
    # clear db if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # create new db from current docs
    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH
    )
    # print(chunks[10:13])
    db.persist()


if __name__ == '__main__':
    main()
