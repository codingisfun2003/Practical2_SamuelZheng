import os
import fitz  
import numpy as np
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

os.environ["OLLAMA_NO_CUDA"] = "1"
VECTOR_DIM = 768
COLLECTION_NAME = "pdf_chunks"
chroma_client = chromadb.PersistentClient(path="./chroma_store")

def select_embedding_model():
    options = ["nomic-embed-text", "mpnet", "instructor-xl"]
    print("Select an embedding model:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        choice = input("Enter the number of your choice: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please enter a valid choice.")

def get_embedding(text, model, st_model=None, instructor_model=None):
    if model == "nomic-embed-text":
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
    elif model == "mpnet":
        return st_model.encode(text, normalize_embeddings=True).tolist()
    elif model == "instructor-xl":
        instruction = "Represent the document for retrieval:"
        return instructor_model.encode([[instruction, text]], normalize_embeddings=True)[0].tolist()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def process_pdfs(data_dir, embedding_model, st_model=None, instructor_model=None):
    if COLLECTION_NAME in chroma_client.list_collections():
        chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    doc_id = 0
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, embedding_model, st_model, instructor_model)
                    metadata = {
                        "file": file_name,
                        "page": str(page_num),
                        "chunk_index": str(chunk_index),
                    }
                    collection.add(
                        documents=[chunk],
                        metadatas=[metadata],
                        ids=[f"doc_{doc_id}"],
                        embeddings=[embedding],
                    )
                    doc_id += 1
            print(f" -----> Processed {file_name}")

def main():
    embedding_model = select_embedding_model()
    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedding_model == "mpnet" else None
    instructor_model = INSTRUCTOR("hkunlp/instructor-xl") if embedding_model == "instructor-xl" else None

    print(f"\nUsing embedding model: {embedding_model}")
    process_pdfs("./notes", embedding_model, st_model, instructor_model)
    print("\nDone processing and storing in Chroma.")

if __name__ == "__main__":
    main()