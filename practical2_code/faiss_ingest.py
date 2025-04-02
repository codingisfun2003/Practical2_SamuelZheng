import os
import fitz
import numpy as np
import faiss
import json
import ollama
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

os.environ["OLLAMA_NO_CUDA"] = "1"
VECTOR_DIM = 768
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.json"

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

def select_option(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        choice = input("Enter the number of your choice: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please enter a valid choice.")

def get_embedding(text, model_name, st_model=None, instructor_model=None):
    if model_name == "nomic-embed-text":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    elif model_name == "mpnet":
        return st_model.encode(text, normalize_embeddings=True)
    elif model_name == "instructor-xl":
        instruction = "Represent the document for retrieval:"
        return instructor_model.encode([[instruction, text]], normalize_embeddings=True)[0]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

def split_text_into_chunks(text, chunk_size=None, overlap=None):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def process_pdfs(data_dir, embedding_model, st_model=None, instructor_model=None, chunk_size=300, overlap=50):
    index = faiss.IndexFlatIP(VECTOR_DIM)
    metadata = []
    doc_id = 0

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, embedding_model, st_model, instructor_model)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    index.add(np.array([embedding]))
                    metadata.append({
                        "id": doc_id,
                        "file": file_name,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "text": chunk
                    })
                    doc_id += 1
            print(f" -----> Processed {file_name}")

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

def main():
    embedding_model = select_embedding_model()
    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedding_model == "mpnet" else None
    instructor_model = INSTRUCTOR("hkunlp/instructor-xl") if embedding_model == "instructor-xl" else None

    chunk_size = select_option("Choose a chunk size:", [300, 500, 1000])
    overlap = select_option("Choose an overlap size:", [0, 50, 100])

    print(f"\nUsing embedding model: {embedding_model}")
    process_pdfs("./notes", embedding_model, st_model, instructor_model, chunk_size, overlap)
    print(f"\nDone processing and storing in FAISS.")

if __name__ == "__main__":
    main()