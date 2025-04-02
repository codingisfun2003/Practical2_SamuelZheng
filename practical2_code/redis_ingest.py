import os
import ollama
import redis
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

# Redis setup
redis_client = redis.Redis(host="localhost", port=6379, db=0)
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc"
DISTANCE_METRIC = "COSINE"
os.environ["OLLAMA_NO_CUDA"] = "1"

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
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    elif model == "mpnet":
        return st_model.encode(text, normalize_embeddings=True).tolist()
    elif model == "instructor-xl":
        instruction = "Represent the document for retrieval:"
        return instructor_model.encode([[instruction, text]], normalize_embeddings=True)[0].tolist()

def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA file TEXT page TEXT chunk TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

def store_embedding(file, page, chunk, embedding):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk[:20].replace(' ', '_')}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )
    print(f"Stored embedding for: {key}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def process_pdfs(data_dir, embedding_model, st_model=None, instructor_model=None):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            for page_num, text in extract_text_from_pdf(pdf_path):
                for chunk in split_text_into_chunks(text):
                    embedding = get_embedding(chunk, embedding_model, st_model, instructor_model)
                    store_embedding(file_name, str(page_num), chunk, embedding)
            print(f" -----> Processed {file_name}")

def main():
    embedding_model = select_embedding_model()
    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedding_model == "mpnet" else None
    instructor_model = INSTRUCTOR("hkunlp/instructor-xl") if embedding_model == "instructor-xl" else None

    print(f"Using embedding model: {embedding_model}")
    clear_redis_store()
    create_hnsw_index()
    process_pdfs("./notes", embedding_model, st_model, instructor_model)
    print("\nDone processing and storing in Redis.")

if __name__ == "__main__":
    main()