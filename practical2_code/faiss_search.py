import faiss
import numpy as np
import json
import os
import time
import ollama
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

os.environ["OLLAMA_NO_CUDA"] = "1"
VECTOR_DIM = 768
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.json"

def select_option(prompt, options):
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    while True:
        choice = input("Enter the number of your choice: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
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

def search_embeddings(query, embedding_model, st_model=None, instructor_model=None, top_k=3):
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    query_vector = get_embedding(query, embedding_model, st_model, instructor_model)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    D, I = index.search(np.array([query_vector]), top_k)
    results = []
    for i in range(top_k):
        idx = I[0][i]
        if idx < len(metadata):
            result = metadata[idx]
            result["similarity"] = float(D[0][i])
            results.append(result)
    return results

def generate_rag_response(query, context_results, llm_model):
    context_str = "\n".join(
        [f"From {res['file']} (page {res['page']}, chunk {res['chunk_index']}):\n{res['text']}"
         for res in context_results]
    )
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def interactive_search():
    embedding_model = select_option("Select an embedding model:", ["nomic-embed-text", "mpnet", "instructor-xl"])
    llm_model = select_option("Select an LLM:", ["mistral", "llama3"])

    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedding_model == "mpnet" else None
    instructor_model = INSTRUCTOR("hkunlp/instructor-xl") if embedding_model == "instructor-xl" else None

    print(f"\n🔍 FAISS Search using '{embedding_model}' + '{llm_model}'")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        start_time = time.time() 
        context = search_embeddings(query, embedding_model, st_model, instructor_model)
        response = generate_rag_response(query, context, llm_model)
        end_time = time.time()

        print(response)
        print(f"{end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    interactive_search()