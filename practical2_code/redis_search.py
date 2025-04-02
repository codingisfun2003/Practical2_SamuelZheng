import os
import json
import time
import numpy as np
import redis
import ollama
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from redis.commands.search.query import Query

os.environ["OLLAMA_NO_CUDA"] = "1"
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"

def select_option(prompt, options):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        choice = input("Enter the number of your choice: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please enter a valid choice.")

def get_embedding(text, model_name, st_model=None, instructor_model=None):
    if model_name == "nomic-embed-text":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    elif model_name == "mpnet":
        return st_model.encode(text, normalize_embeddings=True).tolist()
    elif model_name == "instructor-xl":
        instruction = "Represent the document for retrieval:"
        return instructor_model.encode([[instruction, text]], normalize_embeddings=True)[0].tolist()

def search_embeddings(query, embedding_model, st_model=None, instructor_model=None, top_k=5):
    query_embedding = get_embedding(query, embedding_model, st_model, instructor_model)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "vector_distance")
            .dialect(2)
        )
        results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})

        return [
            {
                "file": doc.file,
                "page": doc.page,
                "chunk": doc.chunk,
                "similarity": doc.vector_distance,
            }
            for doc in results.docs
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

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
    embedding_model = select_option("Choose an embedding model:", ["nomic-embed-text", "mpnet", "instructor-xl"])
    llm_model = select_option("Choose an LLM model:", ["mistral", "llama3"])

    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedding_model == "mpnet" else None
    instructor_model = INSTRUCTOR("hkunlp/instructor-xl") if embedding_model == "instructor-xl" else None

    print(f"\n🔍 Redis Search using '{embedding_model}' + '{llm_model}'")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ").strip()
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