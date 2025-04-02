import os
import time
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

os.environ["OLLAMA_NO_CUDA"] = "1"
COLLECTION_NAME = "pdf_chunks"
chroma_client = chromadb.PersistentClient(path="./chroma_store")

def select_option(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        choice = input("Enter the number of your choice: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please enter a valid choice.")

def get_embedding(text, model, st_model=None, instructor_model=None):
    if model == "nomic-embed-text":
        return ollama.embeddings(model=model, prompt=text)["embedding"]
    elif model == "mpnet":
        return st_model.encode(text, normalize_embeddings=True).tolist()
    elif model == "instructor-xl":
        instruction = "Represent the document for retrieval:"
        return instructor_model.encode([[instruction, text]], normalize_embeddings=True)[0].tolist()

def search_embeddings(query, model, collection, st_model=None, instructor_model=None, top_k=3):
    embedding = get_embedding(query, model, st_model, instructor_model)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    context_results = []
    for i in range(min(top_k, len(results["documents"][0]))):
        context_results.append(
            {
                "file": results["metadatas"][0][i]["file"],
                "page": results["metadatas"][0][i]["page"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "text": results["documents"][0][i],
                "similarity": results["distances"][0][i],
            }
        )
    return context_results

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
    if COLLECTION_NAME not in chroma_client.list_collections():
        print(f"Collection '{COLLECTION_NAME}' not found. Run the chroma ingestion script first.")
        return

    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    embedding_model = select_option("Choose an embedding model:", ["nomic-embed-text", "mpnet", "instructor-xl"])
    llm_model = select_option("Choose an LLM:", ["mistral", "llama3"])

    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") if embedding_model == "mpnet" else None
    instructor_model = INSTRUCTOR("hkunlp/instructor-xl") if embedding_model == "instructor-xl" else None

    print(f"\n🔍 Chroma Search using '{embedding_model}' + '{llm_model}'")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        context = search_embeddings(query, embedding_model, collection, st_model, instructor_model)
        response = generate_rag_response(query, context, llm_model)
        end_time = time.time()

        print(response)
        print(f"{end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    interactive_search()