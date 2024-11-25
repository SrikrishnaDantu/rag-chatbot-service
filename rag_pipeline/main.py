# main.py

from utils import initialize_vector_store, initialize_llm_chain, load_docs, split_docs
from pathlib import Path
import warnings
from uuid import uuid4
warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def get_answer(query, vector_store, chain):
    similar_docs = vector_store.similarity_search(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

def main():
    print("Welcome to the RAG-based Chatbot!")

    if not DATA_DIR.exists():
        print(f"Data directory not found at {DATA_DIR}. Please add your documents.")
        return

    documents = load_docs(str(DATA_DIR))

    print("Splitting documents into chunks...")
    docs = split_docs(documents)
    print(f"Number of chunks created: {len(docs)}")

    vector_store = initialize_vector_store()
    chain = initialize_llm_chain()
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    print("You can now ask questions about the content!")
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        print("Fetching answer...")
        answer = get_answer(query, vector_store, chain)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
