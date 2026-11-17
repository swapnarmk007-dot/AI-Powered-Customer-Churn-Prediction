from pathlib import Path

# RAG document directory
DOCUMENTS_PATH = Path(__file__).parent / "documents"


def load_documents():

    documents = []

    for file_path in DOCUMENTS_PATH.glob("*.txt"):

        with open(file_path, "r", encoding="utf-8") as file:

            content = file.read()

            documents.append({
                "filename": file_path.name,
                "content": content
            })

    return documents


if __name__ == "__main__":

    docs = load_documents()

    print("RAG Documents Loaded Successfully!")
    print("----------------------------------")

    for doc in docs:

        print("File:", doc["filename"])
        print("Content:", doc["content"][:100])
        print("----------------------------------")