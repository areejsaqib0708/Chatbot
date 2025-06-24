import chromadb
from sentence_transformers import SentenceTransformer

# Load model
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")
client = chromadb.Client()

collection = client.get_or_create_collection(name="chatbot")

data = [
    "The sky is blue and beautiful.",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast and beans.",
    "I love green eggs, ham, sausages and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today.",
    "My name is Chatbot.",
    "I don't know your name.",
    "I can help you with questions and answers.",
    "I don't know how many brothers you have.",
    "You can ask me anything you like!",
    "I'm a bot created using sentence transformers and ChromaDB.",
    "You can exit anytime by typing 'bye', 'exit', or 'quit'."
]

# Add to ChromaDB
print("Encoding and saving data...")
collection.add(
    documents=data,
    ids=[f"id_{i}" for i in range(len(data))],
    embeddings=model.encode(data).tolist()
)
print("Data loaded. Start chatting!")

# Chat loop
while True:
    prompt = input("\t\t\tHow can I help you?\n>  ")
    if prompt.lower() in ['exit', 'quit', 'bye']:
        print("Bot: Goodbye!")
        break

    query_embedding = model.encode(prompt).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=2)

    if results["documents"][0]:
        print("Bot:")
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            similarity = 1 - dist
            print(f"{i + 1}. {doc} (Similarity: {similarity:.4f})")

    else:
        print("Bot: Sorry, I don't understand.")
