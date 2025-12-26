import os
import pandas as pd
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

BATCH_SIZE = 10
SLEEP_TIME = 2
DB_PATH = "./chroma_db"
DATA_PATH = "data/books_cleaned.csv"

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY is missing.")

print("--- Starting Smart Database Builder ---")

try:
    books = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: '{DATA_PATH}' not found.")
    exit()

print("-> Initializing Gemini Embeddings...")
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

print("-> Checking existing database...")
db_books = Chroma(
    persist_directory=DB_PATH,
    embedding_function=gemini_embeddings
)


existing_ids = set(db_books.get()['ids'])
print(f"-> Database currently contains {len(existing_ids)} books.")

print("-> Preparing new documents...")
books["tagged_description"] = books["isbn13"].astype(str) + " " + books["description"]

new_documents = []
new_ids = []

for _, row in books.iterrows():
    isbn = str(row["isbn13"])

    if isbn in existing_ids:
        continue
        
    content = str(row["tagged_description"])
    
    new_documents.append(Document(page_content=content, metadata={"isbn": isbn}))
    new_ids.append(isbn)

print(f"   -> {len(new_documents)} new books need to be processed.")

if len(new_documents) == 0:
    print("All books are already in the database!")
    exit()

total_batches = (len(new_documents) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"-> Processing {total_batches} batches (Approx time: {total_batches * SLEEP_TIME / 60:.1f} minutes)...")

for i in range(0, len(new_documents), BATCH_SIZE):
    batch_docs = new_documents[i : i + BATCH_SIZE]
    batch_ids = new_ids[i : i + BATCH_SIZE]

    current_batch = (i // BATCH_SIZE) + 1
    
    try:
        db_books.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"   [Batch {current_batch}/{total_batches}] Added {len(batch_docs)} books. Sleeping {SLEEP_TIME}s...", end="\r")
        time.sleep(SLEEP_TIME)
        
    except Exception as e:
        print(f"\nError on Batch {current_batch}: {e}")
        print("-> Stopping script. Run again to resume exactly where you left off.")
        break

print(f"\n\nProcess Complete! Total books in DB: {len(db_books.get()['ids'])}")