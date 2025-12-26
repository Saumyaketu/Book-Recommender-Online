import pandas as pd
import numpy as np
import gradio as gr
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

if not os.path.exists("./chroma_db"):
    raise FileNotFoundError("The 'chroma_db' folder is missing.")

db_books = Chroma(
    persist_directory="./chroma_db",
    embedding_function=gemini_embeddings
)

books = pd.read_csv("data/books_with_emotions.csv")

# Thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.png",
    books["large_thumbnail"]
)

def recommend_books(query, category, tone):
    if not query.strip():
        return "<div class='no-results'>Please describe the book you are looking for above!</div>"

    recs = db_books.similarity_search(query, k=50)
    books_list = [int(rec.metadata["isbn"]) for rec in recs]

    # Filter by Category
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Sort by Tone
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    # HTML Cards
    if book_recs.empty:
        return "<div class='no-results'>No books found matching your criteria. Try different keywords!</div>"

    cards_html = "<div class='book-grid'>"
    
    for _, row in book_recs.head(16).iterrows():
        desc = row["description"]
        truncated_desc = " ".join(desc.split()[:25]) + "..."
        
        authors = row["authors"].split(";")
        authors_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."

        cards_html += f"""
        <div class="book-card">
            <div class="img-container">
                <img src="{row['large_thumbnail']}" alt="{row['title']}">
            </div>
            <div class="book-info">
                <h3>{row['title']}</h3>
                <p class="author">by {authors_str}</p>
                <p class="desc">{truncated_desc}</p>
            </div>
        </div>
        """
    
    cards_html += "</div>"
    return cards_html

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Custom CSS
custom_css = """
.gradio-container {
    max-width: 95% !important;
}

.book-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 25px;
    padding: 20px 0;
}

.book-card {
    background-color: #111827;
    border: 1px solid #374151;
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    display: flex;
    flex-direction: column;
}

.book-card:hover {
    transform: translateY(-5px);
    border-color: #6366f1; 
}

.img-container {
    height: 320px;
    width: 100%;
    background-color: #000000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    border-bottom: 1px solid #374151;
}

.img-container img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
}

.book-info {
    padding: 15px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}

.book-info h3 {
    margin: 0 0 5px 0;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f3f4f6;
    line-height: 1.3;
}

.book-info .author {
    font-size: 0.9rem;
    color: #9ca3af;
    font-style: italic;
    margin-bottom: 10px;
}

.book-info .desc {
    font-size: 0.85rem;
    color: #d1d5db;
    line-height: 1.4;
    margin-top: auto;
}

.no-results {
    text-align: center;
    padding: 50px;
    font-size: 1.2rem;
    color: #666;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Book Recommender") as dashboard:
    
    # Header
    with gr.Row():
        gr.Markdown("# AI Book Recommender")

    with gr.Row():
        # Left Sidebar (Filters)
        with gr.Column(scale=1):
            gr.Markdown("### Filters")
            category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="Tone", value="All")
            submit_btn = gr.Button("Find Books", variant="primary")

        # Right Main Area (Search + Results)
        with gr.Column(scale=4):
            user_query = gr.Textbox(
                label="What are you looking for?", 
                placeholder="e.g., A mystery set in Paris during the 1920s...",
                lines=1
            )
            
            output_html = gr.HTML(label="Recommendations")

    submit_btn.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_html
    )
    
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_html
    )

if __name__ == "__main__":
    dashboard.launch()